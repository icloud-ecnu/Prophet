// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "scheduled_queue.h"
#include <algorithm>
#include "global.h"
#include "logging.h"

namespace byteps {
namespace common {

BytePSScheduledQueue::BytePSScheduledQueue(QueueType type) {
  if (type == REDUCE && BytePSGlobal::GetNccl()->IsSignalRoot()) {
    _is_scheduled = true;
  } else {
    _is_scheduled = false;
  }

  size_t credit_in_partition = BytePSGlobal::GetNccl()->GetGroupSize() + 1;
  if (getenv("BYTEPS_SCHEDULING_CREDIT")) {
    credit_in_partition = atoi(getenv("BYTEPS_SCHEDULING_CREDIT"));
  }
  if (!credit_in_partition) {
    _is_scheduled = false;
  }

  _qt = type;
  _credits = _is_scheduled
              ? BytePSGlobal::GetPartitionBound() * credit_in_partition
              : 34359738368;  // 32GB, basically disabling credit control
  _rt = nullptr;

  switch (_qt) {
    case REDUCE:
      if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
        _rt = BytePSGlobal::GetReduceTable();
      }
      break;
    case PCIE_REDUCE:
      if (BytePSGlobal::IsCrossPcieSwitch()) {
        if (BytePSGlobal::GetCpuReducer()->isRoot()) {
          _rt = BytePSGlobal::GetPcieReduceTable();
        }
      }
      break;
    case PUSH:
      if (BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetPushTable();
      }
      _tensor_num=0;
      for(int i = 11; i >= 0; i--)
      {
        for(int j = _grad_checkpoint[i]; j <= _middle[i];j++){
            _myqueue.push(j * -1 );
            BPS_LOG(DEBUG) << " PUSH element into myqueue: " << j ;
        }
      }
      for(int i = 0 ; i <= 11; i++)
      {
        for(int j = _middle[i] + 1; j < _grad_checkpoint[i + 1] ; j++){
            _myqueue.push(j * -1);
            BPS_LOG(DEBUG) << " PUSH element into myqueue: " << j ;
        }
      }
      for(int i = 0;i < 160; i++)
      {
        _vis[i] = 0;
        _tensor_part[i] = 0;
      }
      break;
    case COPYH2D:
      if (!BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetCopyTable();
      }
      break;
    case BROADCAST:
      if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
        _rt = BytePSGlobal::GetBroadcastTable();
      }
      break;
    default:
      break;
  }
}

void BytePSScheduledQueue::addTask(std::shared_ptr<TensorTableEntry> entry) {
  std::lock_guard<std::mutex> lock(_mutex);
  _sq.push_back(entry);
  if (_is_scheduled) {
    // TODO: below can be optimized to O(n) using insertion sort
    std::sort(
        _sq.begin(), _sq.end(),
        [](std::shared_ptr<TensorTableEntry> a,
           std::shared_ptr<TensorTableEntry> b) {
          if (a->priority == b->priority) {
            return (a->key < b->key);  // from the first partition to the last
          }
          return (a->priority > b->priority);  // from higher priority to lower
        });
  }
  BPS_CHECK(entry->tensor_name != "");
  BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                 << " addTask: " << entry->tensor_name << " key: " << entry->key
                 << " rank: " << BytePSGlobal::GetLocalRank();
  return;
}

// Record the start time of the sub-tasks for all QueueTypes of each partition.
void BytePSScheduledQueue::recorderTs(std::shared_ptr<TensorTableEntry> task) {
  auto context = task->context;
  // add for profiling
  if (context->profile_flag) {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);

    auto &queue_list = task->queue_list;
    BPS_CHECK_GE(queue_list.size(), 1);
    auto this_op = queue_list[0];

    BPSCommTime *ret = new BPSCommTime;
    ret->start_t = (long long)(us.count());
    ret->key = task->key;
    ret->type = this_op; 
    context->part_comm_time[task->key][this_op].push(ret);
  }
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask() {
  std::lock_guard<std::mutex> lock(_mutex);
  std::shared_ptr<TensorTableEntry> task;
  // TODO: below can be optimized -- if we take task from the tail, erase() can
  // be faster
  for (auto it = _sq.begin(); it != _sq.end(); ++it) {
    if ((*it)->ready_event) {
      if (!(*it)->ready_event->Ready()) {
        continue;
      }
    }
    if (_is_scheduled) {
      if ((*it)->len > _credits) {
        continue;
      }
    }
    if (_rt) {
      if (!_rt->IsKeyReady((*it)->key)) {
        continue;
      }
      _rt->ClearReadyCount((*it)->key);
    }
    std::string tmp = (*it) -> tensor_name;
    if(_qt == PUSH && tmp.find("gradient") != tmp.npos)
    {
      if((*it) -> priority !=  _myqueue.front() && !_vis[(*it) -> priority * -1] && !_myqueue.empty() )
        continue;  
      
      BPS_LOG(TRACE) << "tensor is getting task: " << (*it) -> priority << "   myqueue top element " << _myqueue.front();

      BPS_LOG(TRACE) << "tensor_num is: " << _tensor_num << " total num is:" << (*it) -> total_partnum;    
     _tensor_part[ (*it) -> priority * -1]++;      
     if(_tensor_part[ (*it) -> priority * -1 ] == (*it) -> total_partnum ) //we cannot initialize the _vis and _myqueue immediately, cause some tensors may not be transferred over.
     	_tensor_num++;
      if( !_vis[_myqueue.front()])
        _myqueue.pop();// pop the firt element when the tensor first came.
      _vis[(*it) -> priority * -1] = 1;
   }
    task = *it;
    _sq.erase(it);
    if (task -> priority == 0) {
      _meetzero = 1;
    }
    if (_meetzero) {
      BPS_LOG(INFO) << "[R] After meet zero, try " << task->tensor_name << " key: " << task->key
                   << " dooropen: " << _dooropen;
      if (_dooropen) {
        _dooropen = 0;
      } else {
        break;
      }
    } else {
      if (_is_scheduled) {
        _credits -= task->len;
      }
    }

    //all push process end in this iteration , then reinitalize varibles.
    if(_tensor_num == 157 && _myqueue.empty())
    {
       _meetzero = 0;
       _dooropen = 1;
       _tensor_num = 0;
       for(int i = 0; i < 160; i++)_tensor_part[i] = 0;
       for(int i = 0;i < 160; i++) _vis[i] = 0;
  
      for(int i = 11; i >= 0; i--)
      {
        for(int j = _grad_checkpoint[i]; j <= _middle[i];j++){
            _myqueue.push(j * -1 );
            //BPS_LOG(INFO) << " PUSH element into myqueue: " << _grad_checkpoint[i] + j ;
        }
      }
      for(int i = 0 ; i <= 11; i++)
      {
        for(int j = _middle[i] + 1; j < _grad_checkpoint[i + 1] ; j++){
            _myqueue.push(j * -1);
            //BPS_LOG(INFO) << " PUSH element into myqueue: " << j ;
        }
      }
    }

    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                   << " getTask: " << task->tensor_name << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();
    task->ready_event = nullptr;
    // Add for profiling communication traces
    recorderTs(task);
    return task;
  }
  return nullptr;
}


std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask(uint64_t key) {
  BPS_CHECK(!_is_scheduled);
  std::lock_guard<std::mutex> lock(_mutex);
  std::shared_ptr<TensorTableEntry> task;
  for (auto it = _sq.begin(); it != _sq.end(); ++it) {
    if ((*it)->ready_event) {
      BPS_CHECK((*it)->ready_event->Ready());
    }
    if ((*it)->key != (uint64_t)key) {
      continue;
    }
    task = *it;
    _sq.erase(it);

    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                   << " getTask(key): " << task->tensor_name
                   << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();
    task->ready_event = nullptr;
    // Add for profiling communication traces
    recorderTs(task);
    return task;
  }
  return nullptr;
}

uint32_t BytePSScheduledQueue::pendingSize() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _sq.size();
}

void BytePSScheduledQueue::reportFinish(int size) {
  std::lock_guard<std::mutex> lock(_mutex);
  if (_meetzero) {
    _dooropen = 1;
  } else {
    if (_is_scheduled) {
      _credits += size;
    }
  }
  return;
}

}  // namespace common
}  // namespace byteps
