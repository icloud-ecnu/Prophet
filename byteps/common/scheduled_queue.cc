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

  // 原 B -> Mbits/sec => B * 1000000 (to bits/sec) / 1000 (to bits/millisecond)
//   * 8 (to Bytes/ms), 即 B *= 125
//  B *= (int)((double)batchsize / 64);
//  B *= 125;


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

    case PULL:
      if (BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetPullTable();
      }
      break;
    default:
      break;
  }
}

void BytePSScheduledQueue::addTask(std::shared_ptr<TensorTableEntry> entry) {
  std::lock_guard<std::mutex> lock(_mutex);
  //  BPS_LOG(INFO) << "addTask";
  if (!pre_run_result_sync) {
    if (!BytePSGlobal::pre_run && _qt == PUSH) {
      pre_run_result_sync = true;
      expected_priority = BytePSGlobal::total_grad - 1;
      _pointer = BytePSGlobal::_grad_checkpoint.size() - 1;
      BytePSGlobal::B = 125000;
      BPS_LOG(INFO) << "BytePSGlobal::B = " << BytePSGlobal::B;
      BPS_LOG(INFO) << "expected_priority = " << expected_priority;
      BPS_LOG(INFO) << "_pointer = " << _pointer;

      BPS_LOG(INFO)
          << "=====================_backward_exec=====================";
      for (int i = 0; i < BytePSGlobal::_backward_exec.size(); i++) {
        BPS_LOG(INFO) << BytePSGlobal::_backward_exec[i];
      }
      BPS_LOG(INFO)
          << "=====================_grad_checkpoint=====================";
      for (int i = 0; i < BytePSGlobal::_grad_checkpoint.size(); i++) {
        BPS_LOG(INFO) << BytePSGlobal::_grad_checkpoint[i];
      }
    }
  }
  if (BytePSGlobal::pre_run) {
    _sq.push_back(entry);
    if (_qt == PUSH && (entry->tensor_name).find(tensor_keywords) !=
                           (entry->tensor_name).npos) {
      int pr = entry->priority * -1;
      if (pr > BytePSGlobal::total_grad) {
        BytePSGlobal::total_grad = pr + 1;
      }
      auto now = std::chrono::system_clock::now();
      auto duration = now.time_since_epoch();
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
      long long tic = (long long)us.count();
      if (_grad_tic[pr] == 0) {
        processed_grad_count++;
        _grad_tic[pr] = tic;
      }
      if (processed_grad_count == BytePSGlobal::total_grad) {
        double avg = 0;
        for (int i = 1; i < processed_grad_count; i++) {
          double x = fabs(_grad_tic[i] - _grad_tic[i - 1]);
          avg = (((double)(i - 1)) / i) * avg + (((double)(1)) / i) * x;
        }
        avg *= 2;
        BytePSGlobal::_grad_checkpoint.push_back(-1);
        for (int i = 1; i < BytePSGlobal::total_grad; i++) {
          double diff = fabs(_grad_tic[i] - _grad_tic[i - 1]);
          if (diff > avg) {
            diff /= 1000;  // microsecond to millisecond
            if (BytePSGlobal::_backward_exec.size() == 0) {
              double _diff = fabs(_grad_tic[i - 1] - _grad_tic[0]);
              _diff /= 1000;
              BytePSGlobal::_backward_exec.push_back(_diff);
            }
            BytePSGlobal::_grad_checkpoint.push_back(i - 1);
            BytePSGlobal::_backward_exec.insert(
                BytePSGlobal::_backward_exec.begin(), diff);
          }
        }
        BytePSGlobal::_grad_checkpoint.push_back(BytePSGlobal::total_grad - 1);
      }
    }
  } else {
    if (_qt == PUSH && (entry->tensor_name).find(tensor_keywords) !=
                           (entry->tensor_name).npos) {
      _ms.insert(entry);
      _tensor_part[entry->priority * -1] = entry->total_partnum;
    } else {
      _sq.push_back(entry);
    }
  }
  BPS_CHECK(entry->tensor_name != "");
  BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                 << " addTask: " << entry->tensor_name << " key: " << entry->key
                 << " rank: " << BytePSGlobal::GetLocalRank();
  return;
}

void BytePSScheduledQueue::recorderTs(std::shared_ptr<TensorTableEntry> task) {
  auto context = task->context;
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

struct isTargetPriority {
  int Priority;

  isTargetPriority(int priority) : Priority(priority) {}

  bool operator()(std::shared_ptr<TensorTableEntry> x) {
    return x->priority == Priority;
  }
};

std::multiset<std::shared_ptr<TensorTableEntry>>::iterator
BytePSScheduledQueue::findTask(int priority) {
  if (_ms.size() == 0) {
    return _ms.end();
  }
  std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
  e->priority = priority;
  std::multiset<std::shared_ptr<TensorTableEntry>>::iterator it =
      _ms.lower_bound(e);
  if (it == _ms.end()) {
    return it;
  } else if ((*it)->priority != priority) {
    return _ms.end();
  } else {
    BPS_CHECK_EQ((*it)->priority, priority);
    return it;
  }
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask() {
  std::lock_guard<std::mutex> lock(_mutex);
  std::shared_ptr<TensorTableEntry> task;
  std::multiset<std::shared_ptr<TensorTableEntry>>::iterator msit;
  if (!BytePSGlobal::pre_run && _qt == PUSH && _ms.size() > 0) {
    if (!_dequeue) {
      msit = findTask(expected_priority * -1);
      if (msit == _ms.end()) {
        return nullptr;
      }
      if (!_visited[expected_priority]) {
        for (int x = 0; x < _tensor_part[expected_priority]; x++) {
          _mystack.push(expected_priority * -1);
          if (expected_priority == 0) {
            _meetzero = 1;
          }
        }
        _visited[expected_priority] = 1;
      }
      if (expected_priority >= 0) {
        expected_priority--;
      }
      if (expected_priority == BytePSGlobal::_grad_checkpoint[_pointer - 1]) {
        _dequeue = 1;
        dynamic_size = (long long)BytePSGlobal::_backward_exec[_sizepointer++] * BytePSGlobal::B;
      }
      return nullptr;
    } else {
      if (_mystack.size() == 0) {
        _dequeue = 0;
        if (_pointer > 0) {
          _pointer--;
        }
        _stagestart = 1;
        BytePSGlobal::pushsize[_sizepointer] = _mystack.top() + 1;
        return nullptr;
      }
      msit = findTask(_mystack.top());
      if (msit == _ms.end()) {
        return nullptr;
      }
      task = *msit;
      if (!_meetzero) {
        if (dynamic_size > task->len) {
          dynamic_size -= task->len;
          _ms.erase(msit);
          _mystack.pop();
        } else {
          _dequeue = 0;
          if (_pointer > 0) {
            _pointer--;
          }
          _stagestart = 1;
          BytePSGlobal::pushsize[_sizepointer] = _mystack.top() + 1;
          return nullptr;
        }
      } else if (_bps_credit < task->len) {
        return nullptr;
      } else if (_bps_credit > task->len) {
        _bps_credit -= task->len;
        _ms.erase(msit);
        _mystack.pop();
      }
      if (_mystack.empty() && _meetzero) {
        BPS_LOG(INFO) << "RESET";
        _pointer = BytePSGlobal::_grad_checkpoint.size() - 1;
        _dequeue = 0;
        expected_priority = BytePSGlobal::total_grad - 1;
        _stagestart = 1;
        _meetzero = 0;
        _sizepointer = 0;
        _dooropen = _door;
        _bps_credit = atoi(getenv("BPS_CREDIT"));
        for (int i = 0; i < 1600; i++) {
          _visited[i] = 0;
        }
      }
      task->ready_event = nullptr;
      recorderTs(task);
      return task;
    }
  } else {
    for (auto it = _sq.begin(); it != _sq.end(); ++it) {
      if ((*it)->ready_event) {
        if (!(*it)->ready_event->Ready()) {
          continue;
        }
      }
      if (_is_scheduled) {
        if ((*it)->len > _credits) continue;
      }
      if (_rt) {
        if (!_rt->IsKeyReady((*it)->key)) {
          continue;
        }
        _rt->ClearReadyCount((*it)->key);
      }
      task = *it;
      if (_is_scheduled) {
        _credits -= task->len;
      }
      _sq.erase(it);
      BPS_CHECK(task->tensor_name != "");
      BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                     << " getTask: " << task->tensor_name
                     << " key: " << task->key
                     << " rank: " << BytePSGlobal::GetLocalRank();
      task->ready_event = nullptr;
      recorderTs(task);
      if (BytePSGlobal::pre_run && _qt == PUSH) {
        int id = task->priority * -1;
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(duration);
        if (_push_start_tic[id] == 0) {
          _push_start_tic[id] = (long long)us.count();
        }
      }
      return task;
    }
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
    BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                   << " getTask(key): " << task->tensor_name
                   << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();
    task->ready_event = nullptr;
    recorderTs(task);
    return task;
  }
  return nullptr;
}

uint32_t BytePSScheduledQueue::pendingSize() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _sq.size();
}

void BytePSScheduledQueue::reportFinish(int size, int priority) {
  std::lock_guard<std::mutex> lock(_mutex);
  if (_is_scheduled) {
    _credits += size;
  }
  if (BytePSGlobal::pre_run && _qt == PUSH) {
    int id = priority * -1;
    if (!finish_tag[id]) {
      finish_count += 1;
      auto now = std::chrono::system_clock::now();
      auto duration = now.time_since_epoch();
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
      long long tac = (long long)us.count();
      double t = (double)(tac - _push_start_tic[id]);
      double possible_B = (double)size * 1000.0 / t;
//      possible_B *= 1; //TODO
      if (possible_B > (double)BytePSGlobal::B) {
        BytePSGlobal::B = (long long)possible_B;
        BPS_LOG(INFO) << "update B to " << BytePSGlobal::B;
      }
    }
    finish_tag[id] = true;
    if (finish_count == BytePSGlobal::total_grad) {
      BytePSGlobal::pre_run = false;
    }
  } else if (_qt == PUSH && size > 0 && _meetzero) {
    _bps_credit += size;
  }
  return;
}

}  // namespace common
}  // namespace byteps
