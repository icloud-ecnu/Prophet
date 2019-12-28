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
      //BPS_LOG(DEBUG) << "IN PUSH: " << _is_scheduled ;
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
      // dynamic_size = _backward_exec[_sizepointer + 1];
      _sizepointer=1;
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
  BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
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
      if ((*it)->len > _credits)
        continue;
    }
    if (_rt) {
      if (!_rt->IsKeyReady((*it)->key)) {
        continue;
      }
      _rt->ClearReadyCount((*it)->key);
    }
    std::string tmp = (*it) -> tensor_name;
    task = *it;
  //  BPS_LOG(DEBUG) << _qt << " tensor name: " << tmp;

    if( _qt == PUSH && tmp.find("gradient") != tmp.npos ) 
    {
      
          /////first  enqueue as the gradient block coming, then dequeue dynamically.
        if(_dequeue != 1){
          BPS_LOG(DEBUG) << "Position 1" << " pointer: " <<  _pointer <<" stagestart: " << _stagestart << " mystack empty:" <<  _mystack.empty() \
                << "task name: " << task -> tensor_name ; 
 
          bool taskisstart = task -> priority == -1 * _grad_checkpoint[_pointer]  && _stagestart ;
          bool taskisproc = !_mystack.empty() && task -> priority > -1 * _grad_checkpoint[_pointer] \ 
                    && task -> priority  < -1 * _grad_checkpoint[_pointer - 1] \
                    && task -> priority == _mystack.top() + 1;
          bool starttagged = _stagestart && _tensor_part[_grad_checkpoint[_pointer]] ;
          bool proctagged = !_mystack.empty() && _tensor_part[(_mystack.top() + 1) * -1] \
                    && _mystack.top() + 1 > -1 * _grad_checkpoint[_pointer] \ 
                    && _mystack.top() + 1  < -1 * _grad_checkpoint[_pointer - 1];                                    ;
          if(!_mystack.empty() && _mystack.top() == -156) 
            BPS_LOG(DEBUG) << "proctagged when top is -156: " << proctagged; 
          if( taskisstart || taskisproc || starttagged || proctagged)
          {
            if(starttagged)
              for(int x = 0; x < _tensor_part[_grad_checkpoint[_pointer]]; x++){
                _mystack.push(_grad_checkpoint[_pointer] * -1);
                _stagestart = 0;
                BPS_LOG(DEBUG) << "ENQUEUE at start element not firstly: " << _grad_checkpoint[_pointer] * -1 << " mystack size: " << _mystack.size() ;
              }
            
            else if(proctagged){
              int tmp = _mystack.top() + 1;
              for(int x = 0; x < _tensor_part[tmp * -1]; x++){
                _mystack.push(tmp);
                BPS_LOG(DEBUG) << "ENQUEUE in proc element not firstly: " << tmp  << " mystack size: " << _mystack.size();
              }
            }

            else {
              if(taskisstart) _stagestart = 0; 
              _tensor_part[task -> priority * -1] = task -> total_partnum;
              for(int x = 0; x< task -> total_partnum; x++){
                _mystack.push(task -> priority);
                BPS_LOG(DEBUG) << "ENQUEUE element firstly: " << task -> priority ;
              }
            }
            if(!_mystack.empty() &&  _mystack.top() * -1 == _grad_checkpoint[_pointer - 1] + 1 )
            {
                _dequeue = 1;
                dynamic_size = _backward_exec[_sizepointer++];               
                BPS_LOG(DEBUG) << "enqueue operation of one stage is over." << "_sizepointer:" << _sizepointer << "mystack top is: " << _mystack.top();
                break;
                ///////////////////////////initialize dynamic size of this gradient stage.////////////////////////////
            }
          }
          // BPS_LOG(DEBUG) << "Position 4:"  << "_sq size is: "<< _sq.size();
  
          continue;
        }            
        // Size = Bandwidth * exectime , size decreased by the pop operation of mystack.
       // BPS_LOG(DEBUG) << "Task: " <<  task-> priority << "I have meet zero: " << _meetzero << " and door is open: " << _dooropen;
        if(task -> priority == 0) {
          _meetzero = 1;
         BPS_LOG(DEBUG) << "Meet zero." << "my stack size: " << _mystack.size();
         }
        if(!_meetzero)
        {
            if(task -> priority !=  _mystack.top())continue; 
            if(dynamic_size > task -> len){
              dynamic_size -= task -> len;
              BPS_LOG(DEBUG) << "dequeue element: " << task -> tensor_name << "dynamic size now is: " << dynamic_size;
              _sq.erase(it);
              _mystack.pop();
              BPS_LOG(DEBUG) << "PUSH gradient before 0: " << tmp ;
            }
            else{   //nxet stage enstack could begin.
              _dequeue = 0;
              _pointer--;
              _stagestart = 1;
              BytePSGlobal::pushsize[_sizepointer] = _mystack.top() + 1;
              //BPS_LOG(TRACE) << "PUSH: No left size. Waiting for next gradient block." << "intilized global pushsize" << _sizepointer << ": " << BytePSGlobal::pushsize[_sizepointer];
              break;  
            }      
        }
        else if(!_dooropen) {//we cannot change the value of tensor_part if door is closed.
          BPS_LOG(DEBUG) << "push door is closed.";
          break;
        }
        else {         
          if(task -> priority !=  _mystack.top())continue;
            _dooropen--;
            // dynamic_size -= task -> len;  // if meetzero, dynamic size is no meaning.
            _sq.erase(it);
            _mystack.pop();
            BPS_LOG(DEBUG) << "PUSH gradient after 0: " << tmp ;
            // BPS_LOG(DEBUG) << "The door has been closed.";
        }
        //  BPS_LOG(DEBUG) << "transferred tensor num: " << _tensor_num  << "  empty: " << _mystack.empty() << " size of myqueue: " << _mystack.size();
        if(_mystack.empty() && _meetzero)
        {
          BPS_LOG(DEBUG) << "Clear.";
          _dequeue = 0;

          _pointer = 12;
          _stagestart = 1;
          _meetzero = 0;
          _sizepointer = 0;
          _dooropen = 11;
          //  _pushsize = 0;
        //  _sizepointer =  _qt == PUSH ? 0:1;
        }
        task->ready_event = nullptr;
        // Add for profiling communication TRACEs
        recorderTs(task);
        return task;
    }
///pull process scheduler
//before meetzero, push block size should be equal with the execution time of next gradient block, 
//           while pull block size should be equal with the execution time of the next next gradient block.
//after meetzero, push should utilize full bandwidth,
//          while pull should overlap the forward execution time.

    if(_qt == PULL && tmp.find("gradient") != tmp.npos )
    { 
        if(_dequeue != 1 && _sizepointer < 13 ){
          // BPS_LOG(DEBUG) << "Position 1" << " pointer: " <<  _pointer <<" stagestart: " << _stagestart << " mystack empty:" <<  _mystack.empty() \
          //       << "task name: " << task -> tensor_name;
          bool taskisstart = task -> priority == -1 * _grad_checkpoint[_pointer]  && _stagestart ;
          bool taskisproc = !_mystack.empty() && task -> priority > -1 * _grad_checkpoint[_pointer] \ 
                    && task -> priority  < -1 * _grad_checkpoint[_pointer - 1] \
                    && task -> priority == _mystack.top() + 1;
          bool starttagged = _stagestart && _tensor_part[_grad_checkpoint[_pointer]] ;
          bool proctagged = !_mystack.empty() && _tensor_part[(_mystack.top() + 1) * -1] \
                    && _mystack.top() + 1 > -1 * _grad_checkpoint[_pointer] \ 
                    && _mystack.top() + 1  < -1 * _grad_checkpoint[_pointer - 1];                                    ;
          // if(!_mystack.empty() && _mystack.top() == -156) 
          //   BPS_LOG(DEBUG) << "proctagged when top is -156: " << proctagged; 
          if( taskisstart || taskisproc || starttagged || proctagged)
          {
            if(starttagged)
              for(int x = 0; x < _tensor_part[_grad_checkpoint[_pointer]]; x++){
                _mystack.push(_grad_checkpoint[_pointer] * -1);
                _stagestart = 0;
                BPS_LOG(TRACE) << "PULL: ENQUEUE at start element not firstly: " << _grad_checkpoint[_pointer] * -1 << " mystack size: " << _mystack.size() << "  sizepointer: " << _sizepointer;
              }
            
            else if(proctagged){
              int tmp = _mystack.top() + 1;
              for(int x = 0; x < _tensor_part[tmp * -1]; x++){
                _mystack.push(tmp);
                BPS_LOG(TRACE) << "PULL: ENQUEUE in proc element not firstly: " << tmp  << " mystack size: " << _mystack.size() << "  sizepointer: " << _sizepointer;
              }
            }
            else {
              if(taskisstart) _stagestart = 0; 
              _tensor_part[task -> priority * -1] = task -> total_partnum;
              for(int x = 0; x< task -> total_partnum; x++){
                _mystack.push(task -> priority);
                BPS_LOG(TRACE) << "PULL: ENQUEUE element firstly: " << task -> priority << "  sizepointer: " << _sizepointer ;
              }
            }

               // BPS_LOG(TRACE) << "top=" << _mystack.top() << " ckpt" << (_grad_checkpoint[_pointer - 1] + 1 );
            if(!_mystack.empty() &&  _mystack.top() * -1 == _grad_checkpoint[_pointer - 1] + 1 )
            {
                _dequeue = 1;
                // dynamic_size = _backward_exec[_sizepointer] > _backward_exec[_sizepointer - 1] ? _backward_exec[_sizepointer - 1] : _backward_exec[_sizepointer];               
                dynamic_size = _backward_exec[_sizepointer++];
                BPS_LOG(TRACE) << "PULL: enqueue operation of one stage is over." << "  _sizepointer:" << _sizepointer << "  mystack top is: " << _mystack.top();
                break;
                ///////////////////////////initialize dynamic size of this gradient stage.////////////////////////////
            }
          }
          continue;
        }  
    
      if(_sizepointer < 13)
        {
            if(task -> priority !=  _mystack.top())
              continue;
            // _noleftsize = 1;
            BPS_LOG(TRACE) << "priority=" << task->priority << ", top=" << _mystack.top() << ", line=" << BytePSGlobal::pushsize[_sizepointer - 1]
            << " size=" << dynamic_size << " len=" << task->len;
            if(dynamic_size > task -> len && task -> priority > BytePSGlobal::pushsize[_sizepointer - 1]){
              dynamic_size -= task -> len;
              //BPS_LOG(TRACE) << "PULL: dequeue element: " << task -> tensor_name << "dynamic size now is: " << dynamic_size \
                  << "low bound is:" << BytePSGlobal::pushsize[_sizepointer - 1] ;
              _sq.erase(it);
              _mystack.pop();
              // _pullsize++;
              // BPS_LOG(TRACE) << "PULL SIZE IS: " << _pullsize << " push size is:" << BytePSGlobal::pushsize[_sizepointer - 1];
              // if(!_sq.size() && _pullsize == BytePSGlobal::pushsize[_sizepointer - 1]){
              //   _dequeue = 0; //&& dynamic_size < _backward_exec[]
              //   _pointer--;
              //   _stagestart = 1;                  
              //   _pullsize = 0;         
              //   BPS_LOG(TRACE) << "PULL: size is redundant, waiting...";
              // }
              // BPS_LOG(TRACE) << "PULL: gradient before 0: " << tmp << "meet zero: " << _meetzero;
            }
            else{   //nxet stage enstack could begin.
              _dequeue = 0;
              _pointer--;
              _stagestart = 1;
              // _pullsize = 0;
              BPS_LOG(TRACE) << "PULL: No left size. Waiting for next gradient block.";
              break;  
            }      //maybe no left size, but has no time to search the next one before meetzero;
        }
        else {
            if(!_mystack.empty() && task -> priority != _mystack.top())continue;
            if(!_pulldoor) {
              forward_dynamic_size = _forward_exec[_exec_stage];
              _stagepullnum = 0;
              BPS_LOG(TRACE) << "exec_stage: " << _exec_stage << " initilized." << "  beginning dynamic size:"<< forward_dynamic_size;
            }
            if(!_mystack.empty() && (forward_dynamic_size > task -> len || _exec_stage > 12)){
              _sq.erase(it);
              _mystack.pop();
              forward_dynamic_size -= task -> len;
              _pulldoor++;
              BPS_LOG(TRACE) << "PULL: dequeue after zero: " << task -> tensor_name << "  _exec_stage is:" << _exec_stage << "  forward dynamic size:" \
                  << forward_dynamic_size << "  pull door val is:" <<  _pulldoor;
            }

            else if(!_mystack.empty() && _mystack.top() >= -1 * _grad_checkpoint[_exec_stage + 1]){
              _sq.erase(it);
              _mystack.pop();
              _pulldoor++;
              BPS_LOG(TRACE) << "PULL: dequeue after zero enforced: " << task -> tensor_name << "  _exec_stage is:" << _exec_stage <<  "   forward dynamic size:"  \
                << forward_dynamic_size << "  pull door val is:" <<  _pulldoor;
            }

            else{
              if(!_stagepullnum && _pulldoor){
                _stagepullnum = _pulldoor;
                BPS_LOG(TRACE) << "initilize stagepullnum at stage "<< _exec_stage << ":  " << _stagepullnum;
              }
              break;
            } 
          // BPS_LOG(DEBUG) << "PULL door is closed.";
          // break;
          }
      if(_sizepointer == 13 && !_stagepullnum &&_mystack.empty())//reset parameter
      {
        BPS_LOG(TRACE) << "Clear.";
        _dequeue = 0;
        _pointer = 12;
        _stagestart = 1;
        _meetzero = 0;
        _sizepointer = 1;//different from push process
        _exec_stage = 0;
        _stagepullnum = 0;
        _pulldoor=0;
      }  
      task->ready_event = nullptr;
      // Add for profiling communication TRACEs
      recorderTs(task);
      return task;
    }
 
//Detail: meetzero, pull the first stage immediately. When the stage1 is running, we could transfer `S = forward_stage1_time * bandwidth` data.
//      pull_stack pop over under limit of S, if  pull_stack.top() > stage2_end ; else pull_stack.pop() until pull_stack.top() > stage2_end.
//      num++ as pull_stack pop elements. _pulldoor = num;
//      if _pulldoor = 0, stage++.

    // }


    if (_is_scheduled) 
    {
        _credits -= task->len;
    }
    _sq.erase(it);
    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                   << " getTask: " << task->tensor_name << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();  
    task->ready_event = nullptr;
    // Add for profiling communication TRACEs
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
    BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                   << " getTask(key): " << task->tensor_name
                   << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();
    task->ready_event = nullptr;
    // Add for profiling communication TRACEs
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
  if (_is_scheduled) {
      _credits += size;
  }
  if(_qt == PUSH ) //) || _qt == PULL
  {
    if(_meetzero) {
         if(_dooropen < 11)
              _dooropen++;
         }       
         // BPS_LOG(DEBUG) << "door open value:" << _dooropen;
  }
  if(_qt == PULL)
  {
    if(_stagepullnum > 0){
      _stagepullnum--;
      BPS_LOG(TRACE) << "PULL PROCESS FINISH: _stagepullnum value is:" << _stagepullnum;
      if(!_stagepullnum){
        _exec_stage++;
        _pulldoor = 0;
        BPS_LOG(TRACE) << "STAGE PULL PROCESS FINISH: stage is:" << _exec_stage - 1;
      }
    }

  }
  return;
}

}  // namespace common
}  // namespace byteps
