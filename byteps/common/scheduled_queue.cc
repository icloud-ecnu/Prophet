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
                    _tensor_num = 0;
                    for (int i = 11; i >= 0; i--) {
                        for (int j = _grad_checkpoint[i]; j <= _middle[i]; j++) {
                            _myqueue.push(j * -1);
                            BPS_LOG(DEBUG) << " PUSH element into myqueue: " << j;
                        }
                    }
                    for (int i = 0; i <= 11; i++) {
                        for (int j = _middle[i] + 1; j < _grad_checkpoint[i + 1]; j++) {
                            _myqueue.push(j * -1);
                            BPS_LOG(DEBUG) << " PUSH element into myqueue: " << j;
                        }
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
            _sq.push(entry);
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
                ret->start_t = (long long) (us.count());
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
      while(!_sq.empty())
      {
                task = _sq.top();
                if (task->ready_event) {
                    if (!task->ready_event->Ready()) {
                        return nullptr;
                    }
                }
                if (_is_scheduled) {
                    if (task->len > _credits) {
                        return nullptr;
                    }
                }
                if (_rt) {
                    if (!_rt->IsKeyReady(task->key)) {
                        return nullptr;
                    }
                    _rt->ClearReadyCount(task->key);
                }
                std::string tmp = task->tensor_name;
                if (_qt == PUSH && tmp.find("gradient") != tmp.npos) {
                  if (_rest_part == 0) {
                        if (task->priority == 0) {
                            _meetzero = 1;
                        }
                        if (!_meetzero || (_meetzero && _dooropen)) {
                            _rest_part = task->total_partnum - 1;
                            BPS_LOG(INFO) << task->tensor_name << " has " << _rest_part << " parts left.";
                            if (_rest_part == 0) {
                              _tensor_num++;
                              BPS_LOG(INFO) << _tensor_num << " done.";
                            }
                            if (_meetzero) {
                                //BPS_LOG(INFO) << "close door";
                                _dooropen = 0;
                            }
                        }
                        else {
                            // here, _door must be closed, skip
                            //BPS_LOG(INFO) << "door is closed, skip";
                            break;
                        }
                   // return task;
                   _sq.pop();
                  } else {
                      _rest_part--;
                      BPS_LOG(INFO) << task->tensor_name << " still has " << _rest_part << " parts left.";
                      if (_rest_part == 0) {
                        _tensor_num++;
                          BPS_LOG(INFO) << _tensor_num << " done.";
                      }
                  }
              //all push process end in this iteration , then reinitalize varibles.
                if (_tensor_num == 157 && _myqueue.empty()) {
                    BPS_LOG(INFO) << "Clear";
                    _meetzero = 0;
                    _dooropen = 1;
                    _tensor_num = 0;
                }
                task->ready_event = nullptr;
                recorderTs(task);
                return task;
               }

              if (_is_scheduled) {
                  _credits -= task->len;
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
            // for (auto it = _sq.begin(); it != _sq.end(); ++it) {
            //     if ((*it)->ready_event) {
            //         BPS_CHECK((*it)->ready_event->Ready());
            //     }
            //     if ((*it)->key != (uint64_t) key) {
            //         continue;
            //     }
            //     task = *it;
            //     _sq.erase(it);
            while(!_sq.empty())
            {
              task = _sq.top();
              if ((*it)->ready_event) {
                    BPS_CHECK((*it)->ready_event->Ready());
                }
                if ((*it)->key != (uint64_t) key) {
                    continue;
                }
                _sq.pop();
            }
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
                BPS_LOG(INFO) << "door open again";
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
