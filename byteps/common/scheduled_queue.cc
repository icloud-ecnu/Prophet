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
#include <queue>

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
                    // _sizepointer=1;
                    break;
                default:
                    break;
            }
        }

        void BytePSScheduledQueue::addTask(std::shared_ptr <TensorTableEntry> entry) {
            std::lock_guard <std::mutex> lock(_mutex);
            _sq.push_back(entry);
            if (_is_scheduled) {
                // TODO: below can be optimized to O(n) using insertion sort
                std::sort(
                        _sq.begin(), _sq.end(),
                        [](std::shared_ptr <TensorTableEntry> a,
                           std::shared_ptr <TensorTableEntry> b) {
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
        void BytePSScheduledQueue::recorderTs(std::shared_ptr <TensorTableEntry> task) {
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

        std::shared_ptr <TensorTableEntry> BytePSScheduledQueue::getTask() {
            std::lock_guard <std::mutex> lock(_mutex);
            std::shared_ptr <TensorTableEntry> task;
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
                std::string tmp = (*it)->tensor_name;
                task = *it;

                if (_qt == PUSH && tmp.find("gradient") != tmp.npos)
                {
                    BPS_LOG(INFO) << "IN PUSH";
                    if (_dequeue != 1) {
                        BPS_LOG(INFO) << "task->priority:" << task->priority << " stagestart: " << _stagestart
                                       << " _grad_checkpoint[_pointer]:" << _grad_checkpoint[_pointer];

                        if ((task->priority == -1 * _grad_checkpoint[_pointer] && _stagestart) || (!_mystack.empty() && task->priority > -1 * _grad_checkpoint[_pointer] && task->priority < -1 * _grad_checkpoint[_pointer - 1] && task->priority == _mystack.top() + 1)) {
                            if (task->priority == -1 * _grad_checkpoint[_pointer]) {
                                _stagestart = 0;
                            }
                            int part = 0;
                            for (part = 0; part < task->total_partnum; part++) {
                                _mystack.push(task->priority);
                            }
                            _vis[task->priority * -1] = 1;
                            total_part += task->total_partnum;
                            how_many += 1;
                            BPS_LOG(INFO) << "how many=" << how_many;
                            BPS_LOG(INFO) << "ENQUEUE1 element: " << task->priority << " for " << task->total_partnum << " parts";
                        }
                        if (_vis[task->priority * -1]) {
                            BPS_LOG(INFO) << "pq push " << task->priority;
                            pq.push(task);
                            _sq.erase(it);
                            it--;
                        }
                        BPS_LOG(INFO) << "how_many:" << how_many << "," << "should be:" << (_grad_checkpoint[_pointer] - _grad_checkpoint[_pointer - 1]) << ", total_part: " << total_part
                                       << " pq.size():" << pq.size();
                        if (how_many == _grad_checkpoint[_pointer] - _grad_checkpoint[_pointer - 1] && total_part == pq.size()) {
                            _dequeue = 1;
                            dynamic_size = _execution[_sizepointer++];
                            _pointer--;
                            BPS_LOG(INFO) << "enqueue operation of one stage is over." << "_sizepointer:";
                            break;
                        }
                        continue;
                    } else {
                        if (pq.size() == 0) {
                            _dequeue = 0;
                            BPS_LOG(INFO) << "Clear.";
                            _pointer = 12;
                            _stagestart = 1;
                            _meetzero = 0;
                            _sizepointer = 0;
                            _dooropen = 11;
                            how_many = 0;
                            total_part = 0;
                            for (int i = 0; i < 160; i++) {
                              _vis[i] = 0;
                            }
                            break;
                        }
                        task = pq.top();
                        if (task->priority == 0) {
                            _meetzero = 1;
                            BPS_LOG(INFO) << "Meet zero.";
                        }
                        if (!_meetzero) {
                            if (dynamic_size > task->len) {
                                dynamic_size -= task->len;
                                BPS_LOG(INFO) << "dequeue element: " << task->tensor_name << "dynamic size now is: "
                                               << dynamic_size;
                                pq.pop();
                                _mystack.pop();
                                BPS_LOG(INFO) << "PUSH gradient before 0: " << tmp;
                            } else {   //nxet stage enstack could begin.
                                _dequeue = 0;
                                _pointer--;
                                _stagestart = 1;
                                BPS_LOG(INFO) << "No left size. Waiting for next gradient block.";
                                break;
                            }
                        } else if (!_dooropen) {//we cannot change the value of tensor_part if door is closed.
                            BPS_LOG(INFO) << "door is closed.";
                            break;
                        } else {
                            _dooropen--;
                            // dynamic_size -= task -> len;  // if meetzero, dynamic size is no meaning.
                            pq.pop();
                            _mystack.pop();
                            BPS_LOG(INFO) << "PUSH gradient after 0: " << tmp;
                        }
                        task->ready_event = nullptr;
                        // Add for profiling communication traces
                        recorderTs(task);
                        return task;
                    }
                }
                if (_is_scheduled) {
                    _credits -= task->len;
                }
                _sq.erase(it);
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


        std::shared_ptr <TensorTableEntry> BytePSScheduledQueue::getTask(uint64_t key) {
            BPS_CHECK(!_is_scheduled);
            std::lock_guard <std::mutex> lock(_mutex);
            std::shared_ptr <TensorTableEntry> task;
            for (auto it = _sq.begin(); it != _sq.end(); ++it) {
                if ((*it)->ready_event) {
                    BPS_CHECK((*it)->ready_event->Ready());
                }
                if ((*it)->key != (uint64_t) key) {
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
            std::lock_guard <std::mutex> lock(_mutex);
            return _sq.size();
        }

        void BytePSScheduledQueue::reportFinish(int size) {
            std::lock_guard <std::mutex> lock(_mutex);
            if (_is_scheduled) {
                _credits += size;
            }
            if (_qt == PUSH) //|| _qt == PULL)
            {
                if (_meetzero) {
                    if (_dooropen < 11)
                        _dooropen++;
                }
                // BPS_LOG(DEBUG) << "door open value:" << _dooropen;
            }
            return;
        }

    }  // namespace common
}  // namespace byteps