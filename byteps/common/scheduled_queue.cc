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
                    _sizepointer = 1;
                    break;
                default:
                    break;
            }
        }

        void BytePSScheduledQueue::addTask(std::shared_ptr <TensorTableEntry> entry) {
            std::lock_guard <std::mutex> lock(_mutex);
            if (_qt == PUSH && (entry->tensor_name).find("gradient") != (entry->tensor_name).npos) {
                BPS_LOG(DEBUG) << "insert to _ms:" << entry->tensor_name;
                _ms.insert(entry);
            } else {
                BPS_LOG(DEBUG) << "push_back to _sq:" << entry->tensor_name;
                _sq.push_back(entry);
            }
            BPS_CHECK(entry->tensor_name != "");
            BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                           << " addTask: " << entry->tensor_name << " key: " << entry->key
                           << " rank: " << BytePSGlobal::GetLocalRank();
            return;
        }

        void BytePSScheduledQueue::recorderTs(std::shared_ptr <TensorTableEntry> task) {
            auto context = task->context;
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

        struct isTargetPriority {
            int Priority;

            isTargetPriority(int priority) : Priority(priority) {}

            bool operator()(std::shared_ptr <TensorTableEntry> x) {
                BPS_LOG(DEBUG) << "now comparing " << x->priority  << " and " << Priority << "x name is:" << x -> tensor_name;
                return x->priority == Priority;
            }
        };

        std::multiset < std::shared_ptr < TensorTableEntry >> ::iterator BytePSScheduledQueue::findTask(int priority) {
            //BPS_LOG(INFO) << "finding priority=" << priority << " in " << _ms.size() << " _ms.";
            std::shared_ptr<TensorTableEntry> e(new TensorTableEntry);
            e->priority = priority;

            std::multiset < std::shared_ptr < TensorTableEntry >> ::iterator
            it = _ms.lower_bound(e);
            if (it == _ms.end()) {
                //BPS_LOG(INFO) << "not found"; // TODO if exists bug
                return it;
            } else if ((*it)->priority != priority) {
               // BPS_LOG(INFO) << "(*it)=" << (*it)->priority << ", ignore.";
                return _ms.end();
            } else {
                BPS_CHECK_EQ((*it)->priority, priority);
                //BPS_LOG(INFO) << "found.";
                return it;
            }
        }

        std::shared_ptr <TensorTableEntry> BytePSScheduledQueue::getTask() {
            std::lock_guard <std::mutex> lock(_mutex);
            std::shared_ptr <TensorTableEntry> task;
            std::multiset < std::shared_ptr < TensorTableEntry >> ::iterator msit;
            if (_sq.size() > 0 || _ms.size() > 0)
                BPS_LOG(DEBUG) << "In getTask(" << _qt << "), _sq size=" << _sq.size() << " and _ms size=" << _ms.size();
            if (_qt == PUSH && !_dequeue && _ms.size() > 0) {
                BPS_LOG(DEBUG) << "Call findTask() with " << (expected_priority * -1);
                msit = findTask(expected_priority * -1);
                if (msit == _ms.end()) {
                    return nullptr;
                }
                task = *msit;
                for (int x = 0; x < task->total_partnum; x++) {
                    _mystack.push(task->priority);
                }
                expected_priority--;
                if (expected_priority == _grad_checkpoint[_pointer - 1]) {
                    _dequeue = 1;
                    dynamic_size = _backward_exec[_sizepointer++];
                }
                return nullptr;
            }
            if (_qt == PUSH && _ms.size() > 0) {
                BPS_LOG(DEBUG) << "ignore it, try msit.";
                msit = findTask(_mystack.top());
                if (msit == _ms.end()) {
                    return nullptr;
                }
                task = *msit;
                if (task->priority == 0) {
                    _meetzero = 1;
                    BPS_LOG(DEBUG) << "Meet zero.";
                }
                if (!_meetzero) {
                    if (dynamic_size > task->len) {
                        dynamic_size -= task->len;
                        BPS_LOG(DEBUG) << "dequeue element: " << task->tensor_name << "dynamic size now is: "
                                      << dynamic_size;
                        _ms.erase(msit);
                        _mystack.pop();
                        BPS_LOG(DEBUG) << "PUSH gradient before 0: " << task->tensor_name;
                    } else {
                        BPS_LOG(DEBUG) << "No left space";
                        _dequeue = 0;
                        _pointer--;
                        _stagestart = 1;
                        BytePSGlobal::pushsize[_sizepointer] = _mystack.top() + 1;
                        return nullptr;
                    }
                } else if (!_dooropen) {
                    BPS_LOG(DEBUG) << "push door is closed.";
                    return nullptr;
                } else {
                    msit = findTask(_mystack.top());
                    if (msit == _ms.end()) {
                        return nullptr;
                    }
                    task = *msit;
                    _dooropen--;
                    _ms.erase(msit);
                    _mystack.pop();
                    BPS_LOG(DEBUG) << "PUSH gradient after 0: " << task->tensor_name;
                }
                if (_mystack.empty() && _meetzero) {
                    BPS_LOG(DEBUG) << "Clear.";
                    _dequeue = 0;
                    _pointer = 12;
                    expected_priority = _grad_checkpoint[_pointer];
                    _stagestart = 1;
                    _meetzero = 0;
                    _sizepointer = 0;
                    _dooropen = 11;
                }
                task->ready_event = nullptr;
                recorderTs(task);
                return task;
            } else {
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
                    if (_is_scheduled) {
                        _credits -= task->len;
                    }
                    _sq.erase(it);
                    BPS_CHECK(task->tensor_name != "");
                    BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                                   << " getTask: " << task->tensor_name << " key: " << task->key
                                   << " rank: " << BytePSGlobal::GetLocalRank();
                    task->ready_event = nullptr;
                    recorderTs(task);
                    return task;
                }
            }

            return nullptr;
        }
// TODO: check if the multiset works
        // TODO: use multiset to update the PULL stage
        // TODO: make it faster
//                if (_qt == PULL && tmp.find("gradient") != tmp.npos) {
//                    if (_dequeue != 1 && _sizepointer < 13) {
//                        bool taskisstart = task->priority == -1 * _grad_checkpoint[_pointer] && _stagestart;
//                        bool taskisproc = !_mystack.empty() && task->priority > -1 * _grad_checkpoint[_pointer]
//                                          && task->priority < -1 * _grad_checkpoint[_pointer - 1]
//                                          && task->priority == _mystack.top() + 1;
//                        bool starttagged = _stagestart && _tensor_part[_grad_checkpoint[_pointer]];
//                        bool proctagged = !_mystack.empty() && _tensor_part[(_mystack.top() + 1) * -1]
//                                          && _mystack.top() + 1 > -1 * _grad_checkpoint[_pointer]
//                                          && _mystack.top() + 1 < -1 * _grad_checkpoint[_pointer - 1];;
//                        if (taskisstart || taskisproc || starttagged || proctagged) {
//                            if (starttagged)
//                                for (int x = 0; x < _tensor_part[_grad_checkpoint[_pointer]]; x++) {
//                                    _mystack.push(_grad_checkpoint[_pointer] * -1);
//                                    _stagestart = 0;
//                                    BPS_LOG(TRACE) << "PULL: ENQUEUE at start element not firstly: "
//                                                   << _grad_checkpoint[_pointer] * -1 << " mystack size: "
//                                                   << _mystack.size() << "  sizepointer: " << _sizepointer;
//                                }
//
//                            else if (proctagged) {
//                                int tmp = _mystack.top() + 1;
//                                for (int x = 0; x < _tensor_part[tmp * -1]; x++) {
//                                    _mystack.push(tmp);
//                                    BPS_LOG(TRACE) << "PULL: ENQUEUE in proc element not firstly: " << tmp
//                                                   << " mystack size: " << _mystack.size() << "  sizepointer: "
//                                                   << _sizepointer;
//                                }
//                            } else {
//                                if (taskisstart) _stagestart = 0;
//                                _tensor_part[task->priority * -1] = task->total_partnum;
//                                for (int x = 0; x < task->total_partnum; x++) {
//                                    _mystack.push(task->priority);
//                                    BPS_LOG(TRACE) << "PULL: ENQUEUE element firstly: " << task->priority
//                                                   << "  sizepointer: " << _sizepointer;
//                                }
//                            }
//
//                            if (!_mystack.empty() && _mystack.top() * -1 == _grad_checkpoint[_pointer - 1] + 1) {
//                                _dequeue = 1;
//                                dynamic_size = _backward_exec[_sizepointer++];
//                                BPS_LOG(TRACE) << "PULL: enqueue operation of one stage is over." << "  _sizepointer:"
//                                               << _sizepointer << "  mystack top is: " << _mystack.top();
//                                break;
//                            }
//                        }
//                        continue;
//                    }
//
//                    if (_sizepointer < 13) {
//                        if (task->priority != _mystack.top())
//                            continue;
//                        // _noleftsize = 1;
//                        BPS_LOG(TRACE) << "priority=" << task->priority << ", top=" << _mystack.top() << ", line="
//                                       << BytePSGlobal::pushsize[_sizepointer - 1]
//                                       << " size=" << dynamic_size << " len=" << task->len;
//                        if (dynamic_size > task->len && task->priority > BytePSGlobal::pushsize[_sizepointer - 1]) {
//                            dynamic_size -= task->len;
//                            _sq.erase(it);
//                            _mystack.pop();
//                        } else {
//                            _dequeue = 0;
//                            _pointer--;
//                            _stagestart = 1;
//                            BPS_LOG(TRACE) << "PULL: No left size. Waiting for next gradient block.";
//                            break;
//                        }
//                    } else {
//                        if (!_mystack.empty() && task->priority != _mystack.top())continue;
//                        if (!_pulldoor) {
//                            forward_dynamic_size = _forward_exec[_exec_stage];
//                            _stagepullnum = 0;
//                            BPS_LOG(TRACE) << "exec_stage: " << _exec_stage << " initilized."
//                                           << "  beginning dynamic size:" << forward_dynamic_size;
//                        }
//                        if (!_mystack.empty() && (forward_dynamic_size > task->len || _exec_stage > 12)) {
//                            _sq.erase(it);
//                            _mystack.pop();
//                            forward_dynamic_size -= task->len;
//                            _pulldoor++;
//                            BPS_LOG(TRACE) << "PULL: dequeue after zero: " << task->tensor_name << "  _exec_stage is:"
//                                           << _exec_stage << "  forward dynamic size:"
//                                           << forward_dynamic_size << "  pull door val is:" << _pulldoor;
//                        } else if (!_mystack.empty() && _mystack.top() >= -1 * _grad_checkpoint[_exec_stage + 1]) {
//                            _sq.erase(it);
//                            _mystack.pop();
//                            _pulldoor++;
//                            BPS_LOG(TRACE) << "PULL: dequeue after zero enforced: " << task->tensor_name
//                                           << "  _exec_stage is:" << _exec_stage << "   forward dynamic size:"
//                                           << forward_dynamic_size << "  pull door val is:" << _pulldoor;
//                        } else {
//                            if (!_stagepullnum && _pulldoor) {
//                                _stagepullnum = _pulldoor;
//                                BPS_LOG(TRACE) << "initilize stagepullnum at stage " << _exec_stage << ":  "
//                                               << _stagepullnum;
//                            }
//                            break;
//                        }
//                    }
//                    if (_sizepointer == 13 && !_stagepullnum && _mystack.empty())//reset parameter
//                    {
//                        BPS_LOG(TRACE) << "Clear.";
//                        _dequeue = 0;
//                        _pointer = 12;
//
//
//
//
//
//                         = _grad_checkpoint[_pointer];
//                        _stagestart = 1;
//                        _meetzero = 0;
//                        _sizepointer = 1;//different from push process
//                        _exec_stage = 0;
//                        _stagepullnum = 0;
//                        _pulldoor = 0;
//                    }
//                    task->ready_event = nullptr;
//                    recorderTs(task);
//                    return task;
//                }

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
            std::lock_guard <std::mutex> lock(_mutex);
            return _sq.size();
        }

        void BytePSScheduledQueue::reportFinish(int size) {
            std::lock_guard <std::mutex> lock(_mutex);
            if (_is_scheduled) {
                _credits += size;
            }
            if (_qt == PUSH) {
                if (_meetzero) {
                    if (_dooropen < 11)
                        _dooropen++;
                }
            }
            // TODO: update the PULL stage
            // TODO: consider adding the TTE to another queue so that the PULL stage can be easier
//            if (_qt == PULL) {
//                if (_stagepullnum > 0) {
//                    _stagepullnum--;
//                    BPS_LOG(TRACE) << "PULL PROCESS FINISH: _stagepullnum value is:" << _stagepullnum;
//                    if (!_stagepullnum) {
//                        _exec_stage++;
//                        _pulldoor = 0;
//                        BPS_LOG(TRACE) << "STAGE PULL PROCESS FINISH: stage is:" << _exec_stage - 1;
//                    }
//                }
//            }
            return;
        }

    }  // namespace common
}  // namespace byteps
