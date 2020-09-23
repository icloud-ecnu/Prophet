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

            B *= 125;
            for (int i = 0; i < 13; i++) {
                _backward_exec[i] = (long long)(_backward_exec[i] * (double)batchsize/64.0);
            }
            for (int i = 0; i < 13; i++) {
                _backward_exec[i] *= B;
            }

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

        void BytePSScheduledQueue::SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
        {
        std::string::size_type pos1, pos2;
        pos2 = s.find(c);
        pos1 = 0;
        while(std::string::npos != pos2)
        {
            v.push_back(s.substr(pos1, pos2-pos1));
        
            pos1 = pos2 + c.size();
            pos2 = s.find(c, pos1);
        }
        if(pos1 != s.length())
            v.push_back(s.substr(pos1));
        }

        int BytePSScheduledQueue::getPriority(const std::string& s) {
            std::vector<std::string> ss;
            SplitString(s, ss, "_");
            return stoi(ss[1]);
        }

        void BytePSScheduledQueue::addTask(std::shared_ptr <TensorTableEntry> entry) {
            std::lock_guard <std::mutex> lock(_mutex);
            if (_qt == PUSH && (entry->tensor_name).find("parameter") != (entry->tensor_name).npos) {
                _ms.insert(entry);
                int p = getPriority(entry->tensor_name);
                BPS_LOG(INFO) << "add " << (entry->tensor_name) << " (p=" << (p) << ")";
                _tensor_part[p * -1] = entry->total_partnum;
            } else {
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
                return x->priority == Priority;
            }
        };

        std::multiset < std::shared_ptr < TensorTableEntry >> ::iterator BytePSScheduledQueue::findTask(int priority) {
            if (_ms.size() == 0) {
                return _ms.end();
            }
            std::multiset < std::shared_ptr < TensorTableEntry >> ::iterator = _ms.begin();
            while (it != _ms.end()) {
                if (getPriority(it->tensor_name) == priority) {
                    break;
                }
                it++;
            }
            return it;
        }

        std::shared_ptr <TensorTableEntry> BytePSScheduledQueue::getTask() {
            std::lock_guard <std::mutex> lock(_mutex);
            std::shared_ptr <TensorTableEntry> task;
            std::multiset < std::shared_ptr < TensorTableEntry >> ::iterator msit;
            if (_qt == PUSH && !_dequeue && _ms.size() > 0) {
                msit = findTask(expected_priority * -1);
                if (msit == _ms.end()) {
                    return nullptr;
                }
                BPS_LOG(INFO) << "expect " << (expected_priority);
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
                if (expected_priority == _grad_checkpoint[_pointer - 1]) {
                    _dequeue = 1;
                    dynamic_size = _backward_exec[_sizepointer++];
                }
                return nullptr;
            }
            if (_qt == PUSH && _dequeue && _ms.size() > 0) {
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
                BPS_LOG(INFO) << "prophet _dequeue " << (task->tensor_name);
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
                } else if (_bps_credit < task -> len) {
                    return nullptr;
                } else if (_bps_credit > task->len) {
                    _bps_credit -= task->len;
                    _ms.erase(msit);
                    _mystack.pop();
                }
                if (_mystack.empty() && _meetzero) {
                    _dequeue = 0;
                    _pointer = 12;
                    expected_priority = _grad_checkpoint[_pointer];
                    _stagestart = 1;
                    _meetzero = 0;
                    _sizepointer = 0;
                    _dooropen = _door;
                    _bps_credit = atoi(getenv("BPS_CREDIT"));
                    for (int i = 0; i < 160; i++) {
                        _visited[i] = 0;
                    }
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
                    task = *it;
                    if (_is_scheduled) {
                        _credits -= task->len;
                    }
                    _sq.erase(it);
                    BPS_CHECK(task->tensor_name != "");
                    BPS_LOG(INFO) << "default " << (task->tensor_name);
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
            if (_qt == PUSH && size > 0 && _meetzero) {
                _bps_credit += size;
            }
            return;
        }

    }  // namespace common
}  // namespace byteps
