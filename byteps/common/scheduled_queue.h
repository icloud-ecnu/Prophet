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

#ifndef BYTEPS_SCHEDULED_QUEUE_H
#define BYTEPS_SCHEDULED_QUEUE_H

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>
#include <stack>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "ready_table.h"
#include <sys/timeb.h>

namespace byteps {
    namespace common {
        class BytePSScheduledQueue {
        public:
            BytePSScheduledQueue(QueueType type);

            QueueType getQueueType() { return _qt; }

            void addTask(std::shared_ptr <TensorTableEntry>);

            void recorderTs(std::shared_ptr <TensorTableEntry>);

            std::shared_ptr <TensorTableEntry> getTask();

            std::shared_ptr <TensorTableEntry> getTask(uint64_t key);

            std::multiset < std::shared_ptr < TensorTableEntry >> ::iterator findTask(int priority);

            uint32_t pendingSize();

            void reportFinish(std::shared_ptr <TensorTableEntry> task);

        private:
            struct comparator {
                bool operator()(std::shared_ptr <TensorTableEntry> a, std::shared_ptr <TensorTableEntry> b) {
                    return (a->priority > b->priority);
                }
            };
            std::vector <std::shared_ptr<TensorTableEntry>> _sq;
            std::multiset <std::shared_ptr<TensorTableEntry>, comparator> _ms;
            std::vector <std::shared_ptr<TensorTableEntry>> _mysq;
            std::multiset <int, std::greater<int> > _mystack;
            std::stack<int> _mystackpull;
            std::mutex _mutex;
            uint64_t _credits;
            bool _is_scheduled;
            int _tensor_part[160] = {0};
            int _meetzero = 0;
            int _dooropen = 11;
            int _pulldoor = 0;
            int batchsize = 64;
            int _grad_checkpoint[13] = {-1, 9, 22, 35, 50, 62, 77, 90, 103, 117, 130, 143, 156};
            int B = 125000 ;
            double _backward_exec[13] = {47, 46, 26, 30, 37, 53, 44, 64, 90, 74, 58, 15, 0};
//            int _forward_exec[13] = {0, 1350000, 1400000, 840000, 900000, 1275000, 1620000, 1335000, 1900000, 2700000,
//                                     2200000, 1750000, 0};
            int _exec_stage = 0;
            int pull_num = 0;
            int pulled_num = 0;
            int _noleftsize = 0;
            int forward_dynamic_size;
            int _sizepointer = 0;
            int _stagepullnum = 0;
            int _dequeue = 0;
            int _stagestart = 1;
            int dynamic_size = 0;
            int _pushsize = 0;
            int _pullsize = 0;
            int expected_priority;
            QueueType _qt;
            ReadyTable *_rt;
            //added by qi
            int _init_pointer = 12;
            int _pointer;
            std::string begin_name = "Comm.byteps.gradient_144";
            long long timer = 0;
            int duration = 0;
            long long next_timer = 0;
            long long getSystemTime(){
                timeb t;
                ftime(&t);
                return t.time * 1000 + t.millitm;
            }
            int max_dynamic_size = 1000000;



        };
    }  // namespace common
}  // namespace byteps

#endif  // BYTEPS_SCHEDULED_QUEUE_H
