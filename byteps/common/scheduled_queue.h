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

            void reportFinish(int size);

        private:
            struct comparator {
                bool operator()(std::shared_ptr <TensorTableEntry> a, std::shared_ptr <TensorTableEntry> b) {
                    return (a->priority > b->priority);
                }
            };
            std::vector <std::shared_ptr<TensorTableEntry>> _sq;
            std::multiset <std::shared_ptr<TensorTableEntry>, comparator> _ms;
            std::vector <std::shared_ptr<TensorTableEntry>> _mysq;
            std::stack<int> _mystack;
            std::stack<int> _mystackpull;
            std::mutex _mutex;
            uint64_t _credits;
            bool _is_scheduled;
            int _tensor_part[500] = {0};
            int _visited[500] = {0};
            int _meetzero = 0;
            int _dooropen = 11;
            int _pulldoor = 0;
            int batchsize = atoi(getenv("Z_BATCH_SIZE"));
            int _grad_checkpoint[13] = {-1, 4, 11, 21, 32, 43, 52, 85, 110, 150, 180, 230, 250};
            int B = atoi(getenv("BPS_NET_B"));
            int _door = atoi(getenv("BPS_DOORS"));
            long long _bps_credit = atoi(getenv("BPS_CREDIT"));
            long long _backward_exec[13] =  {520, 274, 340, 260, 185, 260, 274, 75, 15, 18, 14, 7, 0};
            int _exec_stage = 0;
            int _noleftsize = 0;
            int _sizepointer = 0;
            int _stagepullnum = 0;
            int _dequeue = 0;
            int _pointer = 12;
            int _stagestart = 1;
            long long dynamic_size = 0;
            int _pushsize = 0;
            int _pullsize = 0;
            int expected_priority = _grad_checkpoint[_pointer];
            QueueType _qt;
            ReadyTable *_rt;
        };
    }  // namespace common
}  // namespace byteps

#endif  // BYTEPS_SCHEDULED_QUEUE_H
