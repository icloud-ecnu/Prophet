
void ScheduledQueue::reportFinish(int size, int priority) {
  std::lock_guard<std::mutex> lock(_mutex);
  if (_is_scheduled) {
    _credits += size;
  }
  if (Global::pre_run && _qt == PUSH) {
    // 计算
    int id = priority * -1;
    if (!finish_tag[id]) {
      finish_count += 1;
      auto now = std::chrono::system_clock::now();
      auto duration = now.time_since_epoch();
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);
      long long tac = (long long)us.count();
      double t = (double)(tac - _push_start_tic[id]);
      double possible_B = (double)size * 1000.0 / t;
      if (possible_B > (double)Global::B) {
        Global::B = (long long)possible_B;
      }
    }
    finish_tag[id] = true;
    if (finish_count == Global::total_grad) {
      Global::pre_run = false;
    }
  } else if (_qt == PUSH && size > 0 && _meetzero) {
    _bps_credit += size;
  }
  return;
}


void BytePSScheduledQueue::addTask(std::shared_ptr<TensorTableEntry> entry) {
  std::lock_guard<std::mutex> lock(_mutex);
  // 预训练过程结束，从 Global 获取预训练阶段保存的数据
  if (!pre_run_result_sync) {
    if (!Global::pre_run && _qt == PUSH) {
      pre_run_result_sync = true;
      expected_priority = Global::total_grad - 1;
      _pointer = Global::_grad_checkpoint.size() - 1;
    }
  }
  if (Global::pre_run) {
    _sq.push_back(entry);
    if (_qt == PUSH && (entry->tensor_name).find(tensor_keywords) !=
                       (entry->tensor_name).npos) {
      // 预训练阶段，处理优先级、层的个数相关
      int pr = entry->priority * -1;
      if (pr > Global::total_grad) {
        Global::total_grad = pr + 1;
      }
      // 计时
      auto us = get_tic_tac();
      long long tic = (long long)us.count();
      if (_grad_tic[pr] == 0) {
        processed_grad_count++;
        _grad_tic[pr] = tic;
      }
      if (processed_grad_count == Global::total_grad) {
        // 判断数量级上的差距，根据阶梯状计算时间完成分组
        double avg = 0;
        for (int i = 1; i < processed_grad_count; i++) {
          double x = fabs(_grad_tic[i] - _grad_tic[i - 1]);
          avg = (((double)(i - 1)) / i) * avg + (((double)(1)) / i) * x;
        }
        avg *= 2;
        Global::_grad_checkpoint.push_back(-1);
        for (int i = 1; i < Global::total_grad; i++) {
          double diff = fabs(_grad_tic[i] - _grad_tic[i - 1]);
          if (diff > avg) {
            diff /= 1000;  // 微秒转为毫秒
            if (Global::_backward_exec.size() == 0) {
              double _diff = fabs(_grad_tic[i - 1] - _grad_tic[0]);
              _diff /= 1000;
              Global::_backward_exec.push_back(_diff);
            }
            Global::_grad_checkpoint.push_back(i - 1);
            Global::_backward_exec.insert(Global::_backward_exec.begin(), diff);
          }
        }
        Global::_grad_checkpoint.push_back(Global::total_grad - 1);
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
  return;
}
