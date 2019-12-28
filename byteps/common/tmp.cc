if(_qt == PUSH && tmp.find("gradient") != tmp.npos )
    {
        BPS_LOG(DEBUG) << "Task: " <<  task-> priority << "I have meet zero: " << _meetzeropull << " and door is open: " << _dooropenpull;
        if(task -> priority == 0) {
          _meetzeropull = 1;
         BPS_LOG(DEBUG) << "Meet zero.";
         }
        if(!_meetzeropull)
        {
            if(task -> priority !=  _mystackpull.top())continue; 
            BPS_LOG(DEBUG) << "PUSH GRADIENT: " << tmp;
            _tensor_partpull[ task -> priority * -1]++; 
            if(_tensor_partpull[task -> priority * -1 ] == 1 && task -> total_partnum > 1){
              for(int base = 1; base < task-> total_partnum ; base++)
                _mystackpull.push(task -> priority);//the values in the stack and priority are both negative
                BPS_LOG(DEBUG) << "PUSH elements into mystack  IN THE PROCESS: " << tmp;
            }
            if(_tensor_partpull[ task -> priority * -1 ] == task -> total_partnum )_tensor_numpull++;
            _mystackpull.pop();
        }
        else if(!_dooropenpull) {//we cannot change the value of tensor_part if door is closed.
          BPS_LOG(DEBUG) << "door is closed.";
          break;
        }
        else {
           BPS_LOG(DEBUG) << "Tensor name: " << tmp << "   myqueue top: " << _mystackpull.top()  << "  size of _sq: " << _sq.size();    
           if(task -> priority !=  _mystackpull.top())continue; 
           BPS_LOG(DEBUG) << "PUSH GRADIENT: " << tmp;
           BPS_LOG(DEBUG) << "Pass, and dooopen --";
            _tensor_partpull[ task -> priority * -1]++; 
            if(_tensor_partpull[task -> priority * -1 ] == 1 && task -> total_partnum > 1){
              for(int base = 1; base < task-> total_partnum ; base++)
                _mystackpull.push(task -> priority);
                BPS_LOG(DEBUG) << "PUSH elements into mystack IN THE PROCESS: " << tmp;
            }
            if(_tensor_partpull[ task -> priority * -1 ] == task -> total_partnum )_tensor_numpull++;
            _mystackpull.pop();
            _dooropenpull--;
            BPS_LOG(DEBUG) << "PUSH gradient: " << tmp ;
            // BPS_LOG(DEBUG) << "The door has fbeen closed.";
        }
         BPS_LOG(DEBUG) << "transferred tensor num: " << _tensor_numpull  << "  empty: " << _mystackpull.empty() << " size of myqueue: " << _mystackpull.size();

        //all push process end in this iteration , then reinitalize varibles.
        if(_tensor_numpull == 157 && _mystackpull.empty())
        {
          BPS_LOG(DEBUG) << "Clear.";
          _meetzeropull = 0;
          _dooropenpull = 11;
          // _doorcount = 0;
          _tensor_numpull = 0;
          for(int i = 0; i < 160; i++)_tensor_partpull[i] = 0;
          // for(int i = 0;i < 160; i++) _vis[i] = 0;  
          for(int i = 11; i >= 0; i--)
          {
            for(int j = _grad_checkpoint[i + 1] - 1; j > _middle[i]; j--){
                _mystackpull.push(j * -1 );
                BPS_LOG(DEBUG) << " PUSH element into myqueue: " << j ;
            }
          }
          for(int i = 0 ; i <= 11; i++)
          {
            for(int j = _middle[i] ; j >= _grad_checkpoint[i]; j--){
                _mystackpull.push(j * -1);
                BPS_LOG(DEBUG) << " PUSH element into myqueue: " << j ;
            }
          }
        }
    }