
nvidia-smi --query-gpu=timestamp,utilization.gpu,pcie.link.gen.current --format=csv --loop-ms=100 -f /home/ubuntu/gpu-log.txt &
./getnetinfo 
