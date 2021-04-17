sudo modprobe ifb numifbs=1
sudo ip link set dev ifb0 up
sudo tc qdisc add dev ens3 handle ffff: ingress
sudo tc filter add dev ens3 parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev ifb0
sudo tc qdisc add dev ifb0 root handle 1: htb default 10
sudo tc class add dev ifb0 parent 1: classid 1:1 htb rate 3000mbit
sudo tc class add dev ifb0 parent 1:1 classid 1:10 htb rate 3000mbit
sudo tc qdisc add dev ens3 root handle 1: htb default 10
sudo tc class add dev ens3 parent 1: classid 1:1 htb rate 3000mbit
sudo tc class add dev ens3 parent 1:1 classid 1:10 htb rate 3000mbit