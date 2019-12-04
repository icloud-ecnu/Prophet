

path = 'C:/Users/Chris/Desktop/byteps/mxnet-resnet152-bs128-pslimit/worker'
show = 150
import matplotlib.pyplot as plt
for i in range(1):
    gpu_util = []
    timestamp =[]
    with open(path + str(i+1) + '/gpu-log.txt') as f1:
        line = f1.readline()
        while(line):
            line = f1.readline()
            if '2019/12/03' in line and line > '2019/12/03 07:55:00' and line < '2019/12/03 07:58:30':
                time,utilization,pcie = line.split(', ')
                for x  in time.split(' '):
                    if ':' in x:
                        time = x[:-4]
    #             timestamp.append(time)
                gpu_util.append(utilization)
        f1.close()
    recv =[]
    send =[]
    with open(path + str(i+1)+'/netinfo.txt') as f2:
        line = f2.readline()
        while(line):
            line = f2.readline()
            if 'Tue Dec  3' in line and line > 'Tue Dec  3 07:55:00' and line < 'Tue Dec  3 07:58:30':
                ##extract time from string
                time = 0
                for x in line.split(' '):
                    if ':' in  x:
                        time = x
                line = f2.readline()
                line = line[2:]
                if line is not None:
                    recv_band,send_band = line.split(' ')
                    recv.append(int(recv_band) / 1024 / 1024)
                    send.append(int(send_band) / 1024 / 1024)    
        f2.close()


    for i in range(len(gpu_util)):
        gpu_util[i] = int(gpu_util[i].split(' ')[0]) 

#     plt.plot(range(show),send[:show])
#     plt.plot(range(show),recv[:show])
#     plt.plot(range(show),gpu_util[:show])

path = 'C:/Users/Chris/Desktop/byteps/mxnet-resnet152-bs128-pslimit/ps/'
import matplotlib.pyplot as plt
for i in range(1):
    gpu_util = []
    timestamp =[]
    with open(path  + 'gpu-log.txt') as f1:
        line = f1.readline()
        while(line):
            line = f1.readline()
            if '2019/12/03' in line and line > '2019/12/03 07:55:00' and line < '2019/12/03 07:58:30':
                time,utilization,pcie = line.split(', ')
                for x  in time.split(' '):
                    if ':' in x:
                        time = x[:-4]
    #             timestamp.append(time)
                gpu_util.append(utilization)
        f1.close()
    recv =[]
    send =[]
    with open(path +'/netinfo.txt') as f2:
        line = f2.readline()
        while(line):
            line = f2.readline()
            if 'Tue Dec  3' in line and line > 'Tue Dec  3 07:55:00' and line < 'Tue Dec  3 07:58:30':
                ##extract time from string
                time = 0
                for x in line.split(' '):
                    if ':' in  x:
                        time = x
                line = f2.readline()
                line = line[2:]
                if line is not None:
                    recv_band,send_band = line.split(' ')
                    recv.append(int(recv_band) / 1024 / 1024)
                    send.append(int(send_band) / 1024 / 1024)    
        f2.close()


#     for i in range(len(gpu_util)):
#         gpu_util[i] = int(gpu_util[i].split(' ')[0]) 
    plt.plot(range(show),send[:show])
    plt.plot(range(show),recv[:show])
# plt.show()
plt.savefig(path + 'ps-flow.jpeg',dpi=300,bbox_inches = 'tight')
