# Prophet

> This repository is forked from [https://github.com/bytedance/byteps](https://github.com/bytedance/byteps) with a modification on scheduling mechanism

Optimizing performance for Distributed Deep Neural Network (DDNN) training has recently become increasingly compelling, as the DNN model gets complex and the training dataset grows large.

While existing works on communication scheduling mostly focus on overlapping the computation and communication to improve DDNN training performance, the GPU and network resources are still **under-utilized** in DDNN training clusters.

To tackle this issue, we design and implement a **predictable** communication scheduling strategy named **Prophet** to schedule the gradient transfer in an adequate order, with the aim of maximizing the GPU and network resource utilization.

Leveraging our observed **stepwise pattern** of gradient transfer start time, **Prophet** first uses the monitored network bandwidth and the profiled time interval among gradients to predict the appropriate number of gradients that can be grouped into **blocks**.

Then, these **gradient blocks** can be transferred one by one to guarantee high utilization of GPU and network resources while ensuring the priority of gradient transfer (**i.e.,** low-priority gradients cannot preempt high-priority gradients in the network transfer).

**Prophet** can make the forward propagation start as early as possible so as to greedily reduce the waiting (idle) time of GPU resources during the DDNN training process.
