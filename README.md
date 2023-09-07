# Prophet

> This repository is forked from [https://github.com/bytedance/byteps](https://github.com/bytedance/byteps) with a modification on the commununication scheduling mechanism.

Optimizing performance for Distributed Deep Neural Network (DDNN) training has recently become increasingly compelling, as the DNN model gets complex and the training dataset grows large. While existing works on communication scheduling mostly focus on overlapping the computation and communication to improve DDNN training performance, the GPU and network resources are still **under-utilized** in DDNN training clusters.

To tackle this issue, we design and implement a **predictable** communication scheduling strategy named **Prophet** to schedule the gradient transfer in an adequate order, with the aim of maximizing the GPU and network resource utilization. Leveraging our observed **stepwise pattern** of gradient transfer start time, **Prophet** first uses the monitored network bandwidth and the profiled time interval among gradients to predict the appropriate number of gradients that can be grouped into **blocks**. Then, these **gradient blocks** can be transferred one by one to guarantee high utilization of GPU and network resources while ensuring the priority of gradient transfer (**i.e.,** low-priority gradients cannot preempt high-priority gradients in the network transfer). **Prophet** can make the forward propagation start as early as possible so as to greedily reduce the waiting (idle) time of GPU resources during the DDNN training process.

## Publication

Zhenwei Zhang, Qiang Qi, Ruitao Shang, Li Chen, Fei Xu*, “[Prophet: Speeding up Distributed DNN Training with Predictable Communication Scheduling](https://dl.acm.org/doi/abs/10.1145/3472456.3472467),” in: Proc. of ICPP 2021, August 9-12, 2021. Article No. 69.

```
@inproceedings{zhang2021prophet,
  title={Prophet: Speeding up distributed dnn training with predictable communication scheduling},
  author={Zhang, Zhenwei and Qi, Qiang and Shang, Ruitao and Chen, Li and Xu, Fei},
  booktitle={Proceedings of the 50th International Conference on Parallel Processing},
  pages={1--11},
  year={2021}
}
```
