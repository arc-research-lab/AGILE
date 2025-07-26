
<p align="center">
  <picture>
    <img alt="AGILE" src="https://raw.githubusercontent.com/arc-research-lab/AGILE/refs/heads/main/figures/AGILE-logo1.png" width=55%>
  </picture>
</p>

<h2 align="center">
AGILE: Lightweight and Efficient Asynchronous GPU-SSD Integration
</h2>


## Installation

AGILE requires a modified version of [GDRCopy](https://github.com/NVMe-SSD/GDRCopy), which is included in this repo ([./driver/gdrcopy](./driver/gdrcopy)). Please follow the instructions to build and install it.

AGILE's host code requires a continuous physical memory region, which is reserved in **/etc/default/grub** by adding the **GRUB\_CMDLINE\_LINUX** option. For example, **GRUB\_CMDLINE\_LINUX="memmap=1G\\\\\\$128G"** will reserve 1 GB DRAM memory starting at 128 GB. After changing **/etc/default/grub**, executing **sudo update-grub** and **sudo reboot** to apply the modification. 

AGILE relys on the GPUs' BAR1 Memory as the source and destination in GPU-SSD peer-to-peer communication. If the default BAR1 memory size is too small (typically 128MB), please refer [NVIDIA Display Mode Selector Tool](https://developer.nvidia.com/displaymodeselector) (1.67.0) to increase the BAR1 memory size.


## Experiments
AGILE has been evaluated on a Dell R750 server running Ubuntu 20.04, equipped with an Nvidia RTX 5000 Ada GPU, a Dell Ent NVMe AGN MU AIC 1.6TB SSD, and two Samsung 990 PRO 1TB SSDs. The Nvidia Driver version is 550.54 and the CUDA version is 12.8.

For setting up the baseline BaM, please refer to [https://github.com/ZaidQureshi/bam](https://github.com/ZaidQureshi/bam). The BaM version baselines can be found at [./baseline/benchmarks](./baseline/benchmarks).

### Experimental results in Figure 4 - 12

![image](./figures/figure4.png)
![image](./figures/figure5.png)
![image](./figures/figure6.png)
![image](./figures/figure7.png)
![image](./figures/figure8.png)
![image](./figures/figure9.png)
![image](./figures/figure10.png)
![image](./figures/figure11.png)
![image](./figures/figure12.png)

### **Table: Experimental Bash scripts for reproducing results for Figure 4 - 11.**
| Figures      | Corresponding Scripts                |
|--------------|---------------------------------------|
| Figure 4     | `run_ctc.sh`                          |
| Figure 5     | `rand_read.sh`                        |
| Figure 6     | `rand_write.sh`                       |
| Figure 7 - 10| `run_dlrm.sh` & `auto_dlrm.sh`         |
| Figure 11    | `run_bfs*.sh` & `run_spmv*.sh`         |

## Todo-lists

We will keep updating AGILE with more features and you are more than welcome to request us to add more functionalities. Currently, we have following plans for improving AGILE:

- [ ] Add documentations for AGILE (APIs, customizing software-cache policy, etc.)
- [ ] Include CPU DRAM as an additional level of software cache.
- [ ] Support for multi-GPU-multi-SSD.


## Citations

```bibtex
@inproceedings{sc25agile,
author = {Yang, Zhuoping and Zhuang, Jinming and Chen, Xingzhen and Jones, Alex K and Zhou, Peipei},
title = {AGILE: Lightweight and Efficient Asynchronous GPU-SSD Integration},
year = {2025},
booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis, SC 2025},
series = {Supercomputing '25}
}
```


### Thanks for your interest in this project. Your growing engagement will inspire us to improve and enhance AGILE continually.
<p align="center">
  <picture>
    <img alt="AGILE" src="https://agile-traffic.zhuopingyang.com/traffic.png" width=100%>
  </picture>
</p>

