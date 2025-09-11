KDIR ?= /lib/modules/$(shell uname -r)/build
# Path to your NVIDIA driver source dir (you showed 550.54.14):
NVIDIADR ?= /usr/src/nvidia-550.54.14

# nv-p2p.h lives in nvidia-peermem/
ccflags-y += -I$(NVIDIADR)/nvidia-peermem

# So the nv-p2p symbols exported by nvidia.ko resolve at link/load:
KBUILD_EXTRA_SYMBOLS := $(NVIDIADR)/Module.symvers

obj-m += agile_gpu_krnl.o

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean