MODULE_NAME := libnvm
obj-m += $(MODULE_NAME).o
$(MODULE_NAME)-objs := pci.o list.o ctrl.o map.o

KDIR := /lib/modules/$(shell uname -r)/build

# NVIDIA driver source for P2P support (nvidia_p2p_* symbols)
NVIDIA_SRC ?= $(wildcard /usr/src/nvidia-*/nvidia)
NVIDIA_ROOT ?= $(wildcard /usr/src/nvidia-*)
ccflags-y += -D_CUDA
ifneq ($(NVIDIA_SRC),)
ccflags-y += -I$(NVIDIA_SRC)
endif
ifneq ($(NVIDIA_ROOT),)
KBUILD_EXTRA_SYMBOLS += $(NVIDIA_ROOT)/Module.symvers
endif

.PHONY: all clean reload unload load install

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

reload: unload load

unload:
	-rmmod $(MODULE_NAME).ko

load:
	insmod $(MODULE_NAME).ko max_num_ctrls=64

install: all
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
