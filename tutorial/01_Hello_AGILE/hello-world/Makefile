NVCC_FLAG :=  -w -Xptxas -v \
 				-arch=native

AGILE_PATH := /home/zhuoping/workspace/AGILE

INCLUDE_PATH := -I$(AGILE_PATH)/include \
				-I$(AGILE_PATH)/common \
				-I$(AGILE_PATH)/benchmarks/common \

LIBS := -lcuda -lcublas

all:
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) main.cu $(LIBS) -o ./hello-world
