NVCC_FLAG :=  -w -Xptxas -v \
 				-arch=native

AGILE_PATH := /home/zhuoping/workspace/AGILE

INCLUDE_PATH := -I$(AGILE_PATH)/include \
				-I$(AGILE_PATH)/driver/gdrcopy/include \
				-I$(AGILE_PATH)/common \
				-I$(AGILE_PATH)/benchmarks/common \

LIB_PATH := -L$(AGILE_PATH)/driver/gdrcopy/src

LIBS := -lcuda -lgdrapi -lcublas

all:
	nvcc $(NVCC_FLAG) $(INCLUDE_PATH) main.cu $(AGILE_PATH)/common/common.cpp $(LIB_PATH) $(LIBS) -o ./hello-world
