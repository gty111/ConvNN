APP = conv
SRC = src
BIN = bin
DATA = data
CUDNN ?= $(shell spack location -i cudnn)
INC = $(CUDNN)/include $(shell pwd)/include $(OPENCV)/include/opencv4
LIB = $(CUDNN)/lib
FLAG = $(addprefix -I,$(INC)) $(addprefix -L,$(LIB)) -lcudnn 

TEST_LAYER = fc act pool softmax batchnorm
TEST_OPENCV = cifar_cv
EXAMPLE = cifar mnist 

$(EXAMPLE):%:
	@nvcc src/$@.cu $(FLAG) $^ -o $(BIN)/$@ 
	./$(BIN)/$@ | tee log

$(TEST_LAYER):%:
	@nvcc -g $(FLAG) test_layer/$@.cu -o $(BIN)/$@
	./$(BIN)/$@

$(TEST_OPENCV):%:
	@nvcc $(FLAG) test_opencv/$@.cu -o $(BIN)/$@
	./$(BIN)/$@

data:
	$(shell [ ! -d $(DATA) ] && mkdir $(DATA))
	wget -P $(DATA) http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	wget -P $(DATA) http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	wget -P $(DATA) http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	wget -P $(DATA) http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
	gzip -d $(DATA)/train-images-idx3-ubyte.gz
	gzip -d $(DATA)/train-labels-idx1-ubyte.gz
	gzip -d $(DATA)/t10k-images-idx3-ubyte.gz
	gzip -d $(DATA)/t10k-labels-idx1-ubyte.gz
	wget -P $(DATA) http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
	tar -xzf $(DATA)/cifar-10-binary.tar.gz -C $(DATA)

clean:
	@rm -rf data bin
.PHONY: data run