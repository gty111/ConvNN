APP = conv
SRC = src
BIN = bin
DATA = data
CUDNN ?= $(shell spack location -i cudnn)
INC = $(CUDNN)/include $(shell pwd)/include
LIB = $(CUDNN)/lib
FLAG = $(addprefix -I,$(INC)) $(addprefix -L,$(LIB)) -lcudnn 

TEST_LAYER = fc act pool softmax batchnorm
TEST_CUDNN = example test_conv test_fc test_max_pool test_relu
EXAMPLE = cifar mnist 

$(EXAMPLE):%:check check_data
	@nvcc src/$@.cu $(FLAG) -o $(BIN)/$@ 
	./$(BIN)/$@ | tee log

$(TEST_LAYER):%:check 
	@nvcc -g $(FLAG) test_layer/$@.cu -o $(BIN)/$@
	./$(BIN)/$@

$(TEST_CUDNN):%:check
	@nvcc $(FLAG) test_cudnn/$@.cu -o $(BIN)/$@
	./$(BIN)/$@

data:
	rm -rf $(DATA)
	mkdir $(DATA)
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

check:
	@if [ ! -d $(BIN) ] ; then mkdir $(BIN) ; fi

check_data:
	@if [ ! -d $(DATA) ] ; then make data ; fi
	
clean:
	@rm -rf data bin

.PHONY: check_data check data clean $(EXAMPLE) $(TEST_LAYER) $(TEST_CUDNN)