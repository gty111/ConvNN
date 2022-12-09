APP = conv
SRC = src
BIN = bin
DATA = data
FLAG = -I include -O3 -Wall -Wextra $^ -o $(BIN)/$@ 

mnist: $(SRC)/mnist.cpp
	@g++ $(CXXFLAG) $(FLAG)
	@$(BIN)/$@ | tee log

Dmnist: $(SRC)/mnist.cpp
	@g++ -g -D DEBUG $(CXXFLAG) $(FLAG)
	@$(BIN)/$@ | tee log

cifar: $(SRC)/cifar.cpp
	@g++ $(CXXFLAG) $(FLAG)
	@$(BIN)/$@ | tee log

Dcifar: $(SRC)/cifar.cpp
	@g++ -g -D DEBUG $(CXXFLAG) $(FLAG)
	@$(BIN)/$@ | tee log

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
.PHONY: data mnist cifar clean test_layer