APP = conv
SRC = src
BIN = bin
DATA = data

mnist: $(SRC)/mnist.cpp
	$(shell [ ! -d $(BIN) ] && mkdir $(BIN) )
	@g++ $(CXXFLAG) -O3 -Wall -Wextra $^ -o $(BIN)/$@
	@$(BIN)/$@ | tee log

cifar: $(SRC)/cifar.cpp
	@g++ $(CXXFLAG) -O3 -Wall -Wextra $^ -o $(BIN)/$@
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
.PHONY: data mnist cifar clean 