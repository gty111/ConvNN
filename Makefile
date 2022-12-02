APP = conv
SRC = src
BIN = bin
DATA = data

run:
	$(shell [ ! -d $(BIN) ] && mkdir $(BIN) )
	@g++ $(CXXFLAG) -O3 -Wall -Wextra $(SRC)/$(APP).cpp -o $(BIN)/$(APP)
	@$(BIN)/$(APP) | tee log

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

clean:
	@rm -rf data bin
.PHONY: data run clean