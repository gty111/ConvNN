#include "NN.h"
#include "myRandom.h"
#include "mnist_dataset.h"

// var bind to NN
Tensor<double>in(1,28,28); 
int label;

// NN
NN nn;

double valid(DataSet &test,Random &r,int num){
    int correct = 0,idx;
    for(int i=0;i<num;i++){
        idx = r.next();
        in.set(test.image(idx,0,0));
        nn.forward();
        if(nn.pred()==test.label[idx])correct++;
    }
    double corr_rate = (double)correct / num;
    printf("Acc on TestSet(%d): %.2f%%\n",num,corr_rate*100);
    return corr_rate;
}

void train(DataSet &trainSet,DataSet testSet,
        Random &train_r,Random &test_r,int num){
    
    double loss = 0;
    int learn_interval = 2048;
    int interval = 128;
    int batch_size = 64;

    for(int i=0;i<num;i++){
        int index = train_r.next();
        in.set(trainSet.image(index,0,0));
        label = trainSet.label[index];
        nn.forward();

        loss += nn.loss();
        nn.backward();

        if((i+1)%interval==0){
            loss /= interval;
            printf("[%5d/%-5d] Loss: %5.6f\n",i+1,num,loss);
            loss = 0;
        }
        if((i+1)%learn_interval==0){
            valid(testSet,test_r,1200);
        }
        if((i+1)%batch_size==0){
            nn.apply_grad();
        }
        
    }
}

int main(){
    // print at once
    setbuf(stdout, NULL);
    
    // input data 
    DataSet trainSet("data/train-images-idx3-ubyte","data/train-labels-idx1-ubyte");
    DataSet testSet("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte");

    // create NN
    nn.add(new Conv(16,3,1,&in));
    nn.add(new Relu(nn.lastOut()));
    nn.add(new Conv(16,3,1,nn.lastOut()));
    nn.add(new Relu(nn.lastOut()));
    nn.add(new MaxPool(nn.lastOut()));
    nn.add(new Flattern(nn.lastOut()));
    nn.add(new Fc(128,nn.lastOut()));
    nn.add(new Relu(nn.lastOut()));
    nn.add(new Fc(10,nn.lastOut()));
    nn.add(new Relu(nn.lastOut()));
    nn.add(new LogSoftmax(nn.lastOut(),&label));
    
    nn.setLR(5e-3);
    nn.setMomentum(0.9);

    nn.initData(0,0.1);

    // random index
    Random train_r(trainSet.items_num),test_r(testSet.items_num);

    train(trainSet,testSet,train_r,test_r,trainSet.items_num);

}