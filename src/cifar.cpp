#include"cifar_dataset.h"
#include"myRandom.h"
#include"NN.h"


// var bind to NN
Tensor<double>in(3,32,32); 
int label;

NN nn;

double valid(DataSet &data,Random &r,int num){
    int correct = 0,idx;
    for(int i=0;i<num;i++){
        idx = r.next();
        in.set(data.image_test(idx));
        nn.forward();
        if(nn.pred()==data._label_test[idx])correct++;
    }
    double corr_rate = (double)correct / num;
    printf("Acc on TestSet(%d): %.2f%%\n",num,corr_rate*100);
    return corr_rate;
}

void train(DataSet &data,
        Random &train_r,Random &test_r,int num){
    
    printf("SEED:%d\n",seed);
    double loss = 0;
    int learn_interval = 5000;
    int interval = 500;
    int grad_interval = 1;

    for(int i=0;i<num;i++){
        int index = train_r.next();
        in.set(data.image_train(index));
        label = data._label_train[index];
        nn.forward();
        loss += nn.loss();
        nn.backward();

        if((i+1)%interval==0){
            loss /= interval;
            printf("[%5d/%-5d] Loss: %5.6f\n",i+1,num,loss);
            loss = 0;
        }
        if((i+1)%learn_interval==0){
            nn.lr *= 0.98;
            nn.setLR(nn.lr);
            valid(data,test_r,500);
        }
        if((i+1)%grad_interval==0){
            nn.apply_grad();
        }
    }
}

int main(){
    setbuf(stdout, NULL);
    DataSet cifar("data/cifar-10-batches-bin");
    // cifar.saveimg(100);
    // exit(0);

    // create NN

    nn.add(new Conv(4,3,&in));
    nn.add(new Relu(nn.lastOut()));
    nn.add(new Conv(4,3,nn.lastOut()));
    nn.add(new Relu(nn.lastOut()));
    nn.add(new MeanPool(nn.lastOut()));
    nn.add(new Flattern(nn.lastOut()));
    nn.add(new Fc(128,nn.lastOut()));
    nn.add(new Relu(nn.lastOut()));
    nn.add(new Fc(10,nn.lastOut()));
    nn.add(new Relu(nn.lastOut()));
    nn.add(new LogSoftmax(nn.lastOut(),&label));

    nn.setLR(1e-3);

    nn.initData(0,0.1);

    // random index
    Random train_r(cifar.train_num),test_r(cifar.test_num);

    train(cifar,train_r,test_r,cifar.train_num*10);
}