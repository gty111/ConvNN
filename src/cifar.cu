#include "cifar_dataset.h"
#include "NN.h"

#define BATCH_SIZE 50
#define ROWS 32
#define COLS 32
#define CHANNEL 3
#define EPOCH 20

int main(){
    setbuf(stdout, NULL);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    printf("==>loading data...\n");
    CIFAR_DataSet cifar("data/cifar-10-batches-bin");
    
    Tensor<float>in(BATCH_SIZE,CHANNEL,ROWS,COLS);
    Tensor<uint8_t>label(BATCH_SIZE);

    cudnnHandle_t cuhandle;
    cudnnCreate(&cuhandle);

    printf("==>creating NN...\n");
    NN nn;
    nn.add(new Conv(64,3,1,&in));
    nn.add(new BatchNorm2D(nn.lastOut()));
    nn.add(new Activation(CUDNN_ACTIVATION_RELU,nn.lastOut()));
    nn.add(new Conv(64,3,1,nn.lastOut()));
    nn.add(new BatchNorm2D(nn.lastOut()));
    nn.add(new Activation(CUDNN_ACTIVATION_RELU,nn.lastOut()));
    nn.add(new Pooling(CUDNN_POOLING_MAX_DETERMINISTIC,2,2,nn.lastOut()));
    nn.add(new Conv(128,3,1,nn.lastOut()));
    nn.add(new BatchNorm2D(nn.lastOut()));
    nn.add(new Activation(CUDNN_ACTIVATION_RELU,nn.lastOut()));
    nn.add(new Conv(128,3,1,nn.lastOut()));
    nn.add(new BatchNorm2D(nn.lastOut()));
    nn.add(new Activation(CUDNN_ACTIVATION_RELU,nn.lastOut()));
    nn.add(new Pooling(CUDNN_POOLING_MAX_DETERMINISTIC,2,2,nn.lastOut()));
    nn.add(new Fc(10,nn.lastOut()));
    nn.add(new Activation(CUDNN_ACTIVATION_RELU,nn.lastOut()));
    nn.add(new Softmax(nn.lastOut(),&label));

    nn.setLR(1e-3);
    nn.setmomentum(0.7);
    nn.initData(0,0.1);

    printf("==>training...\n");
    printf("SEED:%d\n",SEED);
    int N_batch = cifar.train_num / BATCH_SIZE;
    Random train_r(N_batch);
    int train_acc_num,test_acc_num;
    for(int j=0;j<EPOCH;j++){
        train_acc_num = 0;
        test_acc_num = 0;
        nn.setMode(NN_TRAIN);
        train_r.shuffle();
        for(int i=0;i<N_batch;i++){
            int idx = train_r.next();
            in._data = &cifar._image_train[idx*BATCH_SIZE*CHANNEL*ROWS*COLS];
            label._data = &cifar._label_train[idx*BATCH_SIZE];
            nn.forward();
            nn.backward();
            nn.apply_grad();
            train_acc_num += nn.getAccNum();
            printf("Epoch:%2d [%5d/%-5d] Loss: %5.6f Acc: %3.6f%%\n",j+1,i+1,N_batch,nn.getAvgLoss(),nn.getAcc());
        }
        printf("Epoch:%2d [Acc on Train] %3.6f%%\n",j+1,(float)train_acc_num/cifar.train_num*100);
        int test_N_batch = cifar.test_num / BATCH_SIZE;
        for(int i=0;i<test_N_batch;i++){
            in._data = &cifar._image_test[i*BATCH_SIZE*CHANNEL*ROWS*COLS];
            label._data = &cifar._label_test[i*BATCH_SIZE];
            nn.forward();
            test_acc_num += nn.getAccNum();
        }
        printf("Epoch:%2d [Acc on Test] %3.6f%%\n",j+1,(float)test_acc_num/cifar.test_num*100);
        // getchar();
        nn.setLR(nn._lr*0.99);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("ElapsedTime: %f s\n",elapsedTime/1000);
    
}