#include "mnist_dataset.h"
#include "NN.h"

#define BATCH_SIZE 100
#define ROWS 28
#define COLS 28
#define CHANNEL 1
#define EPOCH 6

int main(){
    setbuf(stdout, NULL);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    printf("==>loading data...\n");
    MNIST_DataSet trainSet("data/train-images-idx3-ubyte","data/train-labels-idx1-ubyte");
    MNIST_DataSet testSet("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte");
    
    Tensor<float>in(BATCH_SIZE,CHANNEL,ROWS,COLS);
    Tensor<uint8_t>label(BATCH_SIZE);

    cudnnHandle_t cuhandle;
    cudnnCreate(&cuhandle);

    printf("==>creating NN...\n");
    NN nn;
    nn.add(new Conv(16,3,1,&in));
    nn.add(new Activation(CUDNN_ACTIVATION_RELU,nn.lastOut()));
    nn.add(new Conv(16,3,1,nn.lastOut()));
    nn.add(new Activation(CUDNN_ACTIVATION_RELU,nn.lastOut()));
    nn.add(new Pooling(CUDNN_POOLING_MAX_DETERMINISTIC,2,2,nn.lastOut()));
    nn.add(new Fc(128,nn.lastOut()));
    nn.add(new Activation(CUDNN_ACTIVATION_RELU,nn.lastOut()));
    nn.add(new Fc(10,nn.lastOut()));
    nn.add(new Activation(CUDNN_ACTIVATION_RELU,nn.lastOut()));
    nn.add(new Softmax(nn.lastOut(),&label));

    nn.setLR(5e-3);
    nn.initData(0,0.1);

    printf("==>training...\n");
    printf("SEED:%d\n",SEED);
    int N_batch = trainSet.items_num / BATCH_SIZE;
    for(int j=0;j<EPOCH;j++){
        for(int i=0;i<N_batch;i++){
            in._data = &trainSet._image[i*BATCH_SIZE*CHANNEL*ROWS*COLS];
            label._data = &trainSet.label[i*BATCH_SIZE];
            nn.forward();
            nn.backward();
            nn.apply_grad();
            printf("Epoch:%2d [%5d/%-5d] Loss: %5.6f Acc: %3.6f%%\n",j+1,i+1,N_batch,nn.getAvgLoss(),nn.getAcc());
        }

        int acc_num = 0;
        int test_N_batch = testSet.items_num / BATCH_SIZE;
        for(int i=0;i<test_N_batch;i++){
            in._data = &testSet._image[i*BATCH_SIZE*CHANNEL*ROWS*COLS];
            label._data = &testSet.label[i*BATCH_SIZE];
            nn.forward();
            acc_num += nn.getAccNum();
        }
        printf("Epoch:%2d [Acc on Test] %3.6f%%\n",j+1,(float)acc_num/testSet.items_num*100);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("ElapsedTime: %f s\n",elapsedTime/1000);
}