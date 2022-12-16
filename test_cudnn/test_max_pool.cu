#include <cudnn.h>
#include <random>

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
std::normal_distribution<float> dis(0,1);

int main(){
    // create handle 
    cudnnHandle_t cuhandle;
    cudnnCreate(&cuhandle);

    // create tensor descriptor
    int n=1,c=1,h=6,w=6;
    cudnnTensorDescriptor_t tensor_des;
    cudnnCreateTensorDescriptor(&tensor_des);
    cudnnSetTensor4dDescriptor(tensor_des, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    // create out_tensor descriptor
    cudnnTensorDescriptor_t out_tensor_des;
    cudnnCreateTensorDescriptor(&out_tensor_des);
    cudnnSetTensor4dDescriptor(out_tensor_des, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h/2, w/2);

    // create maxpool function descriptor
    cudnnPoolingDescriptor_t pool_des;
    cudnnCreatePoolingDescriptor(&pool_des);
    cudnnSetPooling2dDescriptor(pool_des,CUDNN_POOLING_MAX,CUDNN_NOT_PROPAGATE_NAN,2,2,0,0,2,2);

    // create tensor
    float* tensor,*out_tensor;
    cudaMallocManaged(&tensor,n*c*h*w*sizeof(float));
    cudaMallocManaged(&out_tensor,n*c*h/2*w/2*sizeof(float));
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            tensor[i*w+j] = dis(gen);
            printf("%f ",tensor[i*w+j]);
        }
        printf("\n");
    }


    float alpha=1,beta=0;
    cudnnPoolingForward(cuhandle,pool_des,&alpha,tensor_des,tensor,&beta,out_tensor_des,out_tensor);

    cudaDeviceSynchronize();
    
    for(int i=0;i<h/2;i++){
        for(int j=0;j<w/2;j++){
            printf("%f ",out_tensor[i*w/2+j]);
        }
        printf("\n");
    }
}