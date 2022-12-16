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
    int n=1,c=2,h=3,w=3;
    cudnnTensorDescriptor_t tensor_des;
    cudnnCreateTensorDescriptor(&tensor_des);
    cudnnSetTensor4dDescriptor(tensor_des, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);


    // create activation function descriptor
    cudnnActivationDescriptor_t relu;
    cudnnCreateActivationDescriptor(&relu);
    cudnnSetActivationDescriptor(relu,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,0);

    // create tensor
    float* tensor;
    cudaMallocManaged(&tensor,n*c*h*w*sizeof(float));
    for(int i=0;i<n*c*h*w;i++){
        tensor[i] = dis(gen);
        printf("%f ",tensor[i]);
    }
    printf("\n");


    float alpha=1,beta=0;
    cudnnActivationForward(cuhandle,relu,&alpha,tensor_des,tensor,&beta,tensor_des,tensor);

    cudaDeviceSynchronize();
    
    for(int i=0;i<n*c*h*w;i++){
        printf("%f ",tensor[i]);
    }
    printf("\n");
}