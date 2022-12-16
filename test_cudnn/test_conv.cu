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
    int n=1,c=1,h=5,w=5;
    cudnnTensorDescriptor_t tensor_des;
    cudnnCreateTensorDescriptor(&tensor_des);
    cudnnSetTensor4dDescriptor(tensor_des, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);


    // create conv function descriptor
    cudnnConvolutionDescriptor_t conv_des;
    cudnnCreateConvolutionDescriptor(&conv_des);
    cudnnSetConvolution2dDescriptor(conv_des,1,1,1,1,1,1,CUDNN_CONVOLUTION,CUDNN_DATA_FLOAT);

    // create filter descripter
    int filter_w=3,filter_h=3;
    cudnnFilterDescriptor_t filter_des;
    cudnnCreateFilterDescriptor(&filter_des);
    cudnnSetFilter4dDescriptor(filter_des,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,c,c,filter_h,filter_w);

    // create input
    float* tensor;
    cudaMallocManaged(&tensor,n*c*h*w*sizeof(float));
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            tensor[i*w+j] = 1;
            printf("%f ",tensor[i*w+j]);
        }
        printf("\n");
    }

    // create output
    float* out_tensor;
    cudaMallocManaged(&out_tensor,n*c*filter_h*filter_w*sizeof(float));

    // create filter
    float *tensor_filter;
    cudaMallocManaged(&tensor_filter,n*c*filter_w*filter_h*sizeof(float));
    for(int i=0;i<n*c*filter_w*filter_h;i++){
        tensor_filter[i] = 1;
    }

    // workspace
    size_t size;
    cudnnGetConvolutionForwardWorkspaceSize(cuhandle,tensor_des,filter_des,conv_des,tensor_des,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,&size);
    printf("workspace size:%d\n",size);
    
    float alpha=1,beta=0;
    cudnnConvolutionForward( cuhandle
                            ,&alpha
                            ,tensor_des
                            ,tensor
                            ,filter_des
                            ,tensor_filter
                            ,conv_des
                            ,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
                            ,nullptr
                            ,0
                            ,&beta
                            ,tensor_des
                            ,out_tensor
                            );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){
        printf("%d\n",err);
    }
    
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            printf("%f ",out_tensor[i*w+j]);
        }
        printf("\n");
    }
}