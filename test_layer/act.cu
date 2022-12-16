#include "Layer.h"

int main(){

    Tensor<float> in(1,1,3,3);
    in.initData(0,1);
    in.print();

    Activation *conv = new Activation(CUDNN_ACTIVATION_RELU,&in);
    conv->initData(0,1);
    conv->forward();
    cudaDeviceSynchronize();
    conv->_out->print();
    

    conv->_out->initGrad(0,1);
    conv->backward();
    cudaDeviceSynchronize();
    conv->_in->printgrad();
}