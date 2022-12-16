#include "Layer.h"

int main(){

    Tensor<float> in(1,1,3,3);
    in.initData(23,32);
    in.print();

    BatchNorm2D *conv = new BatchNorm2D(&in);
    conv->initData(0,1);
    conv->forward();
    cudaDeviceSynchronize();
    conv->_out->print();
    

    conv->_out->initGrad(2,1);
    conv->backward();
    cudaDeviceSynchronize();
    conv->_in->printgrad();
    conv->_w->printgrad();
    conv->_b->printgrad();
}