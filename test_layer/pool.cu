#include "Layer.h"

int main(){

    Tensor<float> in(1,1,4,4);
    in.initData(0,1);
    in.print();

    Pooling *conv = new Pooling(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,2,2,&in);
    conv->initData(1,0);
    conv->forward();
    cudaDeviceSynchronize();
    conv->_out->print();

    conv->_out->initGrad(1,0);
    conv->backward();
    cudaDeviceSynchronize();
    conv->_in->printgrad();

}