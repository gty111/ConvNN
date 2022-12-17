#include "Layer.h"

int main(){

    Tensor<float> in(2,5,1,1);
    in.initData(0,1);
    in.printTot();

    Tensor<uint8_t> label(2);
    label._data[0] = 1;
    label._data[1] = 2;

    Softmax *conv = new Softmax(&in,&label);
    conv->forward();
    cudaDeviceSynchronize();
    conv->_out->print();

    conv->backward();
    cudaDeviceSynchronize();
    conv->_out->printGradTot();
    conv->_in->printGradTot();

}