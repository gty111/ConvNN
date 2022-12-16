#include "Layer.h"

int main(){

    Tensor<float> in(2,5,1,1);
    in.initData(0,1);
    in.printTot();

    int label = 2;

    Softmax *conv = new Softmax(&in,&label);
    conv->forward();
    cudaDeviceSynchronize();
    conv->_out->print();

    conv->backward();
    cudaDeviceSynchronize();
    conv->_out->printGradTot();
    conv->_in->printGradTot();

}