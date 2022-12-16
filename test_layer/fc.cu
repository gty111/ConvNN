#include "Layer.h"

int main(){

    Tensor<float> in(3,1,3,3);
    in.initData(1,0);
    in.print();

    Fc *conv = new Fc(4,&in);
    conv->initData(1,0);
    conv->_w->print();
    conv->_b->print();
    conv->forward();
    cudaDeviceSynchronize();
    conv->_out->printTot();
    

    conv->_out->initGrad(1,0);
    conv->backward();
    cudaDeviceSynchronize();
    conv->_in->printgrad();
    conv->_w->printgrad();
    conv->_b->printgrad();

    conv->apply_grad();
    cudaDeviceSynchronize();
    conv->_w->print();
    conv->_b->print();
}