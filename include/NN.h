#ifndef __NN__
#define __NN__

#include <vector>
#include <map>
#include "Layer.h"
#include <cmath>

class NN{
public:
    float _lr=1e-3;
    float _momentum=0;
    std::vector<Layer*>_layers;

    void add(Layer *layer){
        printf("%12s (%3d): ",layer->_name.c_str(),_layers.size());
        _layers.push_back(layer);
        _layers.back()->_out->printShape();
    }

    Tensor<float>* lastOut(){
        assert(_layers.size()&&"first layer must bind input");
        return _layers.back()->_out;
    }

    void apply_grad(){
        for(auto e:_layers){
            e->apply_grad();
        }
    }

    float getAvgLoss(){
        cudaDeviceSynchronize();
        float loss = 0;
        auto t = (Softmax*)_layers.back();
        for(int i=0;i<t->_out->shape[0];i++){
            loss += *t->_out->data(i,t->_label->_data[i],0,0);
        }
        return std::log(loss / t->_out->shape[0]);
    }

    float getAcc(){
        cudaDeviceSynchronize();
        int acc_num = 0;
        auto t = (Softmax*)_layers.back();
        for(int i=0;i<t->_out->shape[0];i++){
            bool flag = 1;
            for(int j=0;j<t->_out->shape[1];j++){
                if(j==t->_label->_data[i])continue;
                if(*t->_out->data(i,t->_label->_data[i],0,0) <= *t->_out->data(i,j,0,0)){
                    flag = 0;
                    break;
                }
            }
            acc_num += flag;
        }
        return 1.0*acc_num/t->_out->shape[0]*100;
    }

    int getAccNum(){
        cudaDeviceSynchronize();
        int acc_num = 0;
        auto t = (Softmax*)_layers.back();
        for(int i=0;i<t->_out->shape[0];i++){
            bool flag = 1;
            for(int j=0;j<t->_out->shape[1];j++){
                if(j==t->_label->_data[i])continue;
                if(*t->_out->data(i,t->_label->_data[i],0,0) <= *t->_out->data(i,j,0,0)){
                    flag = 0;
                    break;
                }
            }
            acc_num += flag;
        }
        return acc_num;
    }

    void initData(float mean,float std){
        for(auto e:_layers){
            e->initData(mean,std);
        }
    }

    void forward(){
        for(auto e:_layers){
            e->forward();
        }
    }

    void backward(){
        for(int i=_layers.size()-1;i>=0;i--){
            _layers[i]->backward();
        }
    }

    void setLR(float lr){
        _lr = lr;
        for(auto e:_layers){
            e->_lr = _lr;
        }
    }

    void setmomentum(float momentum){
        _momentum = momentum;
        for(auto e:_layers){
            e->_momentum = momentum;
        }
    }

    void setMode(NN_MODE mode){
        for(auto e:_layers){
            e->_nn_mode = mode;
        }
    }
};

#endif