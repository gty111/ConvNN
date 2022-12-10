#ifndef __NN__
#define __NN__

#include <vector>
#include "Layer.h"

class NN{
public:
    double lr;
    std::vector<Layer*>layers;

    void add(Layer *layer){
        layers.push_back(layer);
    }

    Tensor<double>* lastOut(){
        return layers.back()->_out;
    }

    void apply_grad(){
        for(auto e:layers){
            e->apply_grad();
        }
    }

    int pred(){
        auto layer = (LogSoftmax*)layers.back();
        return layer->_out->maxIdx();
    }

    double loss(){
        auto layer = (LogSoftmax*)layers.back();
        return layer->_out->_data[*layer->_label];
    }

    void initData(double mean,double std){
        for(auto e:layers){
            e->initData(mean,std);
        }
    }

    void forward(){
        for(auto e:layers){
            e->forward();
        }
    }

    void backward(){
        for(int i=layers.size()-1;i>=0;i--){
            layers[i]->backward();
        }
    }

    void setLR(double LR){
        lr = LR;
        for(auto e:layers){
            e->learn_rate = lr;
        }
    }
};

#endif