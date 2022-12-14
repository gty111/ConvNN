#ifndef __NN__
#define __NN__

#include <vector>
#include "Layer.h"

class NN{
public:
    double _lr=1e-3;
    double _momentum=0.9;
    std::vector<Layer*>_layers;

    void add(Layer *layer){
        _layers.push_back(layer);
        _layers.back()->_out->printShape();
    }

    Tensor<double>* lastOut(){
        assert(_layers.size()&&"first layer must bind input");
        return _layers.back()->_out;
    }

    void apply_grad(){
        for(auto e:_layers){
            e->apply_grad();
        }
    }

    int pred(){
        auto layer = (LogSoftmax*)_layers.back();
        return layer->_out->maxIdx();
    }

    double loss(){
        auto layer = (LogSoftmax*)_layers.back();
        return layer->_out->_data[*layer->_label];
    }

    void initData(double mean,double std){
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

    void setLR(double lr){
        _lr = lr;
        for(auto e:_layers){
            e->learn_rate = _lr;
        }
    }

    void setMomentum(double momentum){
        _momentum = momentum;
        for(auto e:_layers){
            e->momentum = _momentum;
        }
    }
};

#endif