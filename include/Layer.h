#ifndef __LAYER__
#define __LAYER__
#include "tensor.h"
#include <cstdio>

class Layer{
public:
    double learn_rate;
    double momentum;

    Tensor<double> *_in = nullptr;
    Tensor<double> *_out = nullptr;
    Tensor<double> *_w = nullptr;
    Tensor<double> *_b = nullptr;

    void initData(double mean,double std){
        if(_w)_w->initData(mean,std);
        if(_b)_w->initData(mean,std);
    }

    virtual void apply_grad(){
        if(_w){
            for(int i=0;i<(*_w).size;i++){
                (*_w)._lgrad[i] = (*_w)._lgrad[i]*momentum + (*_w)._grad[i];
                (*_w)._data[i] += (*_w)._lgrad[i] * learn_rate;
                (*_w)._grad[i] = 0;
            }
        }
        if(_b){
            for(int i=0;i<(*_b).size;i++){
                (*_b)._lgrad[i] = (*_b)._lgrad[i]*momentum + (*_b)._grad[i];
                (*_b)._data[i] += (*_b)._lgrad[i] * learn_rate;
                (*_b)._grad[i] = 0;
            }
        }
    }

    virtual void check() = 0;
    virtual void forward() = 0;
    virtual void backward() = 0;
};

class Conv:public Layer{
public:
    bool _PAD;
    Tensor<double> *_inpad = nullptr; // if PAD pad _in
    Tensor<double> *_pad = nullptr; // to cal _in grad

    Conv(int x,int y,bool PAD,Tensor<double>* in){
        assert(in);
        assert((*in).shape.size()==3);
        _PAD = PAD;
        _in = in;
        _w = new Tensor<double>(x,(*in).shape[0],y,y);
        _b = new Tensor<double>(x);
        int t;
        if(!PAD){
            t = (*in).shape[1]-y+1;
            _out = new Tensor<double>(x,t,t);
            t = (*_w).shape[2]+(*_in).shape[1]-1;
            _pad = new Tensor<double>(x,t,t);
        }else{
            t = (*in).shape[1];
            _out = new Tensor<double>(x,t,t);
            t = (*_w).shape[2]+(*_out).shape[1]-1;
            _pad = new Tensor<double>(x,t,t);
            _inpad = new Tensor<double>((*_in).shape[0],(*_in).shape[1]+2,(*_in).shape[2]+2);
        }
    }

    ~Conv(){
        delete _w;
        delete _b;
        delete _out;
        delete _pad;
    }

    virtual void check(){
        assert(_w);
        assert(_b);
        assert(_in);
        assert(_out);
        assert((*_w).shape.size()==4);
        assert((*_in).shape.size()==3);
        assert((*_out).shape.size()==3);
        assert((*_pad).shape.size()==3);
        assert((*_w).shape[1]==(*_in).shape[0]);
        assert((*_w).shape[0]==(*_b).shape[0]);
        assert((*_b).shape.size()==1);
        assert((*_out).shape[0]==(*_w).shape[0]);
        assert((*_pad).shape[0]==(*_w).shape[0]);
        if(!_PAD){
            assert((*_out).shape[1]==(*_in).shape[1]-(*_w).shape[2]+1);
            assert((*_out).shape[2]==(*_in).shape[2]-(*_w).shape[3]+1);
            assert((*_pad).shape[1]==(*_w).shape[2]+(*_in).shape[1]-1);
            assert((*_pad).shape[2]==(*_w).shape[3]+(*_in).shape[2]-1);
        }else{
            assert((*_out).shape[1]==(*_in).shape[1]);
            assert((*_out).shape[2]==(*_in).shape[2]);
            assert((*_pad).shape[1]==(*_w).shape[2]+(*_inpad).shape[1]-1);
            assert((*_pad).shape[2]==(*_w).shape[3]+(*_inpad).shape[2]-1);
        }
        
        
    }

    virtual void forward(){
        #ifdef DEBUG
        check();
        #endif

        if(!_PAD){
            for(int i=0;i<(*_w).shape[0];i++){
                for(int m=0;m<(*_out).shape[1];m++){
                    for(int n=0;n<(*_out).shape[2];n++){
                        double sum = 0;
                        for(int j=0;j<(*_w).shape[1];j++){
                            for(int p=0;p<(*_w).shape[2];p++){
                                for(int q=0;q<(*_w).shape[3];q++){
                                    sum += *(*_in).data(j,m+p,n+q) * (*(*_w).data(i,j,p,q));
                                }    
                            }
                        }
                        *(*_out).data(i,m,n) = sum + (*(*_b).data(i));
                    }
                }
            }
        }else{
            // cal _inpad
            for(int i=0;i<(*_in).shape[0];i++){
                for(int j=0;j<(*_in).shape[1];j++){
                    for(int k=0;k<(*_in).shape[2];k++){
                        *(*_inpad).data(i,j+((*_w).shape[2]-1)/2,k+((*_w).shape[3]-1)/2) = *(*_in).data(i,j,k);
                    }
                }
            }
            // conv
            for(int i=0;i<(*_w).shape[0];i++){
                for(int m=0;m<(*_out).shape[1];m++){
                    for(int n=0;n<(*_out).shape[2];n++){
                        double sum = 0;
                        for(int j=0;j<(*_w).shape[1];j++){
                            for(int p=0;p<(*_w).shape[2];p++){
                                for(int q=0;q<(*_w).shape[3];q++){
                                    sum += *(*_inpad).data(j,m+p,n+q) * (*(*_w).data(i,j,p,q));
                                }    
                            }
                        }
                        *(*_out).data(i,m,n) = sum + (*(*_b).data(i));
                    }
                }
            }
        }   
    }

    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif

        if(!_PAD){
            // cal _in grad
            for(int i=0;i<(*_out).shape[0];i++){
                for(int j=0;j<(*_out).shape[1];j++){
                    for(int k=0;k<(*_out).shape[2];k++){
                        *(*_pad).data(i,j+(*_w).shape[2]-1,k+(*_w).shape[3]-1) = *(*_out).grad(i,j,k);
                    }
                }
            }
            for(int i=0;i<(*_in).shape[0];i++){
                for(int j=0;j<(*_in).shape[1];j++){
                    for(int k=0;k<(*_in).shape[2];k++){
                        double sum = 0;
                        for(int m=0;m<(*_w).shape[0];m++){
                            for(int p=0;p<(*_w).shape[2];p++){
                                for(int q=0;q<(*_w).shape[3];q++){
                                    sum += *(*_pad).data(m,j+p,k+q) * (*(*_w).data(m,i,(*_w).shape[2]-1-p,(*_w).shape[3]-1-q));
                                }
                            }
                        }
                        *(*_in).grad(i,j,k) = sum;
                    }
                }
            }

            // cal _w grad
            for(int i=0;i<(*_w).shape[0];i++){
                for(int j=0;j<(*_w).shape[1];j++){
                    for(int p=0;p<(*_w).shape[2];p++){
                        for(int q=0;q<(*_w).shape[3];q++){
                            double sum = 0;
                            for(int m=0;m<(*_out).shape[1];m++){
                                for(int n=0;n<(*_out).shape[2];n++){
                                    sum += *(*_in).data(j,m+p,n+q) * (*(*_out).grad(i,m,n));
                                }
                            }
                            *(*_w).grad(i,j,p,q) += sum * this->learn_rate;
                        }
                    }
                }
            }

            // cal _b grad
            for(int i=0;i<(*_out).shape[0];i++){
                for(int j=0;j<(*_out).shape[1];j++){
                    for(int k=0;k<(*_out).shape[2];k++){
                        *(*_b).grad(i) += *(*_out).grad(i,j,k) * this->learn_rate;    
                    }
                }
            }
        }else{
            // cal _in grad
            for(int i=0;i<(*_out).shape[0];i++){
                for(int j=0;j<(*_out).shape[1];j++){
                    for(int k=0;k<(*_out).shape[2];k++){
                        *(*_pad).data(i,j+(*_w).shape[2],k+(*_w).shape[3]) = *(*_out).grad(i,j,k);
                    }
                }
            }
            for(int i=0;i<(*_inpad).shape[0];i++){
                for(int j=0;j<(*_inpad).shape[1];j++){
                    for(int k=0;k<(*_inpad).shape[2];k++){
                        double sum = 0;
                        for(int m=0;m<(*_w).shape[0];m++){
                            for(int p=0;p<(*_w).shape[2];p++){
                                for(int q=0;q<(*_w).shape[3];q++){
                                    sum += *(*_pad).data(m,j+p,k+q) * (*(*_w).data(m,i,(*_w).shape[2]-1-p,(*_w).shape[3]-1-q));
                                }
                            }
                        }
                        *(*_inpad).grad(i,j,k) = sum;
                    }
                }
            }
            for(int i=0;i<(*_in).shape[0];i++){
                for(int j=0;j<(*_in).shape[1];j++){
                    for(int k=0;k<(*_in).shape[2];k++){
                        *(*_in).grad(i,j,k) = *(*_inpad).grad(i,j+((*_w).shape[2]-1)/2,k+((*_w).shape[3]-1)/2);
                    }
                }
            }

            // cal _w grad
            for(int i=0;i<(*_w).shape[0];i++){
                for(int j=0;j<(*_w).shape[1];j++){
                    for(int p=0;p<(*_w).shape[2];p++){
                        for(int q=0;q<(*_w).shape[3];q++){
                            double sum = 0;
                            for(int m=0;m<(*_out).shape[1];m++){
                                for(int n=0;n<(*_out).shape[2];n++){
                                    sum += *(*_inpad).data(j,m+p,n+q) * (*(*_out).grad(i,m,n));
                                }
                            }
                            *(*_w).grad(i,j,p,q) += sum * this->learn_rate;
                        }
                    }
                }
            }

            // cal _b grad
            for(int i=0;i<(*_out).shape[0];i++){
                for(int j=0;j<(*_out).shape[1];j++){
                    for(int k=0;k<(*_out).shape[2];k++){
                        *(*_b).grad(i) += *(*_out).grad(i,j,k) * this->learn_rate;    
                    }
                }
            }
        }
    }
};

class Fc:public Layer{
public:

    Fc(int x,Tensor<double>*in){
        assert(in);
        assert((*in).shape.size()==1);
        _in = in;
        _w = new Tensor<double>((*in).shape[0],x);
        _b = new Tensor<double>(x);
        _out = new Tensor<double>(x);
    }

    ~Fc(){
        delete _w;
        delete _b;
    }

    virtual void check(){
        assert(_w);
        assert(_b);
        assert(_in);
        assert(_out);
        assert((*_in).shape.size()==1);
        assert((*_out).shape.size()==1);
        assert((*_w).shape.size()==2);
        assert((*_in).shape[0]==(*_w).shape[0]);
        assert((*_out).shape[0]==(*_w).shape[1]);
    }

    virtual void forward(){
        #ifdef DEBUG
        check();
        #endif
        for(int i=0;i<(*_w).shape[1];i++){
            double sum = 0;
            for(int j=0;j<(*_w).shape[0];j++){
                sum += (*(*_in).data(j)) * (*(*_w).data(j,i)) ;
            }
            (*(*_out).data(i)) = sum + (*(*_b).data(i));
        }
    }

    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif
        // ----_in grad----
        for(int i=0;i<(*_in).shape[0];i++){
            double sum = 0;
            for(int j=0;j<(*_out).shape[0];j++){
                sum += *(*_w).data(i,j) * (*(*_out).grad(j));
            }
            *(*_in).grad(i) = sum;
        }
        
        // ----_w grad----
        for(int i=0;i<(*_w).shape[0];i++){
            for(int j=0;j<(*_w).shape[1];j++){
                *(*_w).grad(i,j) += (*(*_out).grad(j)) * (*(*_in).data(i)) * this->learn_rate;
            }
        }

        // ----_b grad----
        for(int i=0;i<(*_b).shape[0];i++){
            *(*_b).grad(i) += (*(*_out).grad(i)) * this->learn_rate;
        }
    }
};

class Relu : public Layer{
public:

    Relu(Tensor<double>*in){
        assert(in);
        _in = in;
        if((*in).shape.size()==1)
            _out = new Tensor<double>((*in).shape[0]);
        else if ((*in).shape.size()==2)
            _out = new Tensor<double>((*in).shape[0],(*in).shape[1]);
        else if((*in).shape.size()==3)
            _out = new Tensor<double>((*in).shape[0],(*in).shape[1],(*in).shape[2]);
        else assert(0);
    }

    ~Relu(){    
        delete _out;
    }

    virtual void check(){
        assert(_in);
        assert(_out);
        assert((*_in).size==(*_out).size);
    }

    virtual void forward(){
        #ifdef DEBUG
        check();
        #endif
        for(int i=0;i<(*_in).size;i++){
            (*_out)._data[i] = (*_in)._data[i] > 0 ? (*_in)._data[i] : 0 ;
        }
    }

    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif
        // cal _in grad
        for(int i=0;i<(*_in).size;i++){
            (*_in)._grad[i] = (*_in)._data[i] > 0 ? (*_out)._grad[i] : 0 ;
        }
    }
};

class MaxPool:public Layer{
public:

    Tensor<uint8_t> *_max; // log max index

    MaxPool(Tensor<double>*in){
        in->printShape();
        assert(in);
        assert((*in).shape.size()==3);
        assert((*in).shape[1]%2==0);
        assert((*in).shape[2]%2==0);
        _in = in;
        _out = new Tensor<double>((*in).shape[0],(*in).shape[1]/2,(*in).shape[2]/2);
        _max = new Tensor<uint8_t>((*_out).shape[0],(*_out).shape[1],(*_out).shape[2]);
    }

    ~MaxPool(){
        delete _out;
    }

    virtual void check(){
        assert(_in);
        assert(_out);
        assert((*_out).shape.size()==3);
        assert((*_in).shape.size()==3);
        assert((*_out).shape[0]==(*_in).shape[0]);
        assert((*_out).shape[1]==(*_in).shape[1]/2);
        assert((*_out).shape[2]==(*_in).shape[2]/2);
    }
    virtual void forward(){
        #ifdef DEBUG
        check();
        #endif

        for(int k=0;k<(*_out).shape[0];k++){
            for(int i=0;i<(*_out).shape[1];i++){
                for(int j=0;j<(*_out).shape[2];j++){
                    *(*_out).data(k,i,j) = *(*_in).data(k,i*2,j*2);
                    *(*_max).data(k,i,j) = 0;
                    if(*(*_in).data(k,i*2+1,j*2)>*(*_out).data(k,i,j)){
                        *(*_out).data(k,i,j) = *(*_in).data(k,i*2+1,j*2);
                        *(*_max).data(k,i,j) = 1;
                    }
                    if(*(*_in).data(k,i*2,j*2+1)>*(*_out).data(k,i,j)){
                        *(*_out).data(k,i,j) = *(*_in).data(k,i*2,j*2+1);
                        *(*_max).data(k,i,j) = 2;
                    }
                    if(*(*_in).data(k,i*2+1,j*2+1)>*(*_out).data(k,i,j)){
                        *(*_out).data(k,i,j) = *(*_in).data(k,i*2+1,j*2+1);
                        *(*_max).data(k,i,j) = 3;
                    }
                }
            }
        }

    }

    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif

        // cal _in grad
        for(int k=0;k<(*_out).shape[0];k++){
            for(int i=0;i<(*_out).shape[1];i++){
                for(int j=0;j<(*_out).shape[2];j++){
                    switch(*(*_max).data(k,i,j)){
                    case 0: *(*_out).grad(k,i,j) = *(*_in).grad(k,i*2,j*2);break;
                    case 1: *(*_out).grad(k,i,j) = *(*_in).grad(k,i*2+1,j*2);break;
                    case 2: *(*_out).grad(k,i,j) = *(*_in).grad(k,i*2,j*2+1);break;
                    case 3: *(*_out).grad(k,i,j) = *(*_in).grad(k,i*2+1,j*2+1);break;
                    default:assert(0);
                    }   
                }
            }
        }
    }
};

class MeanPool:public Layer{
public:
    MeanPool(Tensor<double>*in){
        assert(in);
        assert((*in).shape.size()==3);
        assert((*in).shape[1]%2==0);
        assert((*in).shape[2]%2==0);
        _in = in;
        _out = new Tensor<double>((*in).shape[0],(*in).shape[1]/2,(*in).shape[2]/2);
    }

    ~MeanPool(){
        delete _out;
    }

    virtual void check(){
        assert(_in);
        assert(_out);
        assert((*_out).shape.size()==3);
        assert((*_in).shape.size()==3);
        assert((*_out).shape[0]==(*_in).shape[0]);
        assert((*_out).shape[1]==(*_in).shape[1]/2);
        assert((*_out).shape[2]==(*_in).shape[2]/2);
    }
    virtual void forward(){
        #ifdef DEBUG
        check();
        #endif

        for(int k=0;k<(*_out).shape[0];k++){
            for(int i=0;i<(*_out).shape[1];i++){
                for(int j=0;j<(*_out).shape[2];j++){
                    (*(*_out).data(k,i,j)) = 
                        ( *(*_in).data(k,i*2,j*2) + *(*_in).data(k,i*2,j*2+1) + 
                        *(*_in).data(k,i*2+1,j*2) + *(*_in).data(k,i*2+1,j*2+1))/4;
                }
            }
        }
    }

    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif

        // cal _in grad
        for(int k=0;k<(*_in).shape[0];k++){
            for(int i=0;i<(*_in).shape[1];i++){
                for(int j=0;j<(*_in).shape[2];j++){
                    *(*_in).grad(k,i,j) = *(*_out).grad(k,i/2,j/2)/4;
                }
            }
        }
    }
};

class Flattern:public Layer{
public:

    Flattern(Tensor<double>*in){
        assert(in);
        assert((*in).shape.size()==3);
        _in = in;
        _out = new Tensor<double>((*in).size);
    }

    ~Flattern(){
        delete _out;
    }

    virtual void check(){
        assert((*_in).shape.size()==3);
        assert((*_out).shape.size()==1);
        assert((*_in).size==(*_out).size);
    }

    virtual void forward(){
        #ifdef DEBUG
        check();
        #endif
        for(int i=0;i<(*_in).size;i++){
            (*_out)._data[i] = (*_in)._data[i];
        }
    }

    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif
        // cal _in grad
        for(int i=0;i<(*_in).size;i++){
            (*_in)._grad[i] = (*_out)._grad[i];
        }
    }
};

class LogSoftmax:public Layer{
public:
    int *_label;

    LogSoftmax(Tensor<double>*in,int *label){
        assert((*in).shape.size()==1);
        _in = in;
        _out = new Tensor<double>((*in).size);
        _label = label;
    }

    ~LogSoftmax(){
        delete _out;
    }

    virtual void check(){
        assert((*_in).shape.size()==1);
        assert((*_out).shape.size()==1);
        assert((*_in).size==(*_out).size);
        assert(*_label>=0&&*_label<(*_in).size);
    }

    virtual void forward(){
        #ifdef DEBUG
        check();
        #endif

        double maximum = (*_in).max();
        double sum = 0;
        for(int i=0;i<(*_in).shape[0];i++){
            sum += std::exp(*(*_in).data(i)-maximum);
        }
        for(int i=0;i<(*_in).shape[0];i++){
            (*(*_out).data(i)) = *(*_in).data(i) - maximum - std::log(sum);
        }
    }

    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif

        // cal _in grad
        double maximum = (*_in).max();

        double sum = 0;
        for(int i=0;i<(*_in).shape[0];i++){
            sum += std::exp(*(*_in).data(i)-maximum);
        }
        
        for(int i=0;i<(*_in).shape[0];i++){
            *(*_in).grad(i) = i==*_label ? 
                1 - std::exp(*(*_in).data(i)-maximum) / sum:
                - std::exp(*(*_in).data(i)-maximum) / sum  ;
        }
    }
};

#endif