#ifndef __LAYER__
#define __LAYER__
#include "tensor.h"
#include "helper_cuda.h"
#include "common.h"
#include "cukernel.h"
#include <cstdio>
#include <cudnn.h>

class Layer{
public:
    std::string _name;
    // cuda related var
    int blockDim = 512;
    cudnnHandle_t _cuhandle;
    cudnnTensorFormat_t _tensorFormat=CUDNN_TENSOR_NCHW;
    cudnnDataType_t _dataType=CUDNN_DATA_FLOAT;
    cudnnNanPropagation_t _nanPropagation = CUDNN_NOT_PROPAGATE_NAN;
    cudnnTensorDescriptor_t _in_des;
    cudnnTensorDescriptor_t _out_des;
    cudnnTensorDescriptor_t _b_des;
    void *workspace=nullptr; 
    size_t workspace_size = 0; 
    float alpha=1,beta=0;

    // learning rate
    float _lr=1e-3;

    float _momentum=0;

    NN_MODE _nn_mode = NN_TRAIN;

    Tensor<float> *_in = nullptr;
    Tensor<float> *_out = nullptr;
    Tensor<float> *_w = nullptr;
    Tensor<float> *_b = nullptr;

    void initData(float mean,float std){
        if(_w)_w->initData(mean,std);
        if(_b)_b->initData(mean,std);
    }

    void apply_grad(){
        if(_w){
            int gridDim = _w->size / blockDim + 1;
            cuapply_grad<float><<<gridDim,blockDim>>>(_w->_data,_w->_grad,_w->_lgrad,_w->size,_lr,_momentum);
        }
        if(_b){
            int gridDim = _b->size / blockDim + 1;
            cuapply_grad<float><<<gridDim,blockDim>>>(_b->_data,_b->_grad,_b->_lgrad,_b->size,_lr,_momentum);
        }
    }

    virtual void check() = 0;
    virtual void forward() = 0;
    virtual void backward() = 0;

    ~Layer(){
        if(_w)delete _w;
        if(_b)delete _b;
        if(_out)delete _out; 
    }
};

class _Conv:public Layer{
public:
    cudnnFilterDescriptor_t _filter_des;
    cudnnConvolutionDescriptor_t _conv_des;
    cudnnConvolutionFwdAlgo_t _algo=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionBwdDataAlgo_t _bwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    cudnnConvolutionBwdFilterAlgo_t _bwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

    virtual void forward(){
        #ifdef DEBUG
        check();
        #endif
        CudnnSafeCall( cudnnConvolutionForward(
                                            _cuhandle,
                                            &alpha,
                                            _in_des,
                                            _in->_data,
                                            _filter_des,
                                            _w->_data,
                                            _conv_des,
                                            _algo,
                                            workspace,
                                            workspace_size,
                                            &beta,
                                            _out_des,
                                            _out->_data) );
        
        // add bias
        int gridDim = _out->size / blockDim + 1;
        conv_bias_add<float><<<gridDim,blockDim>>>( _out->_data,
                                                    _b->_data,
                                                    _out->shape[0],
                                                    _out->shape[1],
                                                    _out->shape[2],
                                                    _out->shape[3]);
        
    }

    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif
        CudnnSafeCall( cudnnConvolutionBackwardData(
        _cuhandle,
        &alpha,
        _filter_des,
        _w->_data,
        _out_des,
        _out->_grad,
        _conv_des,
        _bwdDataAlgo,
        workspace,
        workspace_size,
        &beta,
        _in_des,
        _in->_grad
        ));

        CudnnSafeCall( cudnnConvolutionBackwardFilter(
        _cuhandle,
        &alpha,
        _in_des,
        _in->_data,
        _out_des,
        _out->_grad,
        _conv_des,
        _bwdFilterAlgo,
        workspace,
        workspace_size,
        &beta,
        _filter_des,
        _w->_grad
        ));

        CudnnSafeCall( cudnnConvolutionBackwardBias(
        _cuhandle,  
        &alpha,
        _out_des,
        _out->_grad,
        &beta,
        _b_des,
        _b->_grad
        ));

    }

    virtual void check(){
        assert(_cuhandle);
        assert(_w);
        assert(_b);
        assert(_in);
        assert(_out);
        assert((*_w).shape.size()==4);
        assert((*_in).shape.size()==4);
        assert((*_out).shape.size()==4);
        assert((*_w).shape[1]==(*_in).shape[1]);
        assert((*_w).shape[0]==(*_b).shape[0]);
        assert((*_b).shape.size()==1);
        assert((*_out).shape[1]==(*_w).shape[0]);
    }

};

class Conv:public _Conv{
public:
    bool _PAD;

    /**
     * x : number of output feature maps
     * y : height or widht of each filter
     * PAD : whether pad
    */
    Conv(int x,int y,bool PAD,Tensor<float>* in){
        _name = "Conv";
        assert(in);
        assert(in->shape.size()==4);
        assert(in->shape[2]==in->shape[3]);
        _in = in;
        _w = new Tensor<float>(x,in->shape[1],y,y);
        _b = new Tensor<float>(x);

        int pad_h,pad_w;
        if(!PAD){
            pad_h = 0;
            pad_w = 0;
            _out = new Tensor<float>(in->shape[0],x,in->shape[2]-y+1,in->shape[3]-y+1);
        }else{
            pad_h = (y-1)/2;
            pad_w = (y-1)/2;
            _out = new Tensor<float>(in->shape[0],x,in->shape[2],in->shape[3]);
        }

        CudnnSafeCall( cudnnCreate(&_cuhandle) );

        CudnnSafeCall( cudnnCreateFilterDescriptor(&_filter_des) );
        CudnnSafeCall( cudnnSetFilter4dDescriptor(
        _filter_des,
        _dataType,
        _tensorFormat,
        x,
        in->shape[1],
        y,
        y
        ));

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_in_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _in_des,
        _tensorFormat,
        _dataType,
        in->shape[0],
        in->shape[1],
        in->shape[2],
        in->shape[3]
        ));

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_b_des) );
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _b_des,
        _tensorFormat,
        _dataType,
        1,
        _b->shape[0],
        1,
        1
        ));

        CudnnSafeCall( cudnnCreateConvolutionDescriptor(&_conv_des) );
        CudnnSafeCall( cudnnSetConvolution2dDescriptor(
        _conv_des,
        pad_h,
        pad_w,
        1,
        1,
        1,
        1,
        CUDNN_CONVOLUTION,
        _dataType
        ));

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_out_des) );
        CudnnSafeCall( cudnnSetTensor4dDescriptor( 
        _out_des,
        _tensorFormat,
        _dataType,
        _out->shape[0],
        _out->shape[1],
        _out->shape[2],
        _out->shape[3]
        ));

        size_t t_size;
        CudnnSafeCall( cudnnGetConvolutionBackwardDataWorkspaceSize(
        _cuhandle,
        _filter_des,
        _out_des,
        _conv_des,
        _in_des,
        _bwdDataAlgo,
        &t_size
        ));
        workspace_size = max(workspace_size,t_size);

        CudnnSafeCall( cudnnGetConvolutionForwardWorkspaceSize(
        _cuhandle,
        _in_des,
        _filter_des,
        _conv_des,
        _out_des,
        _algo,
        &t_size
        ));
        workspace_size = max(workspace_size,t_size);

        CudnnSafeCall( cudnnGetConvolutionBackwardFilterWorkspaceSize(
        _cuhandle,
        _in_des,
        _out_des,
        _conv_des,
        _filter_des,
        _bwdFilterAlgo,
        &t_size)
        );
        workspace_size = max(workspace_size,t_size);

        CudaSafeCall( cudaMalloc(&workspace,workspace_size) );
    }

    virtual void check(){
        assert(_cuhandle);
        assert(_w);
        assert(_b);
        assert(_in);
        assert(_out);
        assert((*_w).shape.size()==4);
        assert((*_in).shape.size()==4);
        assert((*_out).shape.size()==4);
        assert((*_w).shape[1]==(*_in).shape[1]);
        assert((*_w).shape[0]==(*_b).shape[0]);
        assert((*_b).shape.size()==1);
        assert((*_out).shape[1]==(*_w).shape[0]);
        if(!_PAD){
            assert((*_out).shape[1]==(*_in).shape[2]-(*_w).shape[2]+1);
            assert((*_out).shape[2]==(*_in).shape[3]-(*_w).shape[3]+1);
        }else{
            assert((*_out).shape[1]==(*_in).shape[2]);
            assert((*_out).shape[2]==(*_in).shape[3]);
        }
    }
};

class Fc:public _Conv{
public:
    /**
     * x : shape of output
     * (n,a,b,c) => (n,x,1,1)
    */
    Fc(int x,Tensor<float>*in){
        _name = "Fc";
        assert(in);
        assert(in->shape.size()==4);
        _in = in;
        _b = new Tensor<float>(x);
        _out = new Tensor<float>(in->shape[0],x,1,1);
        _w = new Tensor<float>(x,in->shape[1],in->shape[2],in->shape[3]);

        CudnnSafeCall( cudnnCreate(&_cuhandle) );

        CudnnSafeCall( cudnnCreateFilterDescriptor(&_filter_des) );
        CudnnSafeCall( cudnnSetFilter4dDescriptor(
        _filter_des,
        _dataType,
        _tensorFormat,
        x,
        in->shape[1],
        in->shape[2],
        in->shape[3]
        ));

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_in_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _in_des,
        _tensorFormat,
        _dataType,
        in->shape[0],
        in->shape[1],
        in->shape[2],
        in->shape[3]
        ));

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_out_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _out_des,
        _tensorFormat,
        _dataType,
        in->shape[0],
        x,
        1,
        1));

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_b_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _b_des,
        _tensorFormat,
        _dataType,
        1,
        x,
        1,
        1));

        CudnnSafeCall( cudnnCreateConvolutionDescriptor(&_conv_des) );
        CudnnSafeCall( cudnnSetConvolution2dDescriptor(
        _conv_des,
        0,
        0,
        1,
        1,
        1,
        1,
        CUDNN_CONVOLUTION,
        _dataType
        ));

        size_t t_size;
        CudnnSafeCall( cudnnGetConvolutionBackwardDataWorkspaceSize(
        _cuhandle,
        _filter_des,
        _out_des,
        _conv_des,
        _in_des,
        _bwdDataAlgo,
        &t_size
        ));
        workspace_size = max(workspace_size,t_size);

        CudnnSafeCall( cudnnGetConvolutionForwardWorkspaceSize(
        _cuhandle,
        _in_des,
        _filter_des,
        _conv_des,
        _out_des,
        _algo,
        &t_size
        ));
        workspace_size = max(workspace_size,t_size);

        CudnnSafeCall( cudnnGetConvolutionBackwardFilterWorkspaceSize(
        _cuhandle,
        _in_des,
        _out_des,
        _conv_des,
        _filter_des,
        _bwdFilterAlgo,
        &t_size)
        );
        workspace_size = max(workspace_size,t_size);

        CudaSafeCall( cudaMalloc(&workspace,workspace_size) );
    }
};

class Activation : public Layer{
public:

    cudnnActivationDescriptor_t _act_des;
    double _coef = 0;

    Activation(cudnnActivationMode_t activationMode,Tensor<float>*in){
        _name = "Activation";
        assert(in);
        assert(in->shape.size()==4);
        _in = in;
        _out = new Tensor<float>(in->shape[0],in->shape[1],in->shape[2],in->shape[3]);

        CudnnSafeCall( cudnnCreate(&_cuhandle) );

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_in_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _in_des,
        _tensorFormat,
        _dataType,
        in->shape[0],
        in->shape[1],
        in->shape[2],
        in->shape[3]
        ));

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_out_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _out_des,
        _tensorFormat,
        _dataType,
        in->shape[0],
        in->shape[1],
        in->shape[2],
        in->shape[3]
        ));

        CudnnSafeCall( cudnnCreateActivationDescriptor(&_act_des) );
        CudnnSafeCall( cudnnSetActivationDescriptor(
        _act_des,
        activationMode,
        _nanPropagation,
        _coef
        ));

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
        
        CudnnSafeCall( cudnnActivationForward(
        _cuhandle,
        _act_des,
        &alpha,
        _in_des,
        _in->_data,
        &beta,
        _out_des,
        _out->_data
        ));
    }

    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif
        
        CudnnSafeCall( cudnnActivationBackward(
        _cuhandle,
        _act_des,
        &alpha,
        _out_des,
        _out->_data,
        _out_des,
        _out->_grad,
        _in_des,
        _in->_data,
        &beta,
        _in_des,
        _in->_grad 
        ));
    }
};

class Pooling:public Layer{
public:

    cudnnPoolingDescriptor_t _pool_des;

    /**
     * x: size of window
     * y: stride of window
    */
    Pooling(cudnnPoolingMode_t poolingMode,int x,int y,Tensor<float>*in){
        _name = "Pooling";
        assert(in);
        assert((*in).shape.size()==4);
        assert((*in).shape[2]%2==0);
        assert((*in).shape[3]%2==0);
        _in = in;
        _out = new Tensor<float>(in->shape[0],in->shape[1],in->shape[2]/2,in->shape[3]/2);

        CudnnSafeCall( cudnnCreate(&_cuhandle) );

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_in_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _in_des,
        _tensorFormat,
        _dataType,
        in->shape[0],
        in->shape[1],
        in->shape[2],
        in->shape[3]
        ));

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_out_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _out_des,
        _tensorFormat,
        _dataType,
        in->shape[0],
        in->shape[1],
        in->shape[2]/2,
        in->shape[3]/2
        ));

        CudnnSafeCall( cudnnCreatePoolingDescriptor(&_pool_des) );
        CudnnSafeCall( cudnnSetPooling2dDescriptor(
        _pool_des,
        poolingMode,
        _nanPropagation,
        x,
        x,
        0,
        0,
        y,
        y
        ));

    }

    virtual void check(){
        assert(_in);
        assert(_out);
        assert((*_out).shape.size()==4);
        assert((*_in).shape.size()==4);
        assert((*_out).shape[0]==(*_in).shape[0]);
        assert((*_out).shape[1]==(*_in).shape[1]);
        assert((*_out).shape[2]==(*_in).shape[2]/2);
        assert((*_out).shape[3]==(*_in).shape[3]/2);
    }
    virtual void forward(){
        #ifdef DEBUG
        check();
        #endif

        CudnnSafeCall( cudnnPoolingForward(
        _cuhandle,
        _pool_des,
        &alpha,
        _in_des,
        _in->_data,
        &beta,
        _out_des,
        _out->_data
        ));

    }

    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif

        CudnnSafeCall( cudnnPoolingBackward(
        _cuhandle,
        _pool_des,
        &alpha,
        _out_des,
        _out->_data,
        _out_des,
        _out->_grad,
        _in_des,
        _in->_data,
        &beta,
        _in_des,
        _in->_grad
        ));
    }
};

// https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
// https://arxiv.org/abs/1502.03167
class BatchNorm2D:public Layer{
public:
    cudnnTensorDescriptor_t  bnScaleBiasMeanVarDesc;
    cudnnBatchNormMode_t _mode = CUDNN_BATCHNORM_SPATIAL;
    float _epsilon = 1e-5;
    float exponentialAverageFactor = 0.1;
    float *resultRunningMean,*resultRunningVariance;
    float *resultSaveMean,*resultSaveInvVariance;

    BatchNorm2D(Tensor<float>*in){
        _name = "BatchNorm2D";
        assert(in);
        assert(in->shape.size()==4);

        _in = in;
        _out = new Tensor<float>(in->shape[0],in->shape[1],in->shape[2],in->shape[3]);

        CudnnSafeCall( cudnnCreate(&_cuhandle) );

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_in_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _in_des,
        _tensorFormat,
        _dataType,
        in->shape[0],
        in->shape[1],
        in->shape[2],
        in->shape[3]
        ));

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_out_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _out_des,
        _tensorFormat,
        _dataType,
        in->shape[0],
        in->shape[1],
        in->shape[2],
        in->shape[3]
        ));

        CudnnSafeCall( cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc) ); 
        CudnnSafeCall( cudnnDeriveBNTensorDescriptor(
        bnScaleBiasMeanVarDesc,
        _in_des,
        _mode
        ));

        int n,c,h,w,nstride,cstride,hstride,wstride;
        cudnnDataType_t dtype;
        CudnnSafeCall( cudnnGetTensor4dDescriptor(
        bnScaleBiasMeanVarDesc,
        &dtype,
        &n,
        &c,
        &h,
        &w,
        &nstride,
        &cstride,
        &hstride,
        &wstride
        )); 

        _w = new Tensor<float>(n,c,h,w);
        _b = new Tensor<float>(n,c,h,w);
        _w->initData(1,0);
        _b->initData(0,0);

        cudaMalloc(&resultRunningMean,n*c*h*w*sizeof(float));
        cudaMalloc(&resultRunningVariance,n*c*h*w*sizeof(float));
        cudaMalloc(&resultSaveMean,n*c*h*w*sizeof(float));
        cudaMalloc(&resultSaveInvVariance,n*c*h*w*sizeof(float));
    }

    virtual void check(){
        assert(_out->shape[0]==_in->shape[0]);
        assert(_out->shape[1]==_in->shape[1]);
        assert(_out->shape[2]==_in->shape[2]);
        assert(_out->shape[3]==_in->shape[3]);
    }

    virtual void forward(){
        #ifdef DEBUG
        check();
        #endif

        if(_nn_mode==NN_TRAIN)
            CudnnSafeCall( cudnnBatchNormalizationForwardTraining(
            _cuhandle,
            _mode,
            &alpha,
            &beta,
            _in_des,
            _in->_data,
            _out_des,
            _out->_data,
            bnScaleBiasMeanVarDesc,
            _w->_data,
            _b->_data,
            exponentialAverageFactor,
            resultRunningMean,
            resultRunningVariance,
            _epsilon,
            resultSaveMean,
            resultSaveInvVariance
            ));
        else if(_nn_mode==NN_INFERENCE)
            CudnnSafeCall( cudnnBatchNormalizationForwardInference(
            _cuhandle,
            _mode,
            &alpha,
            &beta,
            _in_des,
            _in->_data,
            _out_des,
            _out->_data,
            bnScaleBiasMeanVarDesc,
            _w->_data,
            _b->_data,
            resultRunningMean,
            resultRunningVariance,
            _epsilon
            ));
        else assert(0);
    }

    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif
        
        CudnnSafeCall( cudnnBatchNormalizationBackward(
        _cuhandle,
        _mode,
        &alpha,
        &beta,
        &alpha,
        &beta,
        _in_des,
        _in->_data,
        _out_des,
        _out->_grad,
        _in_des,
        _in->_grad,
        bnScaleBiasMeanVarDesc,
        _w->_data,
        _w->_grad,
        _b->_grad,
        _epsilon,
        resultSaveMean,
        resultSaveInvVariance
        ));
    }
};

class Softmax:public Layer{
public:
    cudnnSoftmaxAlgorithm_t _algo = CUDNN_SOFTMAX_ACCURATE;
    cudnnSoftmaxMode_t _mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    Tensor<uint8_t>*_label;

    Softmax(Tensor<float>*in,Tensor<uint8_t>*label){
        _name = "SoftMax";
        assert(label);
        assert(in);
        assert(in->shape.size()==4);
        assert(in->shape[2]==1&&in->shape[3]==1);
        _in = in;
        _out = new Tensor<float>(in->shape[0],in->shape[1],in->shape[2],in->shape[3]);
        _label = label;

        CudnnSafeCall( cudnnCreate(&_cuhandle) );

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_in_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _in_des,
        _tensorFormat,
        _dataType,
        in->shape[0],
        in->shape[1],
        in->shape[2],
        in->shape[3]
        ));

        CudnnSafeCall( cudnnCreateTensorDescriptor(&_out_des) ); 
        CudnnSafeCall( cudnnSetTensor4dDescriptor(
        _out_des,
        _tensorFormat,
        _dataType,
        in->shape[0],
        in->shape[1],
        in->shape[2],
        in->shape[3]
        ));
    }

    virtual void check(){
        assert((*_in).shape.size()==4);
        assert((*_out).shape.size()==4);
        assert((*_in).size==(*_out).size);
    }

    virtual void forward(){
        #ifdef DEBUG
        check();
        #endif

        CudnnSafeCall( cudnnSoftmaxForward(
        _cuhandle,
        _algo,
        _mode,
        &alpha,
        _in_des,
        _in->_data,
        &beta,
        _out_des,
        _out->_data
        ));
    }
    
    /**
     * before call backward, you must set _label(device pointer)
    */
    virtual void backward(){
        #ifdef DEBUG
        check();
        #endif

        int gridDim = _out->size / blockDim + 1;
        set_grad<float><<<gridDim,blockDim>>>(_out->_data,_out->_grad,_label->_data,_out->size,_out->shape[1]);

        CudnnSafeCall( cudnnSoftmaxBackward(
        _cuhandle,
        _algo,
        _mode,
        &alpha,
        _out_des,
        _out->_data,
        _out_des,
        _out->_grad,
        &beta,
        _in_des,
        _in->_grad
        ));
    }
};

#endif