#ifndef __CUKERNEL__
#define __CUKERNEL__

template<typename T>
__global__ void conv_bias_add(T *_out,T *_bias,int T0,int T1,int T2,int T3){
    int size = T0*T1*T2*T3;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=size)return;
    int bias_idx = (idx/(T2*T3)) % T1;
    _out[idx] += _bias[bias_idx];
}

template<typename T>
__global__ void cuapply_grad(T *arr_data,T *arr_grad,T *arr_lgrad,int size,T lr,T momentum){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=size)return;
    arr_lgrad[idx] = momentum * arr_lgrad[idx] + arr_grad[idx];
    arr_data[idx] += arr_lgrad[idx] * lr;
}

template<typename T>
__global__ void set_grad(T *_out,T *_grad,uint8_t *label,int size,int labelNum){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=size)return;

    _grad[idx] = idx%labelNum==label[idx/labelNum] ? 1 - _out[idx] : - _out[idx];
}

#endif