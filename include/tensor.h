#ifndef __TENSOR__
#define __TENSOR__
#include<vector>
#include<cassert>
#include"myRandom.h"
#include"helper_cuda.h"

template<typename T>
class Tensor{
    public:
    int size = 0;
    std::vector<int>shape;
    T *_data=nullptr; 
    T *_grad=nullptr;
    T *_lgrad=nullptr;
    bool _alloc=0;
    bool _init_data=0;

    void initData(double mean,double std){
        if(!_init_data){
            std::normal_distribution<double> dis(mean,std);
            for(int i=0;i<size;i++){
                _data[i] = dis(gen);
                _lgrad[i] = 0;
            }
            _init_data = 1;
        }
    }

    void initGrad(double mean,double std){
        std::normal_distribution<double> dis(mean,std);
        for(int i=0;i<size;i++){
            _grad[i] = dis(gen);
        }
    }

    void _allocData(){
        _alloc = 1;
        CudaSafeCall( cudaMallocManaged(&_data,size*sizeof(T)) );
        CudaSafeCall( cudaMallocManaged(&_grad,size*sizeof(T)) );
        CudaSafeCall( cudaMallocManaged(&_lgrad,size*sizeof(T)) );
    }

    Tensor(int a,bool alloc=1){
        shape.push_back(a);
        size = a;
        if(alloc)_allocData();
    }

    Tensor(int a,int b,bool alloc=1){
        shape.push_back(a);
        shape.push_back(b);
        size = a * b;
        if(alloc)_allocData();
    }

    Tensor(int a,int b,int c,bool alloc=1){
        shape.push_back(a);
        shape.push_back(b);
        shape.push_back(c);
        size = a * b * c;
        if(alloc)_allocData();
    }

    Tensor(int a,int b,int c,int d,bool alloc=1){
        shape.push_back(a);
        shape.push_back(b);
        shape.push_back(c);
        shape.push_back(d);
        size = a * b * c * d;
        if(alloc)_allocData();
    }

    T max(){
        T maximum = _data[0];
        for(int i=1;i<size;i++){
            maximum = std::max(maximum,_data[i]);
        }
        return maximum;
    }

    int maxIdx(){
        int maxidx = 0;
        T maximum = _data[0];
        for(int i=1;i<size;i++){
            if(_data[i]>maximum){
                maximum = _data[i];
                maxidx = i;
            }
        }
        return maxidx;
    }

    void set(T *from){
        for(int i=0;i<size;i++){
            _data[i] = from[i];
        }
    }

    T* data(int idx0){
        #ifdef DEBUG
        assert(shape.size()==1);
        assert(idx0>=0&&idx0<shape[0]);
        #endif
        return &_data[idx0];
    }
    T* data(int idx0,int idx1){
        #ifdef DEBUG
        assert(shape.size()==2);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        #endif
        return &_data[idx0*shape[1]+idx1];
    }
    T* data(int idx0,int idx1,int idx2){
        #ifdef DEBUG
        assert(shape.size()==3);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        assert(idx2>=0&&idx2<shape[2]);
        #endif
        return &_data[idx0*shape[1]*shape[2]+idx1*shape[2]+idx2];
    }

    T* data(int idx0,int idx1,int idx2,int idx3){
        #ifdef DEBUG
        assert(shape.size()==4);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        assert(idx2>=0&&idx2<shape[2]);
        assert(idx3>=0&&idx3<=shape[3]);
        #endif
        return &_data[idx0*shape[1]*shape[2]*shape[3]+idx1*shape[2]*shape[3]+idx2*shape[3]+idx3];
    }

    T* grad(int idx0){
        #ifdef DEBUG
        assert(shape.size()==1);
        assert(idx0>=0&&idx0<shape[0]);
        #endif
        return &_grad[idx0];
    }
    T* grad(int idx0,int idx1){
        #ifdef DEBUG
        assert(shape.size()==2);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        #endif
        return &_grad[idx0*shape[1]+idx1];
    }
    T* grad(int idx0,int idx1,int idx2){
        #ifdef DEBUG
        assert(shape.size()==3);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        assert(idx2>=0&&idx2<shape[2]);
        #endif
        return &_grad[idx0*shape[1]*shape[2]+idx1*shape[2]+idx2];
    }

    T* grad(int idx0,int idx1,int idx2,int idx3){
        #ifdef DEBUG
        assert(shape.size()==4);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        assert(idx2>=0&&idx2<shape[2]);
        assert(idx3>=0&&idx3<=shape[3]);
        #endif
        return &_grad[idx0*shape[1]*shape[2]*shape[3]+idx1*shape[2]*shape[3]+idx2*shape[3]+idx3];
    }

    void std(){
        double sum0=0,sum1=0,mean,std;
        for(int i=0;i<size;i++){
            sum0 += _data[i];
        }
        mean = sum0 / size;
        for(int i=0;i<size;i++){
            sum1 += (_data[i]-mean) * (_data[i]-mean);
        }
        std = std::sqrt(sum1 / size);
        for(int i=0;i<size;i++){
            _data[i] = (_data[i]-mean) / std;
        }
    }

    void printTot(){
        printf("---------------\n");
        for(int i=0;i<size;i++){
            printf("%d:% 5.6f\n",i,_data[i]);
        }
    }

    void print(){
        printf("---------------\n");
        if(shape.size()==4){
            for(int k=0;k<shape[1];k++)
            for(int i=0;i<shape[2];i++){
                for(int j=0;j<shape[3];j++){
                    printf("% 5.6f ",*(data(0,k,i,j)));
                }
                printf("\n");
            }
        }else if(shape.size()==3){
            // for(int k=0;k<shape[0];k++)
            for(int i=0;i<shape[1];i++){
                for(int j=0;j<shape[2];j++){
                    printf("% 5.6f ",*(data(0,i,j)));
                }
                printf("\n");
            }
        }else if(shape.size()==2){
            for(int i=0;i<shape[0];i++){
                for(int j=0;j<shape[1];j++){
                    printf("% 5.6f ",*(data(i,j)));
                }
                printf("\n");
            }
        }else if(shape.size()==1){
            for(int j=0;j<shape[0];j++){
                printf("% 5.6f\n",*(data(j)));
            }
        } 
    }

    void printGradTot(){
        printf("---------------\n");
        for(int i=0;i<size;i++){
            printf("% 5.6f\n",_grad[i]);
        }
    }

    void printgrad(){
        printf("---------------\n");
        if(shape.size()==4){
            for(int k=0;k<shape[1];k++)
            for(int i=0;i<shape[2];i++){
                for(int j=0;j<shape[3];j++){
                    printf("% 5.6f ",*(grad(0,k,i,j)));
                }
                printf("\n");
            }
        }else if(shape.size()==3){
            // for(int k=0;k<shape[0];k++)
            for(int i=0;i<shape[1];i++){
                for(int j=0;j<shape[2];j++){
                    printf("% 5.6f ",*(grad(0,i,j)));
                }
                printf("\n");
            }
        }else if(shape.size()==2){
            for(int i=0;i<shape[0];i++){
                for(int j=0;j<shape[1];j++){
                    printf("% 5.6f ",*(grad(i,j)));
                }
                printf("\n");
            }
        }else if(shape.size()==1){
            for(int j=0;j<shape[0];j++){
                printf("% 5.6f\n",*(grad(j)));
            }
        } 
    }

    void printShape(){
        for(auto e:shape){
            printf("%4d ",e);
        }
        printf("\n");
    }

    ~Tensor(){
        if(_alloc)cudaFree(_data);
        if(_alloc)cudaFree(_grad);
    }
};

#endif