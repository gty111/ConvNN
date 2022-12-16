#ifndef __CIFAR_DATASET__
#define __CIFAR_DATASET__

#include <cstdint>
#include <string>
#include <cassert>
#include <cmath>
#include "myRandom.h"

class CIFAR_DataSet{
    public:
    int train_num = 50000;
    int test_num = 10000;
    float * _image_train, *_image_test;
    uint8_t * _image_byte_train, *_image_byte_test;
    uint8_t * _label_train, * _label_test;

    std::string label2str[10];

    void process_each(float *out,uint8_t *in,bool argumented){
        for(int k=0;k<3072;k++){
            out[k] = in[k];
        }
        if(argumented){
            randomCrop(out,4);
            randomHorizontalFlip(out);
        }
        for(int c=0;c<3;c++){
            float sum0=0,sum1=0,mean,std;
            for(int k=0;k<1024;k++){
                sum0 += out[c*1024+k];
            }
            mean = sum0/1024;
            for(int k=0;k<1024;k++){
                sum1 += std::pow(out[c*1024+k]-mean,2);
            }
            std = std::sqrt(sum1/1024);
            for(int k=0;k<1024;k++){
                out[c*1024+k] = (out[c*1024+k]-mean) / std;
            }
        }
    }

    void randomCrop(float *in,int padding){
        static float temp[3][32][32];
        memcpy(temp,in,3072*sizeof(float));
        int x_start=0,y_start=0,x,y;
        x_start = rand()%(2*padding);
        y_start = rand()%(2*padding);
        for(int c=0;c<3;c++){
            for(int i=0;i<32;i++){
                for(int j=0;j<32;j++){
                    x = x_start + i;
                    y = y_start + j;
                    if(x>=padding&&x<padding+32&&y>=padding&&y<padding+32){
                        in[c*1024+i*32+j] = temp[c][x-padding][y-padding];
                    }else{
                        in[c*1024+i*32+j] = 0;
                    }
                }
            }
        }
    }

    void randomHorizontalFlip(float *in){
        if(rand()%2)return;
        static float temp[3][32][32];
        memcpy(temp,in,3072*sizeof(float));
        for(int c=0;c<3;c++){
            for(int i=0;i<32;i++){
                for(int j=0;j<32;j++){
                    in[c*1024+i*32+j] = temp[c][i][31-j];
                }
            }
        }
    }

    CIFAR_DataSet(std::string dataPath){
        srand(SEED);
        cudaMallocManaged(&_image_train,50000*3072*sizeof(float));
        cudaMallocManaged(&_image_test,10000*3072*sizeof(float));
        _image_byte_train = (uint8_t*)malloc(50000*3072*sizeof(uint8_t));
        _image_byte_test = (uint8_t*)malloc(10000*3072*sizeof(uint8_t));
        cudaMallocManaged(&_label_train,50000*sizeof(uint8_t));
        cudaMallocManaged(&_label_test,10000*sizeof(uint8_t));
        // read metadata
        FILE *fp = fopen((dataPath+"/batches.meta.txt").c_str(),"r");
        assert(fp);
        for(int i=0;i<10;i++){
            label2str[i].resize(20);
            fscanf(fp,"%s",&label2str[i][0]);
            // printf("%s\n",label2str[i].c_str());
        }
        // read trainSet
        for(int i=1;i<=5;i++){
            fp = fopen((dataPath+"/data_batch_"+std::to_string(i)+".bin").c_str(),"r");
            assert(fp);
            for(int j=0;j<10000;j++){
                int idx = (i-1)*10000 + j;
                fread(&_label_train[idx],sizeof(uint8_t),1,fp);
                fread(&_image_byte_train[idx*3072],sizeof(uint8_t),3072,fp);
                process_each(&_image_train[idx*3072],&_image_byte_train[idx*3072],0);
            }
            fclose(fp);
        }
        // read testSet
        fp = fopen((dataPath+"/test_batch.bin").c_str(),"r");
        for(int j=0;j<10000;j++){
            fread(&_label_test[j],sizeof(uint8_t),1,fp);
            fread(&_image_byte_test[j*3072],sizeof(uint8_t),3072,fp);
            process_each(&_image_test[j*3072],&_image_byte_test[j*3072],0);
        }
        fclose(fp);
    }

    float* image_train(int idx){
        assert(idx>=0&&idx<50000);
        return &_image_train[idx*3072];
    }

    float* image_test(int idx){
        assert(idx>=0&&idx<10000);
        return &_image_test[idx*3072];
    }

    ~CIFAR_DataSet(){
        cudaFree(_image_train);
        cudaFree(_image_test);
        free(_image_byte_train);
        free(_image_byte_test);
        cudaFree(_label_train);
        cudaFree(_label_test);
    }
};

#endif