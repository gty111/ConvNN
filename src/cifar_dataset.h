#ifndef __CIFAR_DATASET__
#define __CIFAR_DATASET__

#include <cstdint>
#include <string>
#include <cassert>
#include <cmath>

class DataSet{
    public:
    int train_num = 50000;
    int test_num = 10000;
    double * _image_train, *_image_test;
    uint8_t * _image_byte_train, *_image_byte_test;
    uint8_t * _label_train, * _label_test;

    std::string label2str[10];

    DataSet(std::string dataPath){
        _image_train = (double*)malloc(50000*3072*sizeof(double));
        _image_test = (double*)malloc(10000*3072*sizeof(double));
        _image_byte_train = (uint8_t*)malloc(50000*3072*sizeof(uint8_t));
        _image_byte_test = (uint8_t*)malloc(10000*3072*sizeof(uint8_t));
        _label_train = (uint8_t*)malloc(50000*sizeof(uint8_t));
        _label_test = (uint8_t*)malloc(10000*sizeof(uint8_t));
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
                for(int k=0;k<3072;k++){
                    _image_train[idx*3072+k] = (double)_image_byte_train[idx*3072+k]/255;
                }
                // double sum0=0,sum1=0,mean,std;
                // for(int k=0;k<3072;k++){
                //     _image_train[idx*3072+k] = (double)_image_byte_train[idx*3072+k];
                //     sum0 += _image_byte_train[idx*3072+k];
                // }
                // mean = sum0/3072;
                // for(int k=0;k<3072;k++){
                //     sum1 += std::pow(_image_train[idx*3072+k]-mean,2);
                // }
                // std = std::sqrt(sum1/3072);
                // for(int k=0;k<3072;k++){
                //     _image_train[idx*3072+k] = (_image_train[idx*3072+k]-mean) / std;
                // }
            }
            fclose(fp);
        }
        // read testSet
        fp = fopen((dataPath+"/test_batch.bin").c_str(),"r");
        for(int j=0;j<10000;j++){
            fread(&_label_test[j],sizeof(uint8_t),1,fp);
            fread(&_image_byte_test[j*3072],sizeof(uint8_t),3072,fp);
            for(int k=0;k<3072;k++){
                _image_test[j*3072+k] = (double)_image_byte_test[j*3072+k]/255;
            }
            // double sum0=0,sum1=0,mean,std;
            // for(int k=0;k<3072;k++){
            //     _image_test[j*3072+k] = (double)_image_byte_test[j*3072+k];
            //     sum0 += _image_byte_test[j*3072+k];
            // }
            // mean = sum0/3072;
            // for(int k=0;k<3072;k++){
            //     sum1 += std::pow(_image_test[j*3072+k]-mean,2);
            // }
            // std = std::sqrt(sum1/3072);
            // for(int k=0;k<3072;k++){
            //     _image_test[j*3072+k] = (_image_test[j*3072+k]-mean) / std;
            // }
        }
        fclose(fp);
    }

    double* image_train(int idx){
        assert(idx>=0&&idx<50000);
        return &_image_train[idx*3072];
    }

    double* image_test(int idx){
        assert(idx>=0&&idx<10000);
        return &_image_test[idx*3072];
    }

    ~DataSet(){
        free(_image_train);
        free(_image_test);
        free(_image_byte_train);
        free(_image_byte_test);
        free(_label_train);
        free(_label_test);
    }
};

#endif