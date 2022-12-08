#include<string>
#include<cassert>
#include<algorithm>

#include"NN.h"
#include"myRandom.h"


class DataSet{
    public:
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
                double sum0=0,sum1=0,mean,std;
                for(int k=0;k<3072;k++){
                    _image_train[idx*3072+k] = (double)_image_byte_train[idx*3072+k];
                    sum0 += _image_byte_train[idx*3072+k];
                }
                mean = sum0/3072;
                for(int k=0;k<3072;k++){
                    sum1 += std::pow(_image_train[idx*3072+k]-mean,2);
                }
                std = std::sqrt(sum1/3072);
                for(int k=0;k<3072;k++){
                    _image_train[idx*3072+k] = (_image_train[idx*3072+k]-mean) / std;
                }
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
};


class MyNN: public NN{
    public:

    MyNN(double mean,double std,double learn_rate){
        
        this->learn_rate = learn_rate;

        // init size
        INPUT.init(3,32,32);
        conv1.init(16,INPUT.shape[0],3,3);
        convb1.init(conv1.shape[0]);
        A.init(conv1.shape[0],INPUT.shape[1]-conv1.shape[2]+1,INPUT.shape[2]-conv1.shape[3]+1);
        B.init(A.shape[0],A.shape[1],A.shape[2]);
        conv2.init(16,B.shape[0],3,3);
        convb2.init(conv2.shape[0]);
        C.init(conv2.shape[0],B.shape[1]-conv2.shape[2]+1,B.shape[2]-conv2.shape[3]+1);
        D.init(C.shape[0],C.shape[1],C.shape[2]);
        E.init(D.shape[0],D.shape[1]/2,D.shape[2]/2);
        F.init(E.size);
        fc1w.init(F.size,128);
        fc1b.init(fc1w.shape[1]);
        G.init(fc1w.shape[1]);
        H.init(G.shape[0]);
        fc2w.init(H.shape[0],10);
        fc2b.init(fc2w.shape[1]);
        I.init(fc2w.shape[1]);
        J.init(I.shape[0]);
        OUTPUT.init(J.shape[0]);


        // init data
        conv1.initData(mean,std);
        conv2.initData(mean,std);
        convb1.initData(mean,std);
        convb2.initData(mean,std);

        fc1w.initData(mean,std);
        fc1b.initData(mean,std);
        fc2w.initData(mean,std);
        fc2b.initData(mean,std);

        // to cal B grad
        CgradPad.init(C.shape[0],B.shape[1]+conv2.shape[2]-1,B.shape[2]+conv2.shape[3]-1);
    }

    void train(DataSet &data,Random &train_r,Random &test_r,int num){
        printf("Seed:%d\n",SEED);
        printf("Lr:%f\n",this->learn_rate);
        printf("ImageNum:%d\n",num);

        double loss = 0;
        int learn_interval = 1000;
        int interval = 100,idx;

        for(int i=0;i<num;i++){
            idx = train_r.next();
            INPUT.set(data.image_train(idx));
            forward();
            if((i+1)%interval==0){
                loss /= interval;
                printf("[%5d/%-5d] Loss: %5.6f\n",i+1,num,loss);
                loss = 0;
            }
            // if((i+1)%50000==0){
            //     learn_rate /= 10;
            // }
            if((i+1)%learn_interval==0){
                validate(data,test_r,1200);
            }
            loss += *OUTPUT.data(data._label_train[idx]);
            backward(data._label_train[idx]);
        }

    }

    double validate(DataSet &data,Random &r,int num){
        int correct = 0,idx;
        for(int i=0;i<num;i++){
            idx = r.next();
            INPUT.set(data.image_test(idx));
            forward();
            if(OUTPUT.maxIdx()==data._label_test[idx])correct++;
        }
        double corr_rate = (double)correct / num;
        printf("Acc on TestSet(%d): %.2f%%\n",num,corr_rate*100);
        return corr_rate;
    }

};


int main(){
    setbuf(stdout, NULL);
    printf("loading data...\n");
    DataSet cifar("data/cifar-10-batches-bin");

    printf("init NN...\n");
    MyNN nn(0,0.1,1e-4);

    Random train_r(50000),test_r(10000);

    nn.train(cifar,train_r,test_r,5e4);
    
}