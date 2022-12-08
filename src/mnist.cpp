#include<cstdio>
#include<cstdint>
#include<string>
#include<cstring>
#include<cmath>
#include<cfloat>
#include<iostream>

#include"NN.h"
#include"myRandom.h"


class DataSet{
    public:
    std::string imagePath,labelPath;
    int rows,cols,items_num;
    double *_image;
    uint8_t *_image_byte;
    uint8_t *label;

    double *image(int idx_item,int idx_row,int idx_col){
        assert(idx_item>=0&&idx_item<items_num);
        assert(idx_row>=0&&idx_row<rows);
        assert(idx_col>=0&&idx_col<cols);
        return &_image[idx_item*rows*cols+idx_row*cols+idx_col];
    }

    uint8_t *imageByte(int idx_item,int idx_row,int idx_col){
        assert(idx_item>=0&&idx_item<items_num);
        assert(idx_row>=0&&idx_row<rows);
        assert(idx_col>=0&&idx_col<cols);
        return &_image_byte[idx_item*rows*cols+idx_row*cols+idx_col];
    }

    void init (std::string imagePath,std::string labelPath){
        this->imagePath = imagePath;
        this->labelPath = labelPath;

        FILE *fp;
        int magic_num,label_items_num;
        // read _image
        fp = fopen(imagePath.c_str(),"r");
        assert(fp);

        fread(&magic_num,sizeof(int),1,fp);
        fread(&items_num,sizeof(int),1,fp);
        fread(&rows,sizeof(int),1,fp);
        fread(&cols,sizeof(int),1,fp);

        magic_num = flip<unsigned>(magic_num);
        items_num = flip<unsigned>(items_num);
        rows = flip<unsigned>(rows);
        cols = flip<unsigned>(cols);

        assert(magic_num==0x00000803);
        
        _image = (double*)malloc(items_num*rows*cols*sizeof(double));
        _image_byte = (uint8_t*)malloc(items_num*rows*cols*sizeof(uint8_t));

        for(int k=0;k<items_num;k++){
            double sum0=0,sum1=0,mean,std;
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    fread(imageByte(k,i,j),1,1,fp);
                    *image(k,i,j) = *imageByte(k,i,j);
                    sum0 += *imageByte(k,i,j);
                }
            }
            mean = sum0 / (rows*cols);
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    sum1 += std::pow(*image(k,i,j)-mean,2);
                }
            }
            std = std::sqrt(sum1/(rows*cols));
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    *image(k,i,j) = (*image(k,i,j)-mean) / std;
                }
            }
        }
        fclose(fp);


        // read label
        fp = fopen(labelPath.c_str(),"r");
        assert(fp);

        fread(&magic_num,sizeof(unsigned),1,fp);
        fread(&label_items_num,sizeof(unsigned),1,fp);  

        magic_num = flip<unsigned>(magic_num);
        label_items_num = flip<unsigned>(label_items_num);

        assert(magic_num==0x00000801);
        assert(label_items_num==items_num);

        label = (uint8_t*)malloc(items_num);
        for(int i=0;i<items_num;i++){
            fread(&label[i],1,1,fp);
        }

        fclose(fp);
    }

    void printIdx(int idx_item){
        assert(idx_item>=0 && idx_item < items_num);
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                printf("%3d ",*imageByte(idx_item,i,j));
            }
            printf("\n");
        }
        printf("Label: %d\n",label[idx_item]);
    }

    template<typename TT>
    TT flip(TT input){
        int size = sizeof(TT);
        TT ret = 0;
        for(int i=0;i<size;i++){
            ret += ((input>>(i*8)) & 0xFF)<<((size-i-1)*8);
        }
        return ret;
    }
};


class MyNN : public NN{
    public:

    MyNN(double mean,double std,double learn_rate){
        
        this->learn_rate = learn_rate;

        // init size
        INPUT.init(1,28,28);
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

    void train(DataSet &trainSet,DataSet testSet,
        Random &train_r,Random &test_r,int num){
        
        printf("Seed:%d\n",SEED);
        printf("Lr:%f\n",this->learn_rate);
        printf("ImageNum:%d\n",num);

        double loss = 0;
        // double corr_rate,corr_rate_l=0;
        int learn_interval = 1200;
        int interval = 240,idx;

        for(int i=0;i<num;i++){
            idx = train_r.next();
            INPUT.set(trainSet.image(idx,0,0));
            forward();
            if((i+1)%interval==0){
                loss /= interval;
                printf("[%5d/%-5d] Loss: %5.6f\n",i+1,num,loss);
                loss = 0;
            }
            
            if((i+1)%learn_interval==0){
                validate(testSet,test_r,1200);
                // corr_rate = validate(testSet,test_r,testSet.items_num);
                // if(corr_rate>0.99)goto Ret;
                // if(corr_rate<corr_rate_l){
                //     this->learn_rate *= 0.5;
                //     if(this->learn_rate<1e-6)goto Ret;
                //     printf("Learn_rate:%.12lf\n",this->learn_rate);
                // }
                // corr_rate_l = corr_rate;
            }
            loss += *OUTPUT.data(trainSet.label[idx]);
            backward(trainSet.label[idx]);
        }
    // Ret:
        // validate(testSet,test_r,testSet.items_num);
    }

    double validate(DataSet &test,Random &r,int num){
        int correct = 0,idx;
        for(int i=0;i<num;i++){
            idx = r.next();
            INPUT.set(test.image(idx,0,0));
            forward();
            if(OUTPUT.maxIdx()==test.label[idx])correct++;
        }
        double corr_rate = (double)correct / num;
        printf("Acc on TestSet(%d): %.2f%%\n",num,corr_rate*100);
        return corr_rate;
    }

    void show(DataSet &dataSet,Random &r){
        int idx;
        while(1){
            printf("PRESS ENTER ...\n");
            getchar();
            idx = r.next();
            dataSet.printIdx(idx);
            INPUT.set(dataSet.image(idx,0,0));
            forward();
            J.print();
            printf("Predict: %d\n",OUTPUT.maxIdx());
        }   
    }
};


int main(){
    setbuf(stdout, NULL);
    DataSet trainSet,testSet;
    printf("loading data...\n");
    trainSet.init("data/train-images-idx3-ubyte","data/train-labels-idx1-ubyte");
    testSet.init("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte");

    printf("init NN...\n");
    MyNN nn(0,0.1,5e-3);

    Random train_r(trainSet.items_num),test_r(testSet.items_num);

    #ifdef SHOW
    nn.train(trainSet,testSet,train_r,test_r,9600);
    #else
    nn.train(trainSet,testSet,train_r,test_r,trainSet.items_num);
    #endif

    #ifdef SHOW
    nn.show(trainSet,train_r);
    #endif
}