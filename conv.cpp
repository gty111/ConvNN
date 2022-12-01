#include<cstdio>
#include<cstdint>
#include<cassert>
#include<vector>
#include<string>
#include<cstring>
#include<cmath>
#include<random>
#include<algorithm>
#include<cfloat>

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()


template<typename T>
T flip(T input){
    size_t size = sizeof(T);
    T ret = 0;
    for(int i=0;i<size;i++){
        ret += ((input>>(i*8)) & 0xFF)<<((size-i-1)*8);
    }
    return ret;
}

class DataSet{
    public:
    std::string imagePath,labelPath;
    unsigned rows,cols,items_num;
    double *image;
    uint8_t *image_byte;
    uint8_t *label;

    double *imageIdx(int idx_item,int idx_row,int idx_col){
        assert(idx_item>=0&&idx_item<items_num);
        assert(idx_row>=0&&idx_row<rows);
        assert(idx_col>=0&&idx_col<cols);
        return &image[idx_item*rows*cols+idx_row*cols+idx_col];
    }

    uint8_t *imageByteIdx(int idx_item,int idx_row,int idx_col){
        assert(idx_item>=0&&idx_item<items_num);
        assert(idx_row>=0&&idx_row<rows);
        assert(idx_col>=0&&idx_col<cols);
        return &image_byte[idx_item*rows*cols+idx_row*cols+idx_col];
    }

    void init (std::string imagePath,std::string labelPath){
        this->imagePath = imagePath;
        this->labelPath = labelPath;

        FILE *fp;
        unsigned magic_num,label_items_num;
        // read image
        fp = fopen(imagePath.c_str(),"r");
        assert(fp);

        fread(&magic_num,sizeof(unsigned),1,fp);
        fread(&items_num,sizeof(unsigned),1,fp);
        fread(&rows,sizeof(unsigned),1,fp);
        fread(&cols,sizeof(unsigned),1,fp);

        magic_num = flip<unsigned>(magic_num);
        items_num = flip<unsigned>(items_num);
        rows = flip<unsigned>(rows);
        cols = flip<unsigned>(cols);

        assert(magic_num==0x00000803);
        
        image = (double*)malloc(items_num*rows*cols*sizeof(double));
        image_byte = (uint8_t*)malloc(items_num*rows*cols*sizeof(uint8_t));

        for(int k=0;k<items_num;k++){
            double sum0=0,sum1=0,mean,std;
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    fread(imageByteIdx(k,i,j),1,1,fp);
                    *imageIdx(k,i,j) = *imageByteIdx(k,i,j);
                    sum0 += *imageByteIdx(k,i,j);
                }
            }
            mean = sum0 / (rows*cols);
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    sum1 += std::pow(*imageIdx(k,i,j)-mean,2);
                }
            }
            std = std::sqrt(sum1/(rows*cols));
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    *imageIdx(k,i,j) = (*imageIdx(k,i,j)-mean) / std;
                }
            }
        }
        fclose(fp);


        // read lable
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
                printf("%3d ",*imageByteIdx(idx_item,i,j));
            }
            printf("\n");
        }
        printf("%d\n",label[idx_item]);
    }
};

template<typename T>
class Tensor{
    public:
    std::vector<unsigned>shape;
    T *data;

    void initData(){
        std::normal_distribution<double> dis(0,0.1);
        for(int i=0;i<getSize();i++){
            data[i] = dis(gen);
        }
    }

    void init(int a){
        shape.push_back(a);
        data = (T*)calloc(getSize(),sizeof(T));
    }

    void init(int a,int b){
        shape.push_back(a);
        shape.push_back(b);
        data = (T*)calloc(getSize(),sizeof(T));
    }

    void init(int a,int b,int c){
        shape.push_back(a);
        shape.push_back(b);
        shape.push_back(c);
        data = (T*)calloc(getSize(),sizeof(T));
    }

    unsigned getSize(){
        unsigned t = 1;
        for(auto e:shape){
            t *= e;
        }
        return t;
    }

    void set(T *from){
        for(int i=0;i<getSize();i++){
            data[i] = from[i];
        }
    }

    T* get(int idx0){
        assert(shape.size()==1);
        assert(idx0>=0&&idx0<shape[0]);
        return &data[idx0];
    }
    T* get(int idx0,int idx1){
        assert(shape.size()==2);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        return &data[idx0*shape[1]+idx1];
    }
    T* get(int idx0,int idx1,int idx2){
        assert(shape.size()==3);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        assert(idx2>=0&&idx2<shape[2]);
        return &data[idx0*shape[1]*shape[2]+idx1*shape[2]+idx2];
    }

    void std(){
        double sum0=0,sum1=0,mean,std;
        for(int i=0;i<getSize();i++){
            sum0 += data[i];
        }
        mean = sum0 / getSize();
        for(int i=0;i<getSize();i++){
            sum1 += (data[i]-mean) * (data[i]-mean);
        }
        std = std::sqrt(sum1 / getSize());
        for(int i=0;i<getSize();i++){
            data[i] = (data[i]-mean) / std;
        }
    }

    void print(){
        printf("---------------\n");
        if(shape.size()==3){
            // for(int k=0;k<shape[0];k++)
            for(int i=0;i<shape[1];i++){
                for(int j=0;j<shape[2];j++){
                    printf("% 5.6f ",*(get(0,i,j)));
                }
                printf("\n");
            }
        }else if(shape.size()==2){
            for(int i=0;i<shape[0];i++){
                for(int j=0;j<shape[1];j++){
                    printf("% 5.6f ",*(get(i,j)));
                }
                printf("\n");
            }
        }else if(shape.size()==1){
            for(int j=0;j<shape[0];j++){
                printf("% 5.6f\n",*(get(j)));
            }
        }
        
    }

    ~Tensor(){
        // if(data)free(data);
    }
};

template<typename T>
class NN{
    public:
    double learn_rate;

    // forward var
    Tensor<T> conv1; // 32x3x3
    Tensor<T> conv2; // 2x3x3
    Tensor<T> convb1; // 32
    Tensor<T> convb2; // 2
    Tensor<T> fc1w; // 9216x128
    Tensor<T> fc1b; // 128
    Tensor<T> fc2w; // 128x10
    Tensor<T> fc2b; // 10

    // intermediate result
    Tensor<T> A; // 32x26x26
    Tensor<T> B; // 32x26x26
    Tensor<T> C; // 64x24x24
    Tensor<T> D; // 64x24x24
    Tensor<T> E; // 64x12x12
    Tensor<T> F; // 9216
    Tensor<T> G; // 128
    Tensor<T> H; // 128
    Tensor<T> I; // 10
    Tensor<T> J; // 10
    Tensor<T> K; // 10

    // backward var
    Tensor<T> deti; // 10
    Tensor<T> detj; // 10
    Tensor<T> detg; // 128
    Tensor<T> deth; // 128
    Tensor<T> detf; // 9216
    Tensor<T> dete; // 64x12x12
    Tensor<T> detc; // 64x24x24
    Tensor<T> detd; // 64x24x24
    Tensor<T> deta; // 32x26x26
    Tensor<T> detb; // 32x26x26

    // to cal detb
    Tensor<T> Cpad0,Cpad1,conv_rot0,conv_rot1,res0,res1,bias;

    void init(){
        // nn
        conv1.init(32,3,3);
        conv2.init(2,3,3);
        convb1.init(32);
        convb2.init(2);
        fc1w.init(9216,128);
        fc1b.init(128);
        fc2w.init(128,10);
        fc2b.init(10);

        conv1.initData();
        conv2.initData();
        convb1.initData();
        convb2.initData();

        fc1w.initData();
        fc1b.initData();
        fc2w.initData();
        fc2b.initData();

        // intermediate
        A.init(32,26,26);
        B.init(32,26,26);
        C.init(64,24,24);
        D.init(64,24,24);
        E.init(64,12,12);
        F.init(9216);
        G.init(128);
        H.init(128);
        I.init(10);
        J.init(10);
        K.init(10);

        // backward
        detj.init(10);
        deti.init(10);
        deth.init(128);
        detg.init(128);
        detf.init(9216);
        dete.init(64,12,12);
        detd.init(64,24,24);
        detc.init(64,24,24);
        detb.init(32,26,26);
        deta.init(32,26,26);

        // cal detb
        Cpad0.init(32,28,28);
        Cpad1.init(32,28,28);
        conv_rot0.init(1,3,3);
        conv_rot1.init(1,3,3);
        res0.init(32,26,26);
        res1.init(32,26,26);
        bias.init(1);
    }

    void backward(Tensor<T> &input,int label){
        assert(label>=0&&label<=9);

        // ----detj----
        T maximum = -1e100;

        for(int i=0;i<10;i++){
            maximum = std::max(maximum,*J.get(i));
        }

        T sum = 0;
        for(int i=0;i<10;i++){
            sum += std::exp(*J.get(i)-maximum);
        }
        
        for(int i=0;i<10;i++){
            if(label==i){
                *detj.get(i) = 1 - std::exp(*J.get(i)-maximum) / sum;
            }else{
                *detj.get(i) = - std::exp(*J.get(i)-maximum) / sum;
            }
        }

        // ----deti----
        for(int i=0;i<10;i++){
            *deti.get(i) = *I.get(i)>0 ? *detj.get(i) : 0;
        }

        // ----deth----
        for(int i=0;i<128;i++){
            T sum = 0;
            for(int j=0;j<10;j++){
                sum += *fc2w.get(i,j) * (*deti.get(j));
            }
            *deth.get(i) = sum;
        }

        // ----detg----
        for(int i=0;i<128;i++){
            *detg.get(i) = *G.get(i)>0 ? *deth.get(i) : 0;
        }

        // ----fc2w----
        for(int i=0;i<10;i++){
            for(int j=0;j<128;j++){
                *fc2w.get(j,i) += (*H.get(j)) * (*deti.get(i)) * learn_rate;
            }
        }

        // ----fc2b----
        for(int i=0;i<10;i++){
            *fc2b.get(i) += (*deti.get(i)) * learn_rate;
        }

        // ----detf----
        for(int i=0;i<9216;i++){
            T sum = 0;
            for(int j=0;j<128;j++){
                sum += *fc1w.get(i,j) * (*detg.get(j));
            }
            *detf.get(i) = sum;
        }
        
        // ----fc1w----
        for(int i=0;i<128;i++){
            for(int j=0;j<9216;j++){
                *fc1w.get(j,i) += (*detg.get(i)) * (*F.get(j)) * learn_rate;
            }
        }

        // ----fc1b----
        for(int i=0;i<128;i++){
            *fc1b.get(i) += (*detg.get(i)) * learn_rate;
        }

        // ----dete----
        for(int i=0;i<64;i++){
            for(int j=0;j<12;j++){
                for(int k=0;k<12;k++){
                    *dete.get(i,j,k) = *detf.get(i*12*12+j*12+k);
                }
            }
        }

        // ----detd----
        for(int i=0;i<64;i++){
            for(int j=0;j<24;j++){
                for(int k=0;k<24;k++){
                    *detd.get(i,j,k) = *dete.get(i,j/2,k/2) / 4;
                }
            }
        }

        // ----detc----
        for(int i=0;i<64;i++){
            for(int j=0;j<24;j++){
                for(int k=0;k<24;k++){
                    *detc.get(i,j,k) = *C.get(i,j,k)>0 ? *detd.get(i,j,k) : 0;
                }
            }
        }

        // ----detb----
        for(int i=0;i<32;i++){
            for(int j=0;j<24;j++){
                for(int k=0;k<24;k++){
                    *Cpad0.get(i,j+2,k+2) = *detc.get(i,j,k);
                    *Cpad1.get(i,j+2,k+2) = *detc.get(i+32,j,k);
                }
            }
        }

        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                *conv_rot0.get(0,i,j) = *conv2.get(0,2-i,2-j);
                *conv_rot1.get(0,i,j) = *conv2.get(1,2-i,2-j);
            }
        }

        conv(res0,Cpad0,conv_rot0,bias);
        conv(res1,Cpad1,conv_rot1,bias);

        for(int i=0;i<32;i++){
            for(int j=0;j<26;j++){
                for(int k=0;k<26;k++){
                    *detb.get(i,j,k) = (*res0.get(i,j,k)) + (*res1.get(i,j,k));
                }
            }
        }

        // ----conv2----
        for(int i=0;i<2;i++){
            for(int j=0;j<3;j++){
                for(int k=0;k<3;k++){
                    T sum = 0;
                    for(int p=0;p<32;p++){
                        for(int m=0;m<24;m++){
                            for(int n=0;n<24;n++){
                                sum += *detc.get(i*32+p,m,n) * (*B.get(p,m+j,n+k));
                            }
                        }
                    }
                    *conv2.get(i,j,k) += sum * learn_rate;
                }
            }
        }


        // ----convb2----
        for(int i=0;i<64;i++){
            for(int j=0;j<24;j++){
                for(int k=0;k<24;k++){
                    *convb2.get(i/32) += *detc.get(i,j,k) * learn_rate;
                }
            }
        }

        // ----deta----
        for(int i=0;i<32;i++){
            for(int j=0;j<26;j++){
                for(int k=0;k<26;k++){
                    *deta.get(i,j,k) = *A.get(i,j,k)>0 ? *detb.get(i,j,k) : 0;
                }
            }
        }

        // ----conv1----
        for(int i=0;i<32;i++){
            for(int j=0;j<3;j++){
                for(int k=0;k<3;k++){
                    for(int m=0;m<26;m++){
                        for(int n=0;n<26;n++){
                            *conv1.get(i,j,k) += *deta.get(i,m,n) * (*input.get(0,m+j,n+k)) * learn_rate;
                        }
                    }
                }
            }
        }

        // ----convb1----
        for(int i=0;i<32;i++){
            for(int j=0;j<26;j++){
                for(int k=0;k<26;k++){
                    *convb1.get(i) += *deta.get(i,j,k) * learn_rate;
                }
            }
        }
        

    }

    void conv(Tensor<T> &output,Tensor<T> &input,Tensor<T> &kernel,Tensor<T> &bias){
        assert(input.shape.size()==3);
        for(int i=0;i<kernel.shape[0];i++){
            for(int j=0;j<input.shape[0];j++){
                for(int p=0;p+kernel.shape[1]-1<input.shape[1];p++){

                    for(int q=0;q+kernel.shape[2]-1<input.shape[2];q++){
                        T sum = 0;
                        for(int m=0;m<kernel.shape[1];m++){
                            for(int n=0;n<kernel.shape[2];n++){
                                sum += (*input.get(j,p+m,q+n)) * (*kernel.get(i,m,n)); 
                            }
                        }
                        (*output.get(i*input.shape[0]+j,p,q)) = sum + *bias.get(i);
                    }
                }
            }
        }
    }

    void relu(Tensor<T> &output,Tensor<T> &input){
        if(output.shape.size()==1){
            for(int i=0;i<output.shape[0];i++){
                (*output.get(i)) = (*input.get(i)) > 0 ? (*input.get(i)) : 0 ;
            }
        }else if(output.shape.size()==3){
            for(int i=0;i<output.shape[0];i++){
                for(int j=0;j<output.shape[1];j++){
                    for(int k=0;k<output.shape[2];k++){
                        (*output.get(i,j,k)) = (*input.get(i,j,k)) > 0 ? (*input.get(i,j,k)) : 0;
                    }
                }
            }
        }else assert(0);
    }

    void meanPool(Tensor<T> &output,Tensor<T> &input){
        // 2x2 
        assert(output.shape.size()==3);
        for(int k=0;k<output.shape[0];k++){
            for(int i=0;i<output.shape[1];i++){
                for(int j=0;j<output.shape[2];j++){
                    (*output.get(k,i,j)) = 
                        ( *input.get(k,i*2,j*2) + *input.get(k,i*2,j*2+1) + 
                        *input.get(k,i*2+1,j*2) + *input.get(k,i*2+1,j*2+1))/4;
                }
            }
        }
    }

    void flattern(Tensor<T> &output,Tensor<T> &input){
        assert(input.shape.size()==3);
        for(int i=0;i<input.shape[0];i++){
            for(int j=0;j<input.shape[1];j++){
                for(int k=0;k<input.shape[2];k++){
                    *output.get(i*input.shape[1]*input.shape[2]+j*input.shape[2]+k) = 
                        *input.get(i,j,k); 
                }
            }
        }
    }

    void fc(Tensor<T> &output,Tensor<T> &input,Tensor<T> &weight,Tensor<T> &bias){
        assert(input.shape.size()==1);
        assert(weight.shape.size()==2);
        for(int i=0;i<weight.shape[1];i++){
            T sum = 0;
            for(int j=0;j<weight.shape[0];j++){
                sum += (*input.get(j)) * (*weight.get(j,i)) ;
            }
            (*output.get(i)) = sum + (*bias.get(i));
        }
    }

    void logSoftmax(Tensor<T> &output,Tensor<T> &input){
        assert(input.shape.size()==1);
        T maximum = 0;
        for(int i=0;i<input.shape[0];i++){
            maximum = std::max(*input.get(i),maximum);
        }
        T sum = 0;
        for(int i=0;i<input.shape[0];i++){
            sum += std::exp(*input.get(i)-maximum);
        }
        for(int i=0;i<input.shape[0];i++){
            (*output.get(i)) = *input.get(i) - maximum - std::log(sum);
        }
    }

    void forward(Tensor<T> input,int label){
        assert(label>=0&&label<=9);
        // input 1x28x28
        assert(input.shape[0]==1&&input.shape[1]==28&&input.shape[2]==28);
        assert(input.shape.size()==3);
        // a 32x26x26
        conv(A,input,conv1,convb1);
        // b 32x26x26
        relu(B,A);
        // c 64x24x24
        conv(C,B,conv2,convb2);
        // d 64x24x24
        relu(D,C);
        // e 64x12x12
        meanPool(E,D);
        // f 9216
        flattern(F,E);
        // g 128
        fc(G,F,fc1w,fc1b);
        // h 128
        relu(H,G);
        // i 10
        fc(I,H,fc2w,fc2b);
        // j 10
        relu(J,I);
        // k 10
        logSoftmax(K,J);
    }
};



int main(){
    DataSet train,test;
    train.init("../data/MNIST/raw/train-images-idx3-ubyte","../data/MNIST/raw/train-labels-idx1-ubyte");
    test.init("../data/MNIST/raw/t10k-images-idx3-ubyte","../data/MNIST/raw/t10k-labels-idx1-ubyte");

    Tensor<double> input;
    input.init(1,28,28);

    NN<double> nn;

    nn.init();

    std::vector<int>train_idx,test_idx;
    for(int i=0;i<train.items_num;i++){
        train_idx.push_back(i);
    }
    std::shuffle(train_idx.begin(),train_idx.end(),gen);
    for(int i=0;i<test.items_num;i++){
        test_idx.push_back(i);
    }
    std::shuffle(test_idx.begin(),test_idx.end(),gen);

    nn.learn_rate = 5e-3;

    double loss = 0,corr_rate,corr_rate_l=0;
    int inteval = 120,learn_inteval = 1200,test_num=1200,correct;
    for(int idx=0;idx<60000;idx++){
        input.set(train.imageIdx(train_idx[idx],0,0));
        nn.forward(input,train.label[train_idx[idx]]);
        if((idx+1)%inteval==0){
            loss /= inteval;
            printf("%d loss:%5.6f\n",idx+1,loss);
            loss = 0;
        }
        
        if((idx+1)%learn_inteval==0){
            correct = 0;
            for(int idx=0;idx<test_num;idx++){
                input.set(test.imageIdx(test_idx[idx],0,0));
                nn.forward(input,test.label[test_idx[idx]]);
                int idx_min;
                double min=-1e100;
                for(int i=0;i<10;i++){
                    if(*nn.K.get(i)>min){
                        min = *nn.K.get(i);
                        idx_min = i;
                    }
                }
                if(idx_min==test.label[test_idx[idx]])correct++;
            }
            corr_rate = (double)correct / test_num;
            printf("Test: %.2f%%\n",corr_rate*100);
            if(corr_rate<corr_rate_l){
                nn.learn_rate *= 0.5;
                if(nn.learn_rate<1e-3)goto Next;
                printf("Learn_rate:%.12lf\n",nn.learn_rate);
            }
            corr_rate_l = corr_rate;
        }
        loss += *nn.K.get(train.label[train_idx[idx]]);
        nn.backward(input,train.label[train_idx[idx]]);
    }

Next:
    correct = 0;
    test_num = test.items_num;
    for(int idx=0;idx<test_num;idx++){
        input.set(test.imageIdx(test_idx[idx],0,0));
        nn.forward(input,test.label[test_idx[idx]]);
        int idx_min;
        double min=-1e100;
        for(int i=0;i<10;i++){
            if(*nn.K.get(i)>min){
                min = *nn.K.get(i);
                idx_min = i;
            }
        }
        if(idx_min==test.label[test_idx[idx]])correct++;
    }
    corr_rate = (double)correct / test_num;
    printf("Test: %.2f%%\n",corr_rate*100);
    
    while(1){
        srand(time(0));
        int idx = rand()%60000;
        train.printIdx(train_idx[idx]);
        input.set(train.imageIdx(train_idx[idx],0,0));
        nn.forward(input,train.label[train_idx[idx]]);
        nn.J.print();
        nn.K.print();
        // nn.fc2w.print();
        int idx_min;
        double min=-1e100;
        for(int i=0;i<10;i++){
            if(*nn.K.get(i)>min){
                min = *nn.K.get(i);
                idx_min = i;
            }
        }
        printf("pred:%d\n",idx_min);
        // nn.backward(input,train.label[train_idx[idx]]);
        getchar();
    }   
    
}
