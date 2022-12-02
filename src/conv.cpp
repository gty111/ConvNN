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
#include<iostream>

#define SEED 0

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(SEED); //Standard mersenne_twister_engine seeded with rd()

class Random{
    public:
    int *_arr;
    int _idx,_size;
    Random(int size){
        _arr = (int *)malloc(size*sizeof(int));
        for(int i=0;i<size;i++){
            _arr[i] = i;
        }
        std::shuffle(_arr , _arr+size , gen);
        _idx = -1;
        _size = size;
    }

    int next(){
        _idx = (_idx+1) % _size;
        return _arr[_idx];
    }

    ~Random(){
        free(_arr);
    }
};

template<typename T>
class DataSet{
    public:
    std::string imagePath,labelPath;
    int rows,cols,items_num;
    T *_image;
    uint8_t *_image_byte;
    uint8_t *label;

    T *image(int idx_item,int idx_row,int idx_col){
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
        
        _image = (T*)malloc(items_num*rows*cols*sizeof(T));
        _image_byte = (uint8_t*)malloc(items_num*rows*cols*sizeof(uint8_t));

        for(int k=0;k<items_num;k++){
            T sum0=0,sum1=0,mean,std;
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
        printf("%d\n",label[idx_item]);
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

template<typename T>
class Tensor{
    public:
    int size = 0;
    std::vector<int>shape;
    T *_data=nullptr;
    T *_grad=nullptr;

    void initData(double mean,double std){
        std::normal_distribution<double> dis(mean,std);
        for(int i=0;i<size;i++){
            _data[i] = dis(gen);
        }
    }

    void init(int a){
        shape.push_back(a);
        size = a;
        _data = (T*)calloc(size,sizeof(T));
        _grad = (T*)calloc(size,sizeof(T));
    }

    void init(int a,int b){
        shape.push_back(a);
        shape.push_back(b);
        size = a * b;
        _data = (T*)calloc(size,sizeof(T));
        _grad = (T*)calloc(size,sizeof(T));
    }

    void init(int a,int b,int c){
        shape.push_back(a);
        shape.push_back(b);
        shape.push_back(c);
        size = a * b * c;
        _data = (T*)calloc(size,sizeof(T));
        _grad = (T*)calloc(size,sizeof(T));
    }

    void init(int a,int b,int c,int d){
        shape.push_back(a);
        shape.push_back(b);
        shape.push_back(c);
        shape.push_back(d);
        size = a * b * c * d;
        _data = (T*)calloc(size,sizeof(T));
        _grad = (T*)calloc(size,sizeof(T));
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
        assert(shape.size()==1);
        assert(idx0>=0&&idx0<shape[0]);
        return &_data[idx0];
    }
    T* data(int idx0,int idx1){
        assert(shape.size()==2);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        return &_data[idx0*shape[1]+idx1];
    }
    T* data(int idx0,int idx1,int idx2){
        assert(shape.size()==3);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        assert(idx2>=0&&idx2<shape[2]);
        return &_data[idx0*shape[1]*shape[2]+idx1*shape[2]+idx2];
    }

    T* data(int idx0,int idx1,int idx2,int idx3){
        assert(shape.size()==4);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        assert(idx2>=0&&idx2<shape[2]);
        assert(idx3>=0&&idx3<=shape[3]);
        return &_data[idx0*shape[1]*shape[2]*shape[3]+idx1*shape[2]*shape[3]+idx2*shape[3]+idx3];
    }

    T* grad(int idx0){
        assert(shape.size()==1);
        assert(idx0>=0&&idx0<shape[0]);
        return &_grad[idx0];
    }
    T* grad(int idx0,int idx1){
        assert(shape.size()==2);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        return &_grad[idx0*shape[1]+idx1];
    }
    T* grad(int idx0,int idx1,int idx2){
        assert(shape.size()==3);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        assert(idx2>=0&&idx2<shape[2]);
        return &_grad[idx0*shape[1]*shape[2]+idx1*shape[2]+idx2];
    }

    T* grad(int idx0,int idx1,int idx2,int idx3){
        assert(shape.size()==4);
        assert(idx0>=0&&idx0<shape[0]);
        assert(idx1>=0&&idx1<shape[1]);
        assert(idx2>=0&&idx2<shape[2]);
        assert(idx3>=0&&idx3<=shape[3]);
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

    void print(){
        printf("---------------\n");
        if(shape.size()==3){
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

    ~Tensor(){
        if(_data)free(_data);
        if(_grad)free(_grad);
    }
};

template<typename T>
class NN{
    public:
    double learn_rate;

    // forward var
    Tensor<T> conv1; // 16x1x3x3
    Tensor<T> convb1; // 16
    Tensor<T> conv2; // 16x16x3x3
    Tensor<T> convb2; // 16
    Tensor<T> fc1w; // 2304x128
    Tensor<T> fc1b; // 128
    Tensor<T> fc2w; // 128x10
    Tensor<T> fc2b; // 10

    // intermediate var
    Tensor<T> INPUT; // 1x28x28
    Tensor<T> A; // 16x26x26
    Tensor<T> B; // 16x26x26
    Tensor<T> C; // 16x24x24
    Tensor<T> D; // 16x24x24
    Tensor<T> E; // 16x12x12
    Tensor<T> F; // 2304
    Tensor<T> G; // 128
    Tensor<T> H; // 128
    Tensor<T> I; // 10
    Tensor<T> J; // 10
    Tensor<T> K; // 10

    // cal detb
    Tensor<T> detcPad; // 64x28x28

    NN(double mean,double std,double learn_rate){
        
        this->learn_rate = learn_rate;

        // nn
        conv1.init(16,1,3,3);
        conv2.init(16,16,3,3);
        convb1.init(16);
        convb2.init(16);
        fc1w.init(2304,128);
        fc1b.init(128);
        fc2w.init(128,10);
        fc2b.init(10);

        conv1.initData(mean,std);
        conv2.initData(mean,std);
        convb1.initData(mean,std);
        convb2.initData(mean,std);

        fc1w.initData(mean,std);
        fc1b.initData(mean,std);
        fc2w.initData(mean,std);
        fc2b.initData(mean,std);

        // intermediate
        INPUT.init(1,28,28);
        A.init(16,26,26);
        B.init(16,26,26);
        C.init(16,24,24);
        D.init(16,24,24);
        E.init(16,12,12);
        F.init(2304);
        G.init(128);
        H.init(128);
        I.init(10);
        J.init(10);
        K.init(10);

        // cal detb
        detcPad.init(16,28,28);
    }

    void backward(int label){
        assert(label>=0&&label<=9);

        // ----detj----
        T maximum = J.max();

        T sum = 0;
        for(int i=0;i<J.shape[0];i++){
            sum += std::exp(*J.data(i)-maximum);
        }
        
        for(int i=0;i<J.shape[0];i++){
            *J.grad(i) = i==label ? 
                1 - std::exp(*J.data(i)-maximum) / sum:
                - std::exp(*J.data(i)-maximum) / sum  ;
        }

        // ----deti----
        for(int i=0;i<I.shape[0];i++){
            *I.grad(i) = *I.data(i)>0 ? *J.grad(i) : 0;
        }

        // ----deth----
        for(int i=0;i<fc2w.shape[0];i++){
            T sum = 0;
            for(int j=0;j<fc2w.shape[1];j++){
                sum += *fc2w.data(i,j) * (*I.grad(j));
            }
            *H.grad(i) = sum;
        }

        // ----detg----
        for(int i=0;i<G.shape[0];i++){
            *G.grad(i) = *G.data(i)>0 ? *H.grad(i) : 0;
        }

        // ----fc2w----
        for(int i=0;i<fc2w.shape[0];i++){
            for(int j=0;j<fc2w.shape[1];j++){
                *fc2w.data(i,j) += (*H.data(i)) * (*I.grad(j)) * learn_rate;
            }
        }

        // ----fc2b----
        for(int i=0;i<fc2b.shape[0];i++){
            *fc2b.data(i) += (*I.grad(i)) * learn_rate;
        }

        // ----detf----
        for(int i=0;i<F.shape[0];i++){
            T sum = 0;
            for(int j=0;j<G.shape[0];j++){
                sum += *fc1w.data(i,j) * (*G.grad(j));
            }
            *F.grad(i) = sum;
        }
        
        // ----fc1w----
        for(int i=0;i<fc1w.shape[0];i++){
            for(int j=0;j<fc1w.shape[1];j++){
                *fc1w.data(i,j) += (*G.grad(j)) * (*F.data(i)) * learn_rate;
            }
        }

        // ----fc1b----
        for(int i=0;i<fc1b.shape[0];i++){
            *fc1b.data(i) += (*G.grad(i)) * learn_rate;
        }

        // ----dete----
        for(int i=0;i<E.shape[0];i++){
            for(int j=0;j<E.shape[1];j++){
                for(int k=0;k<E.shape[2];k++){
                    *E.grad(i,j,k) = *F.grad(i*12*12+j*12+k);
                }
            }
        }

        // ----detd----
        for(int i=0;i<D.shape[0];i++){
            for(int j=0;j<D.shape[1];j++){
                for(int k=0;k<D.shape[2];k++){
                    *D.grad(i,j,k) = *E.grad(i,j/2,k/2) / 4;
                }
            }
        }

        // ----detc----
        for(int i=0;i<C.shape[0];i++){
            for(int j=0;j<C.shape[1];j++){
                for(int k=0;k<C.shape[2];k++){
                    *C.grad(i,j,k) = *C.data(i,j,k)>0 ? *D.grad(i,j,k) : 0;
                }
            }
        }

        // ----detb----
        for(int i=0;i<C.shape[0];i++){
            for(int j=0;j<C.shape[1];j++){
                for(int k=0;k<C.shape[2];k++){
                    *detcPad.data(i,j+2,k+2) = *C.grad(i,j,k);
                }
            }
        }
        for(int i=0;i<B.shape[0];i++){
            for(int j=0;j<B.shape[1];j++){
                for(int k=0;k<B.shape[2];k++){
                    T sum = 0;
                    for(int m=0;m<conv2.shape[0];m++){
                        for(int p=0;p<conv2.shape[2];p++){
                            for(int q=0;q<conv2.shape[3];q++){
                                sum += *detcPad.data(m,j+p,k+q) * (*conv2.data(m,i,2-p,2-q));
                            }
                        }
                    }
                    *B.grad(i,j,k) = sum;
                }
            }
        }

        

        // ----conv2----
        for(int i=0;i<conv2.shape[0];i++){
            for(int j=0;j<conv2.shape[1];j++){
                for(int p=0;p<conv2.shape[2];p++){
                    for(int q=0;q<conv2.shape[3];q++){
                        T sum = 0;
                        for(int m=0;m<C.shape[1];m++){
                            for(int n=0;n<C.shape[2];n++){
                                sum += *B.data(j,m+p,n+q) * (*C.grad(i,m,n));
                            }
                        }
                        *conv2.data(i,j,p,q) += sum * learn_rate;
                    }
                }
            }
        }


        // ----convb2----
        for(int i=0;i<C.shape[0];i++){
            for(int j=0;j<C.shape[1];j++){
                for(int k=0;k<C.shape[2];k++){
                    *convb2.data(i) += *C.grad(i,j,k) * learn_rate;    
                }
            }
        }

        // ----deta----
        for(int i=0;i<A.shape[0];i++){
            for(int j=0;j<A.shape[1];j++){
                for(int k=0;k<A.shape[2];k++){
                    *A.grad(i,j,k) = *A.data(i,j,k)>0 ? *B.grad(i,j,k) : 0;
                }
            }
        }

        // ----conv1----
        for(int i=0;i<conv1.shape[0];i++){
            for(int p=0;p<conv1.shape[1];p++){
                for(int j=0;j<conv1.shape[2];j++){
                    for(int k=0;k<conv1.shape[3];k++){
                        T sum = 0;
                        for(int m=0;m<A.shape[1];m++){
                            for(int n=0;n<A.shape[2];n++){
                                sum += *A.grad(i,m,n) * (*INPUT.data(p,m+j,n+k));
                            }
                        }
                        *conv1.data(i,p,j,k) += sum * learn_rate;
                    }
                }
            }
        }

        // ----convb1----
        for(int i=0;i<A.shape[0];i++){
            T sum = 0;
            for(int j=0;j<A.shape[1];j++){
                for(int k=0;k<A.shape[2];k++){
                    sum += *A.grad(i,j,k);
                }
            }
            *convb1.data(i) += sum * learn_rate;
        }
        

    }

    void conv(Tensor<T> &output,Tensor<T> &input,Tensor<T> &kernel,Tensor<T> &bias){
        assert(kernel.shape.size()==4);
        assert(input.shape.size()==3);
        assert(kernel.shape[1]==input.shape[0]);
        assert(kernel.shape[0]==bias.shape[0]);
        assert(bias.shape.size()==1);
        assert(output.shape[0]==kernel.shape[0]);

        for(int i=0;i<kernel.shape[0];i++){
            for(int m=0;m<output.shape[1];m++){
                for(int n=0;n<output.shape[2];n++){
                    T sum = 0;
                    for(int j=0;j<kernel.shape[1];j++){
                        for(int p=0;p<kernel.shape[2];p++){
                            for(int q=0;q<kernel.shape[3];q++){
                                sum += *input.data(j,m+p,n+q) * (*kernel.data(i,j,p,q));
                            }    
                        }
                    }
                    *output.data(i,m,n) = sum + (*bias.data(i));
                }
            }
        }
        
    }

    void relu(Tensor<T> &output,Tensor<T> &input){
        if(output.shape.size()==1){
            for(int i=0;i<output.shape[0];i++){
                (*output.data(i)) = (*input.data(i)) > 0 ? (*input.data(i)) : 0 ;
            }
        }else if(output.shape.size()==3){
            for(int i=0;i<output.shape[0];i++){
                for(int j=0;j<output.shape[1];j++){
                    for(int k=0;k<output.shape[2];k++){
                        (*output.data(i,j,k)) = (*input.data(i,j,k)) > 0 ? (*input.data(i,j,k)) : 0;
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
                    (*output.data(k,i,j)) = 
                        ( *input.data(k,i*2,j*2) + *input.data(k,i*2,j*2+1) + 
                        *input.data(k,i*2+1,j*2) + *input.data(k,i*2+1,j*2+1))/4;
                }
            }
        }
    }

    void flattern(Tensor<T> &output,Tensor<T> &input){
        assert(input.shape.size()==3);
        for(int i=0;i<input.shape[0];i++){
            for(int j=0;j<input.shape[1];j++){
                for(int k=0;k<input.shape[2];k++){
                    *output.data(i*input.shape[1]*input.shape[2]+j*input.shape[2]+k) = 
                        *input.data(i,j,k); 
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
                sum += (*input.data(j)) * (*weight.data(j,i)) ;
            }
            (*output.data(i)) = sum + (*bias.data(i));
        }
    }

    void logSoftmax(Tensor<T> &output,Tensor<T> &input){
        assert(input.shape.size()==1);
        T maximum = input.max();
        T sum = 0;
        for(int i=0;i<input.shape[0];i++){
            sum += std::exp(*input.data(i)-maximum);
        }
        for(int i=0;i<input.shape[0];i++){
            (*output.data(i)) = *input.data(i) - maximum - std::log(sum);
        }
    }

    void forward(){
        // input 1x28x28
        conv(A,INPUT,conv1,convb1); // conv1 16x1x3x3 convb1 16
        // A 16x26x26
        relu(B,A); 
        // B 16x26x26
        conv(C,B,conv2,convb2); // conv2 16x16x3x3 convb2 16
        // C 16x24x24
        relu(D,C);
        // D 16x24x24
        meanPool(E,D); // meanPool 2x2
        // E 16x12x12
        flattern(F,E);
        // F 2304
        fc(G,F,fc1w,fc1b); // fc1w 2304x128 fc1b 128
        // G 128
        relu(H,G); 
        // H 128
        fc(I,H,fc2w,fc2b); // fc2w 128x10 fc2b 10
        // I 10
        relu(J,I);
        // J 10
        logSoftmax(K,J); 
        // K 10
    }

    void train(DataSet<T> &trainSet,DataSet<T> testSet,
        Random &train_r,Random &test_r,int num){
        
        printf("Seed:%d\n",SEED);
        printf("Lr:%f\n",learn_rate);
        printf("ImageNum:%d\n",num);

        double loss = 0;
        double corr_rate,corr_rate_l=0;
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
                corr_rate = validate(testSet,test_r,testSet.items_num);
            //     if(corr_rate>0.99)goto Ret;
            //     if(corr_rate<corr_rate_l){
            //         learn_rate *= 0.5;
            //         if(learn_rate<1e-6)goto Ret;
            //         printf("Learn_rate:%.12lf\n",learn_rate);
            //     }
            //     corr_rate_l = corr_rate;
            }
            loss += *K.data(trainSet.label[idx]);
            backward(trainSet.label[idx]);
        }
    Ret:
        corr_rate = validate(testSet,test_r,testSet.items_num);
    }

    T validate(DataSet<T> &test,Random &r,int num){
        int correct = 0,idx;
        for(int i=0;i<num;i++){
            idx = r.next();
            INPUT.set(test.image(idx,0,0));
            forward();
            if(K.maxIdx()==test.label[idx])correct++;
        }
        T corr_rate = (T)correct / num;
        printf("Acc on TestSet(%d): %.2f%%\n",num,corr_rate*100);
        return corr_rate;
    }

    void show(DataSet<T> &dataSet,Random &r){
        int idx;
        while(1){
            printf("PRESS ENTER ...\n");
            getchar();
            idx = r.next();
            dataSet.printIdx(idx);
            INPUT.set(dataSet.image(idx,0,0));
            forward();
            J.print();
            printf("Pred:%d\n",K.maxIdx());
        }   
    }
};


int main(){
    setbuf(stdout, NULL);
    DataSet<double> trainSet,testSet;
    printf("loading data...\n");
    trainSet.init("data/train-images-idx3-ubyte","data/train-labels-idx1-ubyte");
    testSet.init("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte");

    printf("init NN...\n");
    NN<double> nn(0,0.1,5e-3);

    Random train_r(trainSet.items_num),test_r(testSet.items_num);

    nn.train(trainSet,testSet,train_r,test_r,trainSet.items_num);
    
    // nn.show(trainSet,train_r);
}