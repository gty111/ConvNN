#ifndef __MNIST_DATASET__
#define __MNIST_DATASET__
#include<string>
#include<cassert>
#include<cmath>

class DataSet{
    public:
    std::string imagePath,labelPath;
    int rows,cols,items_num;
    double *_image=nullptr;
    uint8_t *_image_byte=nullptr;
    uint8_t *label=nullptr;

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

    DataSet (std::string imagePath,std::string labelPath){
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

    ~DataSet(){
        free(_image);
        free(_image_byte);
        free(label);
    }
};

#endif