#ifndef __NN__
#define __NN__
#include "tensor.h"
class NN{
    public:
    double learn_rate;

    // forward var
    Tensor<double> conv1;
    Tensor<double> convb1;
    Tensor<double> conv2; 
    Tensor<double> convb2;
    Tensor<double> fc1w;
    Tensor<double> fc1b; 
    Tensor<double> fc2w;
    Tensor<double> fc2b;

    // intermediate var
    Tensor<double> INPUT;
    Tensor<double> A;
    Tensor<double> B;
    Tensor<double> C;
    Tensor<double> D;
    Tensor<double> E;
    Tensor<double> F;
    Tensor<double> G;
    Tensor<double> H;
    Tensor<double> I;
    Tensor<double> J; 
    Tensor<double> OUTPUT;

    // to cal B grad
    Tensor<double> CgradPad;

    void backward(int label){
        assert(label>=0&&label<=9);

        // ----J grad----
        double maximum = J.max();

        double sum = 0;
        for(int i=0;i<J.shape[0];i++){
            sum += std::exp(*J.data(i)-maximum);
        }
        
        for(int i=0;i<J.shape[0];i++){
            *J.grad(i) = i==label ? 
                1 - std::exp(*J.data(i)-maximum) / sum:
                - std::exp(*J.data(i)-maximum) / sum  ;
        }

        // ----I grad----
        for(int i=0;i<I.shape[0];i++){
            *I.grad(i) = *I.data(i)>0 ? *J.grad(i) : 0;
        }

        // ----H grad----
        for(int i=0;i<fc2w.shape[0];i++){
            double sum = 0;
            for(int j=0;j<fc2w.shape[1];j++){
                sum += *fc2w.data(i,j) * (*I.grad(j));
            }
            *H.grad(i) = sum;
        }

        // ----G grad----
        for(int i=0;i<G.shape[0];i++){
            *G.grad(i) = *G.data(i)>0 ? *H.grad(i) : 0;
        }

        // ----fc2w grad----
        for(int i=0;i<fc2w.shape[0];i++){
            for(int j=0;j<fc2w.shape[1];j++){
                *fc2w.data(i,j) += (*H.data(i)) * (*I.grad(j)) * this->learn_rate;
            }
        }

        // ----fc2b grad----
        for(int i=0;i<fc2b.shape[0];i++){
            *fc2b.data(i) += (*I.grad(i)) * this->learn_rate;
        }

        // ----F grad----
        for(int i=0;i<F.shape[0];i++){
            double sum = 0;
            for(int j=0;j<G.shape[0];j++){
                sum += *fc1w.data(i,j) * (*G.grad(j));
            }
            *F.grad(i) = sum;
        }
        
        // ----fc1w grad----
        for(int i=0;i<fc1w.shape[0];i++){
            for(int j=0;j<fc1w.shape[1];j++){
                *fc1w.data(i,j) += (*G.grad(j)) * (*F.data(i)) * this->learn_rate;
            }
        }

        // ----fc1b grad----
        for(int i=0;i<fc1b.shape[0];i++){
            *fc1b.data(i) += (*G.grad(i)) * this->learn_rate;
        }

        // ----E grad----
        for(int i=0;i<E.shape[0];i++){
            for(int j=0;j<E.shape[1];j++){
                for(int k=0;k<E.shape[2];k++){
                    *E.grad(i,j,k) = *F.grad(i*12*12+j*12+k);
                }
            }
        }

        // ----D grad----
        for(int i=0;i<D.shape[0];i++){
            for(int j=0;j<D.shape[1];j++){
                for(int k=0;k<D.shape[2];k++){
                    *D.grad(i,j,k) = *E.grad(i,j/2,k/2) / 4;
                }
            }
        }

        // ----C grad----
        for(int i=0;i<C.shape[0];i++){
            for(int j=0;j<C.shape[1];j++){
                for(int k=0;k<C.shape[2];k++){
                    *C.grad(i,j,k) = *C.data(i,j,k)>0 ? *D.grad(i,j,k) : 0;
                }
            }
        }

        // ----B grad----
        for(int i=0;i<C.shape[0];i++){
            for(int j=0;j<C.shape[1];j++){
                for(int k=0;k<C.shape[2];k++){
                    *CgradPad.data(i,j+2,k+2) = *C.grad(i,j,k);
                }
            }
        }
        for(int i=0;i<B.shape[0];i++){
            for(int j=0;j<B.shape[1];j++){
                for(int k=0;k<B.shape[2];k++){
                    double sum = 0;
                    for(int m=0;m<conv2.shape[0];m++){
                        for(int p=0;p<conv2.shape[2];p++){
                            for(int q=0;q<conv2.shape[3];q++){
                                sum += *CgradPad.data(m,j+p,k+q) * (*conv2.data(m,i,2-p,2-q));
                            }
                        }
                    }
                    *B.grad(i,j,k) = sum;
                }
            }
        }

        

        // ----conv2 grad----
        for(int i=0;i<conv2.shape[0];i++){
            for(int j=0;j<conv2.shape[1];j++){
                for(int p=0;p<conv2.shape[2];p++){
                    for(int q=0;q<conv2.shape[3];q++){
                        double sum = 0;
                        for(int m=0;m<C.shape[1];m++){
                            for(int n=0;n<C.shape[2];n++){
                                sum += *B.data(j,m+p,n+q) * (*C.grad(i,m,n));
                            }
                        }
                        *conv2.data(i,j,p,q) += sum * this->learn_rate;
                    }
                }
            }
        }


        // ----convb2 grad----
        for(int i=0;i<C.shape[0];i++){
            for(int j=0;j<C.shape[1];j++){
                for(int k=0;k<C.shape[2];k++){
                    *convb2.data(i) += *C.grad(i,j,k) * this->learn_rate;    
                }
            }
        }

        // ----A grad----
        for(int i=0;i<A.shape[0];i++){
            for(int j=0;j<A.shape[1];j++){
                for(int k=0;k<A.shape[2];k++){
                    *A.grad(i,j,k) = *A.data(i,j,k)>0 ? *B.grad(i,j,k) : 0;
                }
            }
        }

        // ----conv1 grad----
        for(int i=0;i<conv1.shape[0];i++){
            for(int p=0;p<conv1.shape[1];p++){
                for(int j=0;j<conv1.shape[2];j++){
                    for(int k=0;k<conv1.shape[3];k++){
                        double sum = 0;
                        for(int m=0;m<A.shape[1];m++){
                            for(int n=0;n<A.shape[2];n++){
                                sum += *A.grad(i,m,n) * (*INPUT.data(p,m+j,n+k));
                            }
                        }
                        *conv1.data(i,p,j,k) += sum * this->learn_rate;
                    }
                }
            }
        }

        // ----convb1 grad----
        for(int i=0;i<A.shape[0];i++){
            double sum = 0;
            for(int j=0;j<A.shape[1];j++){
                for(int k=0;k<A.shape[2];k++){
                    sum += *A.grad(i,j,k);
                }
            }
            *convb1.data(i) += sum * this->learn_rate;
        }
        

    }

    void forward(){

        this->conv(A,INPUT,conv1,convb1); 

        this->relu(B,A); 

        this->conv(C,B,conv2,convb2); 

        this->relu(D,C);

        this->meanPool(E,D);

        this->flattern(F,E);

        this->fc(G,F,fc1w,fc1b);

        this->relu(H,G); 

        this->fc(I,H,fc2w,fc2b); 

        this->relu(J,I);

        this->logSoftmax(OUTPUT,J); 

    }

    void conv(Tensor<double> &output,Tensor<double> &input,Tensor<double> &kernel,Tensor<double> &bias){
        assert(kernel.shape.size()==4);
        assert(input.shape.size()==3);
        assert(kernel.shape[1]==input.shape[0]);
        assert(kernel.shape[0]==bias.shape[0]);
        assert(bias.shape.size()==1);
        assert(output.shape[0]==kernel.shape[0]);

        for(int i=0;i<kernel.shape[0];i++){
            for(int m=0;m<output.shape[1];m++){
                for(int n=0;n<output.shape[2];n++){
                    double sum = 0;
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

    void relu(Tensor<double> &output,Tensor<double> &input){
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

    void meanPool(Tensor<double> &output,Tensor<double> &input){
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

    void flattern(Tensor<double> &output,Tensor<double> &input){
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

    void fc(Tensor<double> &output,Tensor<double> &input,Tensor<double> &weight,Tensor<double> &bias){
        assert(input.shape.size()==1);
        assert(weight.shape.size()==2);
        for(int i=0;i<weight.shape[1];i++){
            double sum = 0;
            for(int j=0;j<weight.shape[0];j++){
                sum += (*input.data(j)) * (*weight.data(j,i)) ;
            }
            (*output.data(i)) = sum + (*bias.data(i));
        }
    }

    void logSoftmax(Tensor<double> &output,Tensor<double> &input){
        assert(input.shape.size()==1);
        double maximum = input.max();
        double sum = 0;
        for(int i=0;i<input.shape[0];i++){
            sum += std::exp(*input.data(i)-maximum);
        }
        for(int i=0;i<input.shape[0];i++){
            (*output.data(i)) = *input.data(i) - maximum - std::log(sum);
        }
    }
};

#endif