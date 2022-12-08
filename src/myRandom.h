#ifndef __MYRANDOM__
#define __MYRANDOM__

#include<random>
#include<algorithm>

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

#endif