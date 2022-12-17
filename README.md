# A CNN training framework

> just for play

## Features

- An example to use [CUDNN](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html)

- A simple CNN training framework without any existing machine learning framework

- You can build CNN like building blocks through this framework

- I provide two examples(One is on [MNIST](src/mnist.cu)(achieve 98% Acc),the other is on [CIFAR-10](src/cifar.cu)(achieve 70% Acc)) to build CNN through this framework 

- whose backend is on GPU (call CUDNN to accelerate)

## Doc

[You can find document about this repo from here.](https://gty111.github.io/doc/A%20Convolutional%20Neural%20Network%20Framework%20support%20on%20CPU%20and%20GPU.pdf)

## Env

- CUDA 
- CUDNN
- CUBLAS
- GPU support CUDA and CUDNN

before run anything 
```
export CUDNN=<your path to CUDNN>
```

## Prepare data

```
make data
```

## Run example

> At each run, the standard output will be recorded in the `log` file

### MNIST
> you can find log (which I run) at [here](runlog/mnist_log)
```
make mnist
```

### CIFAR
> you can find log (which I run) at [here](runlog/cifar_log)
```
make cifar
```

## Run test

### test cudnn

```
make example 
make test_conv 
make test_fc 
make test_max_pool 
make test_relu
```

### test layer

```
make fc
make act 
make pool 
make softmax 
make batchnorm
```
