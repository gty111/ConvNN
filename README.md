# A CNN training framework

> just for play

## Features

- A simple CNN training framework without any existing machine learning framework

- You can build DNN like building blocks through this framework

- I provide two examples(One is on MNIST,the other is on CIFAR-10) to build CNN through this framework 

- whose backend is on cpu (try GPU version: checkout cudnn)

## Doc

[You can find document about this repo from here.](https://gty111.github.io/doc/A%20Convolutional%20Neural%20Network%20without%20AI.pdf)

## Prepare Data Set

```
make data
```

## Run

> At each run, the standard output will be recorded in the `log` file

### MNIST

```
make mnist
```

### CIFAR-10

```
make cifar
```
