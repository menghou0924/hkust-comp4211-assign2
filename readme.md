#Simple Backward Propagation Neural Network

A simple implementation of the neural network with one hidden layer using c++.

The program could modify: 
1. number of input/hidden units
2. learning rate 
3. the initial weight setting

And the settings are:
1. both hidden/output units use the sigmoid function 
2. the standard squared error is used
3. the input, network output, target output, and error are shown in each iteration

# How to compile and run the program

The program has a Makefile. Simply type `make`, and it will compile for you!

To run the program: execute with `./backprop`

To run with your own datafile: put your datafile in the same folder of the source codes and change its name to **trainingData.txt**, and re-run the above.



