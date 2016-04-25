# Code details for Question 2

## Target

- Implement the back-propagation algorithm for  fully-connected neural nets
- Use stochastic gradient descent

## The files you should *NOT* modify:

- **main.cpp**: The program entry, read data file, create an instance of `Net`, call its methods to train on the data, and output results
- **TrainingData.txt**: The data file
- **TrainingData.h**, **TrainingData.cpp**: Used for reading data files in main function


## Details 

The algorithm is implemented in a class `Net`, with definition as following (in **Net.h**):

```cpp

class Net
{
public:

	/*
	    You should *not* change this part
	*/

	// constructor. 
	// topology is a container representing net structure. 
	//   e.g. {2, 4, 1} represents 2 neurons for the first layer, 4 for the second layer, 1 for last layer
	// if you want to hard-code the structure, just ignore the variable topology 
	// eta: learning rate 
	Net(const std::vector<unsigned> &topology, const double eta);

	// given a input sample inputVals, propagate input forward, compute the output of each neuron 
	void feedForward(const std::vector<double> &inputVals);

	// given the vector targetVals (ground truth of output), propagate errors backward, and update each weight
	void backProp(const std::vector<double> &targetVals);

	// output the prediction for the current sample to the vector resultVals
	void getResults(std::vector<double> &resultVals) const;

	// return the error of the current sample
	double getError(void) const;

	
	/*
	    Add what you need in the below
	*/


	// ...

private:
	// ...
};


```

- You should implement all methods listed above.
- Only c++/c standard library can be used.


## Test
- If everything goes well, output should look like the following:

```

made a new neuron
made a new neuron
made a new neuron
made a new neuron
made a new neuron
made a new neuron
made a new neuron
made a new neuron
made a new neuron
made a new neuron

Iteration 1: Inputs: 0 0 
Output: 0.900552 
Target: 0 
error: 0.405497

......

Iteration 9999: Inputs: 1 1 
Outputs: 0.0470346 
Targets: 0 
loss: 0.00110613

Iteration 10000: Inputs: 1 0 
Outputs: 0.953927 
Targets: 1 
loss: 0.00106136

Training Complete
 

```

- We will compile your program using g++ 4.9.3 and run it on another data file, so make sure it can be compiled properly.

## Note 

- The marking scheme is based on both the running result and the source code.
- Your code should be well-formatted and clearly commented.
- Plagiarism: All involved parties will get zero mark.
