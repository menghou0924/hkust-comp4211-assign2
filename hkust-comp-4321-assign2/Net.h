#ifndef NET_H
#define NET_H

#include <iostream>
#include <vector>
#include <cassert>
#include <math.h>
#include <fstream>

struct position
{
    int inputStart;
    int inputEnd;
    int inputBias;
    int hiddenStart;
    int hiddenEnd;
    int hiddenBias;
    int outputStart;
    int outputEnd;
    int outputBias;
};

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

	// given an input sample inputVals, propagate input forward, compute the output of each neuron 
	void feedForward(const std::vector<double> &inputVals);

	// given the vector targetVals (ground truth of output), propagate errors backward, and update each weights
	void backProp(const std::vector<double> &targetVals);

	// output the prediction for the current sample to the vector resultVals
	void getResults(std::vector<double> &resultVals) const;

	// return the error of the current sample
	double getError(void);

	
	/*
	    Add what you need in the below
	*/

    // initialize weights with the given from and to neurons
    void initialWeight(int fromBegin, int fromEnd, int toBegin, int toEnd, bool isBias);
  
    // feed forward: update neurons value with the given from and to neurons
    void updateValue(int fromBegin, int fromEnd, int toBegin, int toEnd);
    
    // back progagate: update weight with the given from and to neurons
    void updateWeight(int fromBegin, int fromEnd, int toBegin, int toEnd);
    
    // reset the values in the m_value table;
    void cleanTables();
    
    // return the value transformed by the sigmoid function with the given input
    double sigmoid(double input) const;
    
    // print out the weight matrix
    void printWeight() const;
    
    // print out the values at each neurons
    void printValue() const;
    
    // print out the error at each neurons
    void printError() const;
    
    // print out the vector to csv
    void printCSV(std::vector<unsigned> input, std::string name) const;
    
private:
    
    int m_inputNo;
    int m_hiddenNo;
    int m_outputNo;
    int m_biasNo;
    double m_eta;
    // m_weight[from][to]
    std::vector< std::vector<double> > m_weight;
    // m_value is composed of neurons for [input] to [bias1] to [hidden] to [bias2] to [output] to [bias3]
    std::vector<double> m_value;
    // m_target to store the target of the most recent iteration
    std::vector<double> m_target;
    // m_error to store the error of each unit at the most recent iteration
    std::vector<double> m_error;
    // m_allError stores all the errors from trainings
    std::vector<double> m_allError;
    // position struct that stores the numeric position of each part of the topology
    struct position posit;
};

#endif//NET_H
