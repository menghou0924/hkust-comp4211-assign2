//
//  Net.cpp
//  hkust-comp-4321-assign2
//
//  Created by Raymond Sak on 24/4/2016.
//  Copyright © 2016 Raymond Sak. All rights reserved.
//

#include "Net.h"
using namespace std;

// constructor.
// topology is a container representing net structure.
//   e.g. {2, 4, 1} represents 2 neurons for the first layer, 4 for the second layer, 1 for last layer
// if you want to hard-code the structure, just ignore the variable topology
// eta: learning rate
Net::Net(const vector<unsigned> &topology, const double eta)
{
    m_inputNo = topology[0];
    m_hiddenNo = topology[1];
    m_outputNo = topology[2];
    m_biasNo = (int)topology.size();
    m_eta = eta;
    vector<double> temp(m_inputNo + m_hiddenNo + m_outputNo + m_biasNo);
    
    posit.inputStart = 0;
    posit.inputEnd = m_inputNo - 1;
    posit.inputBias = m_inputNo;
    posit.hiddenStart = m_inputNo + 1;
    posit.hiddenEnd = m_inputNo + 1 + m_hiddenNo - 1;
    posit.hiddenBias = m_inputNo + 1 + m_hiddenNo;
    posit.outputStart = m_inputNo + 1 + m_hiddenNo + 1;
    posit.outputEnd = m_inputNo + 1 + m_hiddenNo + 1 + m_outputNo - 1;
    posit.outputBias = m_inputNo + 1 + m_hiddenNo + 1 + m_outputNo;

    // initialize the value and error vectors
    for (int i = 0; i < m_inputNo + m_hiddenNo + m_outputNo + m_biasNo; i++)
    {
        cout << "made a new neuron" << endl;
        m_value.push_back(0);
        m_error.push_back(0);
    }
    
    // initialize bias value
    this->cleanTables();
    
    // initialize input -> hidden weight
    this->initialWeight(posit.inputStart, posit.inputEnd, posit.hiddenStart, posit.hiddenEnd, false);
    
    // initialize bias -> hidden weight
    this->initialWeight(posit.inputBias, posit.inputBias, posit.hiddenStart, posit.hiddenEnd, true);
    
    // initialize hidden -> output weight
    this->initialWeight(posit.hiddenStart, posit.hiddenEnd, posit.outputStart, posit.hiddenEnd, false);
    
    // initialize bias -> output weight
    this->initialWeight(posit.hiddenBias, posit.hiddenBias, posit.outputStart, posit.outputEnd, true);
    
    // initialize output -> empty space
    this->initialWeight(posit.outputStart, posit.outputEnd, 0, -1, false);
    
    // initialize bias -> empty space
    this->initialWeight(posit.outputBias, posit.outputBias, 0, -1, true);
    
//    this->printWeight();
}

// given an input sample inputVals, propagate input forward, compute the output of each neuron
void Net::feedForward(const vector<double> &inputVals)
{
    // reset the table
    this->cleanTables();
    
    this->printWeight();
    
    // feed in input values
    for (int i = 0; i < m_inputNo; i++)
    {
        m_value[i] = inputVals[i];
    }
    
    // update hidden values
    this->updateValue(posit.inputStart, posit.inputBias, posit.hiddenStart, posit.hiddenEnd);

    // update output values
    this->updateValue(posit.hiddenStart, posit.hiddenBias, posit.outputStart, posit.outputEnd);
    
//    this->printValue();
}

// given the vector targetVals (ground truth of output), propagate errors backward, and update each weights
void Net::backProp(const vector<double> &targetVals)
{
    m_target = targetVals;
    double outputError = 0;
    double hiddenErrorSum = 0;
    double hidden = 0;

    // update the output error
    for (int i = posit.outputStart, j = 0; i < posit.outputEnd + 1; i++, j++)
    {
        outputError = m_value[i];
        m_error[i] = outputError * (1 - outputError) * (targetVals[j] - outputError);
    }
    
    // update the hidden and bias@hidden layer error
    for (int i = posit.hiddenStart; i < posit.hiddenBias + 1; i++)
    {
        for (int j = posit.outputStart; j < posit.outputEnd + 1; j++)
        {
            hiddenErrorSum += m_weight[i][j] * m_error[j];
        }
        hidden = m_value[i];
        m_error[i] = hidden * (1 - hidden) * hiddenErrorSum;
    }
    
//    this->printError();

    // update hidden->output and bias@hidden->output weight
    this->updateWeight(posit.outputStart, posit.outputEnd, posit.hiddenStart, posit.hiddenBias);
    
    // update input->hidden and bias@input->hidden weight
    this->updateWeight(posit.hiddenStart, posit.hiddenEnd, posit.inputStart, posit.inputBias);
}

// output the prediction for the current sample to the vector resultVals
void Net::getResults(vector<double> &resultVals) const
{
    vector<double> result(m_value.begin() + posit.outputStart, m_value.begin() + posit.outputEnd + 1);
    resultVals = result;
}

// return the error of the current sample
double Net::getError(void) 
{
    // store all the errors in each iteration
    // assume only one output here
    m_allError.push_back(0.5 * pow((m_target[0] - m_value[posit.outputStart]),2));
    
    return 0.5 * pow((m_target[0] - m_value[posit.outputStart]),2) ;
}

// initialize weights with the given from and to neurons
void Net::initialWeight(int fromBegin, int fromEnd, int toBegin, int toEnd, bool isBias)
{
    vector<double> temp(m_inputNo + m_hiddenNo + m_outputNo + m_biasNo, 0);
    double randomNo = (double)rand() / RAND_MAX - 0.5;

    for (int i = fromBegin; i < fromEnd + 1; i++)
    {
        fill(temp.begin(), temp.end(), 0);
        for (int j = toBegin; j < toEnd + 1; j++)
        {
            temp[j] = isBias ? randomNo : (double)rand() * 8 / RAND_MAX - 4;
        }
        m_weight.push_back(temp);
    }
}

// feed forward: update neurons value with the given from and to neurons
void Net::updateValue(int fromBegin, int fromEnd, int toBegin, int toEnd)
{
    for (int i = toBegin; i < toEnd + 1; i++)
    {
        for (int j = fromBegin; j < fromEnd + 1; j++)
        {
            m_value[i] += m_value[j] * m_weight[j][i];
        }
        m_value[i] = sigmoid(m_value[i]);
    }
}

// back progagate: update weight with the given from and to neurons
void Net::updateWeight(int fromBegin, int fromEnd, int toBegin, int toEnd)
{
    for (int i = toBegin; i < toEnd + 1; i++)
    {
        for (int j = fromBegin; j < fromEnd + 1; j++)
        {
            m_weight[i][j] += this->m_eta * m_error[j] * m_value[i];
        }
    }
}

// reset all the values in m_value to zero
void Net::cleanTables()
{
    for(int i = 0; i < m_value.size(); i++)
    {
        m_value[i] = 0;
        m_error[i] = 0;
    }
    m_value[posit.inputBias] = 1;
    m_value[posit.hiddenBias] = 1;
    m_value[posit.outputBias] = 1;
}

// return the value transformed by the sigmoid function
double Net::sigmoid(double input) const
{
    return 1 / (1 + exp(-1 * input));
}

// print out the weight matrix
void Net::printWeight() const
{
    cout << "Weight matrix" << endl;
    for (int i = 0; i < m_weight.size(); i++)
    {
        for (int j = 0; j < m_weight[i].size(); j++)
        {
            cout << m_weight[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// print out the values at each neurons
void Net::printValue() const
{
    cout << "Value" << endl;
    for (int i = 0; i < m_value.size(); i++)
    {
        cout << i << ": " << m_value[i] << endl;
    }
    cout << endl;
}

// print out the error at each neurons
void Net::printError() const
{
    cout << "Error" << endl;
    for (int i = 0; i < m_error.size(); i++)
    {
        cout << i << ": " << m_error[i] << endl;
    }
    cout << endl;
}

// print out the vector to csv
void Net::printCSV(std::vector<unsigned> input, string name) const
{
    ofstream file(name);
    copy(input.begin(), input.end(), ostream_iterator<double>(file, ",\n"));
    file.close();
}


