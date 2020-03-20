#pragma once
#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
#include <string>
#include "Node.h"
#include "MathLib.h"
using namespace std;
class Net {
private:
	unsigned sz_input;
	unsigned sz_output;
	bool actF;
	vector<Node> layer;
	double eta;
public:
	Net() {}
	Net(unsigned inputSize, unsigned outputSize, bool type, double s);
	Net(unsigned inputSize, unsigned outputSize, bool type, double s, vector<vector<double>> &weights);
	void feedForward(vector<double> &input, vector<double> &output);
	void backProp(vector<double> &gradient);
	void getGrad(vector<double> &gradient);
	void saveNet(ofstream &fout);
};
