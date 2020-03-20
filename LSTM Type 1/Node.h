#pragma once
#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
#include <string>
#include "MathLib.h"
using namespace std;
class Node {
private:
	vector<double> weights;
	vector<double> gradients;
	vector<double> inputVals;
	double outputVal;
	bool sigFunc;
	double eta;
public:
	Node(unsigned prevLayer_sz, bool sigF, double e);
	Node(unsigned prevLayer_sz, bool sigF, double e, vector<double> &weightList);
	void feedForward(vector<double> &input);
	double getOutput();
	void backProp(double grad);
	void getGradient(vector<double> &grad);
	void getWeights(vector<double> &w);
};
