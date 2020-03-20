#pragma once
#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
#include <string>
#include "Net.h"
#include "MathLib.h"
using namespace std;

class LSTMModel {
private:
	unsigned sz_input, sz_output, sz_batch, predWin;
	vector<vector<Net>> layers;
	vector<vector<vector<double>>> inputSeq;
public:
	LSTMModel(unsigned inputSize, unsigned outputSize, double eta, unsigned batchSize);
	void trainModel(vector<vector<double>> &inputSet, vector<double> &output);
	void testModel(vector<vector<double>> &inputSet, vector<double> &net_output);
	void saveModel(string &fileName);
	LSTMModel(string &fileName);
};
