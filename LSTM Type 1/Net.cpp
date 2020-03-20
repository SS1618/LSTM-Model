#include "Net.h"
Net::Net(unsigned inputSize, unsigned outputSize, bool type, double s) {
	sz_input = inputSize;
	sz_output = outputSize;
	actF = type;
	for (int n = 0; n < outputSize; n++) {
		layer.push_back(Node(inputSize, type, s));
	}
	eta = s;
}
Net::Net(unsigned inputSize, unsigned outputSize, bool type, double s, vector<vector<double>> &weights) {
	sz_input = inputSize;
	sz_output = outputSize;
	actF = type;
	assert(outputSize == weights.size());
	for (int n = 0; n < outputSize; n++) {
		layer.push_back(Node(inputSize, type, s, weights[n]));
	}
	eta = s;
}
void Net::feedForward(vector<double> &input, vector<double> &output) {
	assert(sz_input == input.size());
	output.clear();
	for (int n = 0; n < layer.size(); n++) {
		layer[n].feedForward(input);
		output.push_back(layer[n].getOutput());
	}
}
void Net::backProp(vector<double> &gradient) {
	assert(gradient.size() == layer.size());
	for (int n = 0; n < layer.size(); n++) {
		layer[n].backProp(gradient[n]);
	}
}
void Net::getGrad(vector<double> &gradient) {
	for (int n = 0; n < layer.size(); n++) {
		vector<double> g;
		layer[n].getGradient(g);
		MathLib::addVec(gradient, g);
	}
}
void Net::saveNet(ofstream &fout) {
	fout << sz_input << endl;
	fout << sz_output << endl;
	fout << eta << endl;
	fout << actF << endl;
	vector<double> weights;
	for (int n = 0; n < layer.size(); n++) {
		layer[n].getWeights(weights);
		for (int w = 0; w < weights.size(); w++) {
			fout << weights[w] << endl;
		}
	}
}