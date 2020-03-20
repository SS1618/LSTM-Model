#include "Node.h"
Node::Node(unsigned prevLayer_sz, bool sigF, double e) {
	for (int i = 0; i <= prevLayer_sz; i++) {
		weights.push_back(rand() / double(RAND_MAX * 10.0));
		int sign = rand() % 2;
		weights.back() *= (-1 * sign);
	}
	sigFunc = sigF;
	outputVal = 0;
	eta = e;
}
Node::Node(unsigned prevLayer_sz, bool sigF, double e, vector<double> &weightList) {
	for (int i = 0; i <= prevLayer_sz; i++) {
		weights.push_back(weightList[i]);
	}
	sigFunc = sigF;
	outputVal = 0;
	eta = e;
}
void Node::feedForward(vector<double> &input) {
	assert(input.size() == weights.size() - 1);
	outputVal = 0.0;
	for (int i = 0; i < input.size(); i++) {
		outputVal += (input[i] * weights[i]);
	}
	outputVal += weights.back();
	if (sigFunc) {
		outputVal = MathLib::sigmoidFunc(outputVal);
	}
	else {
		outputVal = MathLib::tanhFunc(outputVal);
	}
	inputVals = input;
}
double Node::getOutput() {
	return outputVal;
}
void Node::backProp(double grad) {
	gradients.clear();
	for (int w = 0; w < weights.size(); w++) {
		double g = grad;
		if (sigFunc) {
			g *= (outputVal * (1.0 - outputVal));
		}
		else {
			g *= (1.0 - (outputVal * outputVal));
		}
		if (w != weights.size() - 1) {
			gradients.push_back(g * weights[w]);
			g *= inputVals[w];
		}
		weights[w] += eta * g;
	}
}
void Node::getGradient(vector<double> &grad) {
	for (int i = 0; i < gradients.size() / 2; i++) { //excluding gradients for previous input nodes
		grad.push_back(gradients[i]);
	}
}
void Node::getWeights(vector<double> &w) {
	w = weights;
}