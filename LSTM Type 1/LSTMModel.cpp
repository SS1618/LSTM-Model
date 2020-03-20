#include "LSTMModel.h"
LSTMModel::LSTMModel(unsigned inputSize, unsigned layerNum, double eta, unsigned batchSize) {
	sz_input = inputSize;
	sz_output = inputSize;
	sz_batch = batchSize;
	cout << "Creating Model" << endl;
	for (int l = 0; l < layerNum; l++) {
		layers.push_back({ Net(2 * inputSize, inputSize, true, eta), //keepgate
			Net(inputSize * 2, inputSize, true, eta), //modgatesig
			Net(inputSize * 2, inputSize, false, eta), //modgatetanh
			Net(inputSize * 2, inputSize, true, eta) }); //outgate
	}
}
void LSTMModel::trainModel(vector<vector<double>> &inputSet, vector<double> &net_output) {
	//cout << "Training Model" << endl;
	vector<vector<double>> cell = vector<vector<double>>(layers.size(), vector<double>(sz_output, 0.0));
	vector<vector<double>> oldCell = vector<vector<double>>(layers.size());
	vector<vector<double>> prevOutput = vector<vector<double>>(layers.size(), vector<double>(sz_output, 0.0));
	vector<vector<double>> keepGate_output(layers.size());
	vector<vector<double>> modGateSig_output(layers.size());
	vector<vector<double>> modGateTanh_output(layers.size());
	vector<vector<double>> outGate_output(layers.size());
	//feed forward
	for (int i = 0; i < inputSet.size() - 1; i++) {
		vector<double> input = inputSet[i];
		for (int l = 0; l < layers.size(); l++) {
			oldCell[l] = cell[l];
			input.insert(input.end(), prevOutput[l].begin(), prevOutput[l].end());
			layers[l][0].feedForward(input, keepGate_output[l]);
			MathLib::multVec(cell[l], keepGate_output[l]);
			layers[l][1].feedForward(input, modGateSig_output[l]);
			layers[l][2].feedForward(input, modGateTanh_output[l]);
			MathLib::multVec(modGateTanh_output[l], modGateSig_output[l]);
			MathLib::addVec(cell[l], modGateTanh_output[l]);
			layers[l][3].feedForward(input, outGate_output[l]);
			vector<double> output = cell[l];
			MathLib::tanhVec(output);
			MathLib::multVec(output, outGate_output[l]);
			input = output;
			prevOutput[l] = output;
		}
	}
	//back prop
	vector<double> keepGate_grad(sz_output, 1.0);
	vector<double> modGateSig_grad(sz_output, 1.0);
	vector<double> modGateTanh_grad(sz_output, 1.0);
	vector<double> outGate_grad(sz_output, 1.0);

	for (int l = layers.size() - 1; l >= 0; l--) {
		vector<double> modCell = prevOutput[l];
		MathLib::divVec(modCell, outGate_output[l]);
		MathLib::divVec(modGateTanh_output[l], modGateSig_output[l]);
		for (int i = 0; i < sz_output; i++) {
			double e = 1.0;
			if (l == layers.size() - 1) {
				e = (inputSet.back()[i] - prevOutput.back()[i]);
			}
			keepGate_grad[i] *= (e * outGate_output[l][i] * (1.0 - (modCell[i] * modCell[i])) * oldCell[l][i]);
			modGateSig_grad[i] *= (e * outGate_output[l][i] * modGateTanh_output[l][i]);
			modGateTanh_grad[i] *= (e * outGate_output[l][i] * modGateSig_output[l][i]);
			outGate_grad[i] *= (e * modCell[i]);
		}
		layers[l][0].backProp(keepGate_grad);
		layers[l][1].backProp(modGateSig_grad);
		layers[l][2].backProp(modGateTanh_grad);
		layers[l][3].backProp(outGate_grad);
		vector<double> grad = vector<double>(sz_output, 0.0);
		layers[l][0].getGrad(grad);
		MathLib::multVec(keepGate_grad, grad);

		grad = vector<double>(sz_output, 0.0);
		layers[l][1].getGrad(grad);
		MathLib::multVec(modGateSig_grad, grad);

		grad = vector<double>(sz_output, 0.0);
		layers[l][2].getGrad(grad);
		MathLib::multVec(modGateTanh_grad, grad);

		grad = vector<double>(sz_output, 0.0);
		layers[l][3].getGrad(grad);
		MathLib::multVec(outGate_grad, grad);
	}
	for (int i = 0; i < prevOutput.back().size(); i++) {
		net_output.push_back(prevOutput.back()[i]);
	}
}
void LSTMModel::testModel(vector<vector<double>> &inputSet, vector<double> &net_output) {
	vector<vector<double>> cell = vector<vector<double>>(layers.size(), vector<double>(sz_output, 0.0));
	vector<vector<double>> oldCell = vector<vector<double>>(layers.size());
	vector<vector<double>> prevOutput = vector<vector<double>>(layers.size(), vector<double>(sz_output, 0.0));
	vector<vector<double>> keepGate_output(layers.size());
	vector<vector<double>> modGateSig_output(layers.size());
	vector<vector<double>> modGateTanh_output(layers.size());
	vector<vector<double>> outGate_output(layers.size());
	//feed forward
	for (int i = 0; i < inputSet.size(); i++) {
		vector<double> input = inputSet[i];
		for (int l = 0; l < layers.size(); l++) {
			oldCell[l] = cell[l];
			input.insert(input.end(), prevOutput[l].begin(), prevOutput[l].end());
			layers[l][0].feedForward(input, keepGate_output[l]);
			MathLib::multVec(cell[l], keepGate_output[l]);
			layers[l][1].feedForward(input, modGateSig_output[l]);
			layers[l][2].feedForward(input, modGateTanh_output[l]);
			MathLib::multVec(modGateTanh_output[l], modGateSig_output[l]);
			MathLib::addVec(cell[l], modGateTanh_output[l]);
			layers[l][3].feedForward(input, outGate_output[l]);
			vector<double> output = cell[l];
			MathLib::tanhVec(output);
			MathLib::multVec(output, outGate_output[l]);
			input = output;
			prevOutput[l] = output;
		}
	}
	for (int i = 0; i < prevOutput.back().size(); i++) {
		net_output.push_back(prevOutput.back()[i]);
	}
}
/*
number of layers
number of nets in layer 1
layer 1  net 1 input size
layer 1  net 1 output size
layer 1 net 1 eta
layer 1 net 1 activation type
layer 1 net 1 node 1 input weights
.
.
.
layer 1  net 1 node n input weights
*/
void LSTMModel::saveModel(string &fileName) {
	ofstream fout(fileName);
	fout << layers.size() << endl;
	for (int l = 0; l < layers.size(); l++) {
		fout << layers[l].size() << endl;
		for (int n = 0; n < layers[l].size(); n++) {
			layers[l][n].saveNet(fout);
		}
	}
}
LSTMModel::LSTMModel(string &fileName) {
	ifstream fin(fileName);
	unsigned layerNum;
	fin >> layerNum;
	for (int l = 0; l < layerNum; l++) {
		unsigned netNum;
		fin >> netNum;
		layers.push_back({});
		for (int n = 0; n < netNum; n++) {
			unsigned inputSz, outputSz;
			bool type;
			double eta;
			fin >> inputSz >> outputSz >> eta >> type;
			sz_input = inputSz;
			sz_output = outputSz;
			vector<vector<double>> weights;
			for (int i = 0; i < outputSz; i++) {
				weights.push_back({});
				for (int w = 0; w <= inputSz; w++) {
					double v;
					fin >> v;
					weights.back().push_back(v);
				}
			}
			layers.back().push_back(Net(inputSz, outputSz, type, eta, weights));
		}
	}
}