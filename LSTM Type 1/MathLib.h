#pragma once
#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
#include <string>
using namespace std;
class MathLib {
public:
	static void multVec(vector<double> &a, vector<double> &b) {
		assert(a.size() == b.size());
		for (int i = 0; i < a.size(); i++) {
			a[i] *= b[i];
		}
	}
	static double dotProd(vector<double> &a, vector<double> &b) {
		assert(a.size() == b.size());
		double out = 0.0;
		for (int i = 0; i < a.size(); i++) {
			out += (a[i] * b[i]);
		}
		return out;
	}
	static double cosSim(vector<double> &a, vector<double> &b) {
		assert(a.size() == b.size());
		double magProd = sqrt(dotProd(a, a)) * sqrt(dotProd(b, b));
		double dot = dotProd(a, b);
		return dot / magProd;
	}
	static void divVec(vector<double> &a, vector<double> &b) {
		assert(a.size() == b.size());
		for (int i = 0; i < a.size(); i++) {
			a[i] /= b[i];
		}
	}
	static void addVec(vector<double> &a, vector<double> &b) {
		assert(a.size() == b.size());
		for (int i = 0; i < a.size(); i++) {
			a[i] += b[i];
		}
	}
	static void tanhVec(vector<double> &a) {
		for (int i = 0; i < a.size(); i++) {
			a[i] = tanh(a[i]);
		}
	}
	static double sigmoidFunc(double val) {
		return 1.0 / (1.0 + exp(-val));
	}
	static double tanhFunc(double val) {
		return tanh(val);
	}
};
