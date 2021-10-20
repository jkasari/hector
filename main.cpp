#include <iostream>
#include "mnistParser.hpp"
#include "eigen/Eigen/Dense"

using namespace std;

int main() {

  cout << "Hello Hector\n";
  vector<vector<double>> testImages;
  vector<vector<double>> trainImages;
  vector<double> testLabels;
  vector<double> trainLabels;
  ReadMNISTImages("t10k-images-idx3-ubyte", 10000, 784, testImages);
  ReadMNISTImages("train-images-idx3-ubyte", 60000, 784, trainImages);
  ReadMNISTLabels("t10k-labels-idx1-ubyte", 10000, testLabels);
  ReadMNISTLabels("train-labels-idx1-ubyte", 60000, trainLabels);
  cout << testImages.size() << endl;
  cout << trainImages.size() << endl;
  cout << testLabels.size() << endl;
  cout << trainLabels.size() << endl;

  return 0;
}
