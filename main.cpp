#include <iostream>
#include "mnistParser.hpp"
#include "logic.hpp"

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

  eigenTest(trainImages[0]);

  return 0;
}
