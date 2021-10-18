#include <iostream>
#include "mnistParser.hpp"
#include "eigen/Eigen/Dense"

using namespace std;

int main() {

  cout << "Hello world\n";
  vector<vector<double>> testImages;
  vector<vector<double>> trainImages;
  ReadMNISTImages("t10k-images-idx3-ubyte", 10000, 784, testImages);
  ReadMNISTImages("train-images-idx3-ubyte", 60000, 784, trainImages);
  cout << trainImages.size() << endl;
  cout << trainImages[0].size() << endl;
  //for (int i = 0; i < trainImages[0].size(); ++i) {
  //  cout << trainImages[9][i] << endl;
  //}

  return 0;
}
