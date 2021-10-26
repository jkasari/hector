#include <iostream>
#include <math.h>
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
  static const uint16_t ILS = 784; // Input Layer Size
  static const uint16_t HLS = 100; // Hidden Layer Size
  static const uint16_t OLS = 10; // Output Layer Size
  Eigen::VectorXd HiddenLayerInput = Eigen::VectorXd::Random(ILS);
  Eigen::MatrixXd HiddenLayerWeights = Eigen::MatrixXd::Random(HLS, ILS);
  Eigen::VectorXd HiddenLayerBias = Eigen::VectorXd::Random(HLS);
  Eigen::VectorXd HiddenLayerOutPut(HLS);
  Eigen::MatrixXd OutPutLayerWeights = Eigen::MatrixXd::Random(OLS, HLS);
  Eigen::VectorXd OutPutLayerBias = Eigen::VectorXd::Random(OLS);
  Eigen::VectorXd FinalOutPut(OLS);


  for (int i = 0; i < 10; ++i) {

    for (int j = 0; j < ILS; ++j) {
      if (j % 28 == 0) {
        cout << endl;
      }
      if (trainImages[i][j] < 100) {
        cout << " ";
      }
      if (trainImages[i][j] < 10) {
        cout << " ";
      }
      cout << trainImages[i][j];
      HiddenLayerInput(j) = trainImages[i][j];
    }
    doTheThing(HiddenLayerInput, HiddenLayerWeights, HiddenLayerBias, HiddenLayerOutPut);
    doTheThing(HiddenLayerOutPut, OutPutLayerWeights, OutPutLayerBias, FinalOutPut);
    Eigen::VectorXd::Index row, col;
    cout << endl << endl;
    FinalOutPut.maxCoeff(&row, &col);
    cout << "Hector is fairly sure that's a ";
    cout << "\x1B[31m" << row << "\033[0m" << endl;
  }

  //cout  << trainLabels[5000] << endl;

  return 0;
}
