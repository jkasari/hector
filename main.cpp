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

  for (int i = 0; i < ILS; ++i) {
    HiddenLayerInput(i) = trainImages[0][i];
  }

  //int index = 9;
  //for (int i = 0; i < 784; ++i) {
  //  if (trainImages[index][i] < 100) {
  //    cout << " ";
  //  }
  //  if (trainImages[index][i] < 10) {
  //    cout << " ";
  //  }
  //  cout << trainImages[index][i];
  //  if (i % 28 == 0) {
  //    cout << endl;
  //  }
  //}


 // Eigen::MatrixXd hiddenWeights(3, 4);
 // hiddenWeights << 1, 2, 3, 4,
 //                  5, 6, 7, 8,
 //                  9, 1, 2, 3;
 // Eigen::VectorXd input(4);
 // input << 1, 2, 3, 4;
 // Eigen::VectorXd hiddenBias(3);
 // hiddenBias << 29, 69.5, 27;
 // Eigen::VectorXd hiddenOutPut(3);

 // Eigen::MatrixXd outPutWeights(2, 3);
 // outPutWeights << 3, 5, 6,
 //                  2, 7, 4;
 // Eigen::VectorXd outPut(2);
 // Eigen::VectorXd outPutBias(2);
 // outPutBias << 3, 4;

 // doTheThing(input, hiddenWeights, hiddenBias, hiddenOutPut);

 // doTheThing(hiddenOutPut, outPutWeights, outPutBias, outPut);


  //cout << outPut.maxCoeff() << endl;


  doTheThing(HiddenLayerInput, HiddenLayerWeights, HiddenLayerBias, HiddenLayerOutPut);
  doTheThing(HiddenLayerOutPut, OutPutLayerWeights, OutPutLayerBias, FinalOutPut);

  cout << FinalOutPut << endl;

  return 0;
}
