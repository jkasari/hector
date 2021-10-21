#include <iostream>
#include <math.h>
#include <random>
#include "eigen/Eigen/Dense"

// Rows then Colomns!!!


void eigenTest(std::vector<double> cVec) {
  Eigen::VectorXd vec(784);
  for (int i = 0; i < 784; ++i) {
    vec(i) = cVec[i];
  }
  vec *= .1;
  double temp = vec.sum() -2750;
  std::cout << temp << std::endl;
  temp = 1 / (1 + exp(-1 * temp));
  std::cout << temp << std::endl;
  }