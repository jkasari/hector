#include <iostream>
#include <math.h>
#include "eigen/Eigen/Dense"

using namespace Eigen;
 
void eigenTest(std::vector<double> cVec) {
  double* ptr = &cVec[0];
  Eigen::Map<Eigen::VectorXd> vec(ptr, 784);

  vec *= .1;
  double temp = vec.sum() -2750;
  std::cout << temp << std::endl;
  temp = 1 / (1 + exp(-1 * temp));
  std::cout << temp << std::endl;
  }