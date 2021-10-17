#include <iostream>
#include "../mnist/include/mnist/mnist_reader.hpp"
#include "../eigen/Eigen/Dense"

using namespace std;

auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

int main() {

  cout << "Hello world\n";

  return 0;
}
