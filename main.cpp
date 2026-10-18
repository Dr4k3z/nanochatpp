#include <iostream>
#include "headers/tensor/tensor.h"

int main() {
    NanoChat::Tensor<> tensor(10, 10, 10);
    for (std::size_t i = 0; i < tensor.get_shape()[0]; ++i) {
        for (std::size_t j = 0; j < tensor.get_shape()[1]; ++j) {
            for (std::size_t k = 0; k < tensor.get_shape()[2]; ++k)
                std::cout << tensor[i, j, k].get() << ' ';
            std::cout << '\n';
        }
    }
}
