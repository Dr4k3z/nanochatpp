#include <iostream>
#include "headers/tensor/tensor.h"

int main() {
    auto t = NanoChat::Tensor<>::create(10, 10, 10);

    for (auto i{0uz}; i < t.shape()[0]; i++) {
        for (auto j{0uz}; j < t.shape()[1]; j++) {
            std::cout << t[i, j].vector().transpose() << '\n';
        }
    }

    return 0;
}
