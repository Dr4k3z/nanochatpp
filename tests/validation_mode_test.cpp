#include <gtest/gtest.h>
#include "../headers/tensor/tensor.h"

TEST(ValidationMode, PredicatesAreNotEvaluatedWhenDisabled) {
    namespace validation = NanoChat::TensorValidation;
    int calls = 0;
    validation::require(
            [&] {
                ++calls;
                return true;
            },
            NanoChat::TensorError::InvalidShape, "lazy check");
    const int input = 10;
    validation::validate_input(input)
            .require(
                    [&](int value) {
                        ++calls;
                        return value > 0;
                    },
                    NanoChat::TensorError::InvalidDimension, "positive")
            .require(
                    [&](int value) {
                        ++calls;
                        return value == 10;
                    },
                    NanoChat::TensorError::InvalidShape, "equals ten");
    EXPECT_EQ(calls, validation::enabled ? 3 : 0);
    int eagerCalls = 0;
    validation::require(++eagerCalls == 1, NanoChat::TensorError::InvalidShape, "Boolean arguments are eager");
    EXPECT_EQ(eagerCalls, 1);
    if constexpr (validation::enabled) {
        EXPECT_THROW(validation::require(false, NanoChat::TensorError::InvalidShape, "failure"),
                     NanoChat::TensorException);
    } else {
        EXPECT_NO_THROW(validation::require(false, NanoChat::TensorError::InvalidShape, "ignored"));
        EXPECT_NO_THROW(validation::require([] { return false; }, NanoChat::TensorError::InvalidShape, "ignored"));
    }
}

TEST(ValidationMode, ValidTensorOperationsWorkInBothModes) {
    auto tensor = NanoChat::Tensor<int>::create(2, 3);
    for (std::size_t i = 0; i < tensor.numel(); ++i)
        tensor.data()[i] = static_cast<int>(i);
    auto transposed = tensor.transpose();
    EXPECT_EQ((transposed[2, 1].get()), 5);
    auto copy = transposed.contiguous();
    EXPECT_EQ(copy.data()[1], 3);
    auto flat = copy.reshape(6);
    EXPECT_EQ(flat[5].get(), 5);
    auto sum = transposed + transposed;
    EXPECT_EQ((sum[2, 1].get()), 10);
    auto empty = NanoChat::Tensor<int>::create(0, 3);
    EXPECT_EQ(empty.transpose().contiguous().numel(), 0u);
}
