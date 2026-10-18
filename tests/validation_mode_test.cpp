#include <gtest/gtest.h>
#include "../headers/tensor/tensor.h"

namespace validation = NanoChat::TensorValidation;
using NanoChat::TensorError;
using NanoChat::TensorException;

TEST(ValidationMode, OptionalPredicateRunsOnlyWhenEnabled) {
    int calls = 0;
    validation::require(
            [&] {
                ++calls;
                return true;
            },
            TensorError::InvalidShape, "optional");
    EXPECT_EQ(calls, validation::enabled ? 1 : 0);
    if constexpr (validation::enabled) {
        EXPECT_THROW(validation::require(false, TensorError::InvalidShape, "failure"), TensorException);
        EXPECT_THROW(validation::require([] { return false; }, TensorError::InvalidShape), TensorException);
    } else {
        EXPECT_NO_THROW(validation::require(false, TensorError::InvalidShape));
        EXPECT_NO_THROW(validation::require([] { return false; }, TensorError::InvalidShape));
    }
    int eagerCalls = 0;
    validation::require(++eagerCalls == 1, TensorError::InvalidShape);
    EXPECT_EQ(eagerCalls, 1);
}

TEST(ValidationMode, InvariantsAlwaysRunAndPreserveErrorDetails) {
    int calls = 0;
    validation::require_invariant(
            [&] {
                ++calls;
                return true;
            },
            TensorError::InvalidShape);
    EXPECT_EQ(calls, 1);
    try {
        validation::require_invariant(
                [&] {
                    ++calls;
                    return false;
                },
                TensorError::IndexOutOfBounds, "invariant failure");
        FAIL() << "Expected invariant failure";
    } catch (const TensorException& error) {
        EXPECT_EQ(error.code(), TensorError::IndexOutOfBounds);
        EXPECT_STREQ(error.what(), "invariant failure");
    }
    EXPECT_EQ(calls, 2);
    EXPECT_THROW(validation::require_invariant(false, TensorError::InvalidShape), TensorException);
}

TEST(ValidationMode, TensorInvariantsRemainEnabledInBothModes) {
    NanoChat::Tensor<int> tensor(2, 3);
    EXPECT_THROW(static_cast<void>(tensor[-1, 0]), TensorException);
    EXPECT_THROW(static_cast<void>(tensor[2, 0]), TensorException);
    EXPECT_THROW(static_cast<void>(tensor[0, 0, 0]), TensorException);
    EXPECT_THROW(static_cast<void>(tensor.transpose(0, 2)), TensorException);
    EXPECT_THROW(static_cast<void>(tensor.reshape({7})), TensorException);
    EXPECT_THROW(static_cast<void>(tensor.get()), TensorException);
    EXPECT_THROW((NanoChat::Tensor<int>(-1, 2)), TensorException);
    NanoChat::Tensor<int> other(6);
    EXPECT_THROW(static_cast<void>(tensor + other), TensorException);
}

TEST(ValidationMode, SharedStorageAndMaterializationWorkInBothModes) {
    NanoChat::Tensor<int> tensor(2, 3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            tensor[i, j].get() = i * 3 + j;
    auto transposed = tensor.transpose(0, 1);
    EXPECT_EQ((transposed[2, 1].get()), 5);
    auto flat = transposed.reshape({6});
    EXPECT_EQ(flat[1].get(), 3);
    flat[1].get() = 90;
    EXPECT_EQ((tensor[1, 0].get()), 3);
    auto sum = transposed + transposed;
    EXPECT_EQ((sum[2, 1].get()), 10);
    auto shared = tensor.make_contiguous();
    shared[0, 0].get() = 80;
    EXPECT_EQ((tensor[0, 0].get()), 80);
}
