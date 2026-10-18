#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include "../headers/tensor/tensor.h"

using NanoChat::Tensor;
using NanoChat::TensorError;
using NanoChat::TensorException;

namespace {
    template<typename Operation>
    void expect_tensor_error(Operation operation, TensorError code, const char* message) {
        try {
            operation();
            FAIL() << "Expected TensorException";
        } catch (const TensorException& error) {
            EXPECT_EQ(error.code(), code);
            EXPECT_STREQ(error.what(), message);
        }
    }

    Tensor<int> sequence() {
        Tensor<int> tensor(2, 3, 4);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 4; ++k)
                    tensor[i, j, k].get() = i * 12 + j * 4 + k;
        return tensor;
    }
} // namespace

TEST(TensorConstruction, ShapeAndDimensionsAgreeAndInitializeToZero) {
    Tensor<> dimensions(2, 3, 4);
    Tensor<> shape(Tensor<>::Shape{2, 3, 4});
    EXPECT_EQ(dimensions.get_shape(), shape.get_shape());
    EXPECT_EQ(dimensions.get_strides(), (Tensor<>::Shape{12, 4, 1}));
    EXPECT_EQ(dimensions.get_ndim(), 3u);
    EXPECT_EQ(dimensions.get_numel(), 24u);
    EXPECT_TRUE(dimensions.is_contiguous());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ((shape[i, j, k].get()), 0.0f);
}

TEST(TensorConstruction, SupportsScalarTypesAndMixedIntegralDimensions) {
    Tensor<double> tensor(2u, std::size_t{3});
    tensor[1, 2].get() = 1.25;
    static_assert(std::is_same_v<decltype(tensor[1, 2].get()), double&>);
    EXPECT_DOUBLE_EQ((tensor[1, 2].get()), 1.25);
    Tensor<int> integers(1);
    integers[0].get() = 7;
    EXPECT_EQ(integers[0].get(), 7);
}

TEST(TensorConstruction, SupportsRankZeroAndEmptyAxes) {
    Tensor<int> scalar(Tensor<int>::Shape{});
    EXPECT_EQ(scalar.get_ndim(), 0u);
    EXPECT_EQ(scalar.get_numel(), 1u);
    scalar.get() = 9;
    EXPECT_EQ(scalar.get(), 9);
    Tensor<int> empty(2, 0, 3);
    EXPECT_EQ(empty.get_shape(), (Tensor<int>::Shape{2, 0, 3}));
    EXPECT_EQ(empty.get_numel(), 0u);
    EXPECT_TRUE(empty.is_contiguous());
    EXPECT_EQ(empty.clone().get_numel(), 0u);
    EXPECT_EQ(empty.transpose(0, 2).get_shape(), (Tensor<int>::Shape{3, 0, 2}));
    EXPECT_EQ(empty.reshape({0, 6}).get_shape(), (Tensor<int>::Shape{0, 6}));
}

TEST(TensorConstruction, RejectsNegativeDimensionsAndOverflow) {
    expect_tensor_error([] { Tensor<> tensor(2, -1); }, TensorError::InvalidDimension, "Invalid tensor dimension");
    expect_tensor_error([] { Tensor<> tensor(Tensor<>::Shape{std::numeric_limits<std::size_t>::max(), 2}); },
                        TensorError::SizeOverflow, "Tensor element count overflow");
    expect_tensor_error([] { Tensor<> tensor(Tensor<>::Shape{0, std::numeric_limits<std::size_t>::max(), 2}); },
                        TensorError::SizeOverflow, "Tensor stride overflow");
    expect_tensor_error([] { Tensor<> tensor(Tensor<>::Shape{std::numeric_limits<std::size_t>::max()}); },
                        TensorError::SizeOverflow, "Tensor exceeds Eigen index range");
}

TEST(TensorIndexing, PrefixSlicesAndChainedIndicesShareStorage) {
    auto tensor = sequence();
    auto matrix = tensor[1];
    auto vector = matrix[2];
    EXPECT_EQ(matrix.get_shape(), (Tensor<int>::Shape{3, 4}));
    EXPECT_EQ(matrix.get_strides(), (Tensor<int>::Shape{4, 1}));
    EXPECT_EQ(vector.get_shape(), (Tensor<int>::Shape{4}));
    EXPECT_EQ(vector[3].get(), 23);
    vector[3].get() = 99;
    EXPECT_EQ((tensor[1, 2, 3].get()), 99);
    EXPECT_EQ(tensor[1][2][3].get(), 99);
    EXPECT_EQ((tensor[0, 2, 3].get()), 11);
    EXPECT_EQ((tensor[1, 2, 3].get_ndim()), 0u);
}

TEST(TensorIndexing, RejectsWrongRankNegativeAndOutOfRangeIndices) {
    Tensor<int> tensor(2, 3);
    expect_tensor_error([&] { static_cast<void>(tensor[0, 0, 0]); }, TensorError::IndexOutOfBounds,
                        "Too many indices for tensor rank");
    expect_tensor_error([&] { static_cast<void>(tensor[-1, 0]); }, TensorError::IndexOutOfBounds,
                        "Tensor index cannot be negative or exceed size_t");
    expect_tensor_error([&] { static_cast<void>(tensor[0, 3]); }, TensorError::IndexOutOfBounds,
                        "Tensor index exceeds dimension");
    expect_tensor_error([&] { static_cast<void>(tensor[2, 0]); }, TensorError::IndexOutOfBounds,
                        "Tensor index exceeds dimension");
    expect_tensor_error([&] { static_cast<void>(tensor.get()); }, TensorError::InvalidShape,
                        "get() requires a rank-zero tensor");
    Tensor<int> empty(2, 0, 3);
    expect_tensor_error([&] { static_cast<void>(empty[0, 0]); }, TensorError::IndexOutOfBounds,
                        "Tensor index exceeds dimension");
}

TEST(TensorOwnership, CopiesAndAssignmentsShareStorageButKeepTheirOwnMetadata) {
    auto tensor = sequence();
    auto copy = tensor;
    auto reshaped = copy.reshape({6, 4});
    reshaped[5, 3].get() = 42;
    EXPECT_EQ((tensor[1, 2, 3].get()), 42);
    EXPECT_EQ(copy.get_shape(), (Tensor<int>::Shape{2, 3, 4}));
    Tensor<int> assigned(1);
    assigned = tensor;
    assigned[0, 0, 0].get() = 17;
    EXPECT_EQ((tensor[0, 0, 0].get()), 17);
}

TEST(TensorOwnership, SlicesSurviveSourceDestructionReassignmentAndTemporaries) {
    auto slice = [] { return sequence()[1]; }();
    EXPECT_EQ((slice[2, 3].get()), 23);
    auto tensor = sequence();
    auto retained = tensor[1, 2];
    tensor = Tensor<int>(1);
    EXPECT_EQ(retained[3].get(), 23);
    auto temporarySlice = sequence().transpose(0, 2)[3];
    EXPECT_EQ((temporarySlice[2, 1].get()), 23);
    retained[3].get() = 81;
    EXPECT_EQ(retained[3].get(), 81);
}

TEST(TensorOwnership, CloneIsIndependentForBothContiguousAndStridedData) {
    auto tensor = sequence();
    auto copy = tensor.clone();
    copy[1, 2, 3].get() = 70;
    EXPECT_EQ((tensor[1, 2, 3].get()), 23);
    auto transposed = tensor.transpose(0, 2);
    auto packed = transposed.clone();
    EXPECT_TRUE(packed.is_contiguous());
    EXPECT_EQ(packed.get_shape(), transposed.get_shape());
    EXPECT_EQ(packed.get_strides(), (Tensor<int>::Shape{6, 2, 1}));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 2; ++k)
                EXPECT_EQ((packed[i, j, k].get()), (tensor[k, j, i].get()));
    packed[3, 2, 1].get() = 100;
    EXPECT_EQ((tensor[1, 2, 3].get()), 23);
    auto offsetCopy = tensor[1, 2].clone();
    EXPECT_EQ(offsetCopy[0].get(), 20);
    offsetCopy[3].get() = 50;
    EXPECT_EQ((tensor[1, 2, 3].get()), 23);
}

TEST(TensorLayout, TransposeSwapsAxesAndPreservesOffsets) {
    auto tensor = sequence();
    auto transposed = tensor.transpose(0, 2);
    EXPECT_EQ(transposed.get_shape(), (Tensor<int>::Shape{4, 3, 2}));
    EXPECT_EQ(transposed.get_strides(), (Tensor<int>::Shape{1, 4, 12}));
    EXPECT_FALSE(transposed.is_contiguous());
    EXPECT_EQ((transposed[3, 2, 1].get()), 23);
    transposed[3, 2, 1].get() = 60;
    EXPECT_EQ((tensor[1, 2, 3].get()), 60);
    EXPECT_EQ(tensor.get_shape(), (Tensor<int>::Shape{2, 3, 4}));
    EXPECT_TRUE(transposed.transpose(0, 2).is_contiguous());
    EXPECT_EQ(tensor.transpose(1, 1).get_strides(), tensor.get_strides());
    auto matrix = tensor[1].transpose(0, 1);
    EXPECT_EQ((matrix[3, 2].get()), 60);
    expect_tensor_error([&] { static_cast<void>(tensor.transpose(0, 3)); }, TensorError::IndexOutOfBounds,
                        "Transpose axes outside rank");
}

TEST(TensorLayout, PermuteReordersAllAxesAndSharesData) {
    auto tensor = sequence();
    auto permuted = tensor.permute(std::array<std::size_t, 3>{2, 0, 1});
    EXPECT_EQ(permuted.get_shape(), (Tensor<int>::Shape{4, 2, 3}));
    EXPECT_EQ(permuted.get_strides(), (Tensor<int>::Shape{1, 12, 4}));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 3; ++k)
                EXPECT_EQ((permuted[i, j, k].get()), (tensor[j, k, i].get()));
    permuted[3, 1, 2].get() = 80;
    EXPECT_EQ((tensor[1, 2, 3].get()), 80);
    auto restored = permuted.permute(std::array<std::size_t, 3>{1, 2, 0});
    EXPECT_EQ(restored.get_shape(), tensor.get_shape());
    EXPECT_TRUE(restored.is_contiguous());
    auto offset = tensor[1].permute(std::array<std::size_t, 2>{1, 0});
    EXPECT_EQ((offset[3, 2].get()), 80);
    Tensor<int> scalar(Tensor<int>::Shape{});
    EXPECT_EQ(scalar.permute(std::array<std::size_t, 0>{}).get_numel(), 1u);
}

TEST(TensorLayout, RejectsIncompleteDuplicateAndOutOfRangePermutations) {
    Tensor<int> tensor(2, 3, 4);
    expect_tensor_error([&] { static_cast<void>(tensor.permute(std::array<std::size_t, 2>{0, 1})); },
                        TensorError::InvalidShape, "Permutation must include every axis");
    expect_tensor_error([&] { static_cast<void>(tensor.permute(std::array<std::size_t, 3>{0, 0, 2})); },
                        TensorError::InvalidShape, "Duplicate permutation axis");
    expect_tensor_error([&] { static_cast<void>(tensor.permute(std::array<std::size_t, 3>{0, 1, 3})); },
                        TensorError::IndexOutOfBounds, "Permutation axis outside rank");
}

TEST(TensorLayout, ContiguousReshapeSharesStorageIncludingSliceOffsets) {
    auto tensor = sequence();
    auto reshaped = tensor.reshape({6, 4});
    EXPECT_EQ(reshaped.get_strides(), (Tensor<int>::Shape{4, 1}));
    reshaped[5, 3].get() = 51;
    EXPECT_EQ((tensor[1, 2, 3].get()), 51);
    auto offset = tensor[1].reshape({12});
    EXPECT_EQ(offset[0].get(), 12);
    offset[11].get() = 71;
    EXPECT_EQ((tensor[1, 2, 3].get()), 71);
    Tensor<int> scalar(Tensor<int>::Shape{});
    scalar.get() = 9;
    auto vector = scalar.reshape({1});
    vector[0].get() = 10;
    EXPECT_EQ(scalar.get(), 10);
    EXPECT_EQ(vector.reshape({}).get(), 10);
    expect_tensor_error([&] { static_cast<void>(tensor.reshape({25})); }, TensorError::InvalidShape,
                        "Reshape must preserve the number of elements");
}

TEST(TensorLayout, StridedReshapeCopiesInLogicalOrder) {
    auto tensor = sequence();
    auto flat = tensor[1].transpose(0, 1).reshape({12});
    const std::array expected{12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23};
    EXPECT_TRUE(flat.is_contiguous());
    for (int i = 0; i < 12; ++i)
        EXPECT_EQ(flat[i].get(), expected[i]);
    flat[0].get() = 100;
    EXPECT_EQ((tensor[1, 0, 0].get()), 12);
}

TEST(TensorLayout, MakeContiguousSharesPackedStorageAndCopiesStridedStorage) {
    auto tensor = sequence();
    auto shared = tensor.make_contiguous();
    shared[0, 0, 0].get() = 30;
    EXPECT_EQ((tensor[0, 0, 0].get()), 30);
    auto slice = tensor[1].make_contiguous();
    slice[2, 3].get() = 40;
    EXPECT_EQ((tensor[1, 2, 3].get()), 40);
    auto packed = tensor.transpose(0, 2).make_contiguous();
    EXPECT_TRUE(packed.is_contiguous());
    EXPECT_EQ((packed[3, 2, 1].get()), 40);
    packed[3, 2, 1].get() = 90;
    EXPECT_EQ((tensor[1, 2, 3].get()), 40);
    Tensor<int> singleton(1, 3);
    EXPECT_TRUE(singleton.transpose(0, 1).is_contiguous());
}

TEST(TensorArithmetic, StridedAndOffsetOperandsAreEvaluatedInLogicalOrder) {
    auto tensor = sequence();
    auto left = tensor[1].transpose(0, 1);
    auto right = tensor[0].transpose(0, 1);
    auto sum = left + right;
    auto product = left * right;
    auto namedSum = left.sum(right);
    auto namedProduct = left.product(right);
    EXPECT_TRUE(sum.is_contiguous());
    EXPECT_EQ(sum.get_shape(), left.get_shape());
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j) {
            const int a = 12 + j * 4 + i;
            const int b = j * 4 + i;
            EXPECT_EQ((sum[i, j].get()), a + b);
            EXPECT_EQ((product[i, j].get()), a * b);
            EXPECT_EQ((namedSum[i, j].get()), a + b);
            EXPECT_EQ((namedProduct[i, j].get()), a * b);
        }
    sum[0, 0].get() = 70;
    EXPECT_EQ((tensor[1, 0, 0].get()), 12);
    expect_tensor_error([&] { static_cast<void>(left + tensor); }, TensorError::IncompatibleShapes,
                        "Tensor operands must have identical shapes");
    expect_tensor_error([&] { static_cast<void>(left * tensor); }, TensorError::IncompatibleShapes,
                        "Tensor operands must have identical shapes");
}

TEST(TensorArithmetic, ScalarAndEmptyOperands) {
    Tensor<double> a(Tensor<double>::Shape{}), b(Tensor<double>::Shape{});
    a.get() = 1.5;
    b.get() = 2.0;
    EXPECT_DOUBLE_EQ((a + b).get(), 3.5);
    EXPECT_DOUBLE_EQ((a * b).get(), 3.0);
    Tensor<int> empty(0, 3);
    EXPECT_EQ((empty + empty).get_shape(), empty.get_shape());
    EXPECT_EQ((empty * empty).get_numel(), 0u);
}

TEST(TensorAccess, DirectConstScalarAccessIsReadOnly) {
    Tensor<double> tensor(Tensor<double>::Shape{});
    const auto& readonly = tensor;
    static_assert(std::is_same_v<decltype(tensor.get()), double&>);
    static_assert(std::is_same_v<decltype(readonly.get()), const double&>);
    tensor.get() = 3.5;
    EXPECT_DOUBLE_EQ(readonly.get(), 3.5);
}

TEST(TensorArithmetic, OperandsCanHaveDifferentLayouts) {
    Tensor<int> tensor(2, 2);
    tensor[0, 0].get() = 1;
    tensor[0, 1].get() = 2;
    tensor[1, 0].get() = 3;
    tensor[1, 1].get() = 4;
    auto transposed = tensor.transpose(0, 1);
    auto leftPacked = tensor + transposed;
    auto rightPacked = transposed + tensor;
    auto product = tensor * transposed;
    EXPECT_EQ((leftPacked[0, 1].get()), 5);
    EXPECT_EQ((rightPacked[1, 0].get()), 5);
    EXPECT_EQ((product[0, 1].get()), 6);
    EXPECT_EQ((product[1, 0].get()), 6);
    product[0, 1].get() = 100;
    EXPECT_EQ((tensor[0, 1].get()), 2);
}
