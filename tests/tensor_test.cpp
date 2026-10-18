#include "../headers/tensor/tensor.h"
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>


using NanoChat::Tensor;
using NanoChat::TensorError;
using NanoChat::TensorException;

template<typename Operation>
void expect_tensor_error(Operation operation, TensorError code) {
    try {
        operation();
        FAIL() << "Expected TensorException";
    } catch (const TensorException& error) {
        EXPECT_EQ(error.code(), code);
        EXPECT_NE(std::string(error.what()).size(), 0u);
    }
}

TEST(TensorFactory, VariadicAndShapeFactoryAgree) {
    auto variadic = Tensor<>::create(2, 3, 4);
    Tensor<>::Shape shape{2, 3, 4};
    auto from_shape = Tensor<>::create(shape);
    EXPECT_EQ(variadic.shape(), from_shape.shape());
    EXPECT_EQ(variadic.strides(), from_shape.strides());
    EXPECT_EQ(variadic.numel(), 24u);
    EXPECT_EQ(variadic.strides(), (Tensor<>::Shape{12, 4, 1}));
    for (float x: variadic.data())
        EXPECT_FLOAT_EQ(x, 0.0f);
}

TEST(TensorFactory, SupportsDifferentScalarTypes) {
    auto tensor = Tensor<double>::create(2u, std::size_t{3});
    static_assert(std::is_same_v<decltype(tensor.data()[0]), double&>);
    EXPECT_EQ(tensor.numel(), 6u);
}

TEST(TensorFactory, RejectsNegativeDimensions) {
    expect_tensor_error([] { Tensor<>::create(2, -1, 4); }, TensorError::InvalidDimension);
}
TEST(TensorFactory, DetectsSizeOverflow) {
    expect_tensor_error([] { Tensor<>::create(NanoChat::Shape{std::numeric_limits<std::size_t>::max(), 2}); },
                        TensorError::SizeOverflow);
}

TEST(TensorFactory, SupportsEmptyAxes) {
    auto result = Tensor<>::create(2, 0, 3);
    EXPECT_EQ(result.numel(), 0u);
    EXPECT_EQ(result.shape(), (Tensor<>::Shape{2, 0, 3}));
    EXPECT_TRUE(result.data().empty());
}

TEST(TensorFactory, SupportsScalarShape) {
    auto result = Tensor<>::create(Tensor<>::Shape{});
    EXPECT_EQ(result.ndim(), 0u);
    EXPECT_EQ(result.numel(), 1u);
}

TEST(TensorIndexing, MutableAndConstRead) {
    auto result = Tensor<>::create(2, 3, 4);
    auto& tensor = result;
    auto element = tensor[1, 2, 3];
    element.get() = 7.5f;
    const auto& readonly = tensor;
    auto value = readonly[1, 2, 3];
    EXPECT_FLOAT_EQ(value.get(), 7.5f);
    EXPECT_FLOAT_EQ(tensor.data()[23], 7.5f);
}

TEST(TensorIndexing, RejectsInvalidIndices) {
    auto result = Tensor<>::create(2, 3);
    auto& tensor = result;
    EXPECT_EQ(tensor[1].ndim(), 1u);
    EXPECT_THROW((tensor[1].get()), TensorException);
    EXPECT_THROW((tensor[1, 0, 0]), TensorException);
    EXPECT_THROW((tensor[2, 0]), TensorException);
    EXPECT_THROW((tensor[-1, 0]), TensorException);
}

TEST(TensorArithmetic, ElementwiseOperations) {
    auto a = Tensor<>::create(2, 2);
    auto b = Tensor<>::create(2, 2);
    for (std::size_t i = 0; i < 4; ++i) {
        a.data()[i] = static_cast<float>(i + 1);
        b.data()[i] = 2.0f;
    }
    auto sum = a + b;
    auto diff = a - b;
    auto product = a * b;
    auto negative = -a;
    EXPECT_FLOAT_EQ(sum.data()[3], 6.0f);
    EXPECT_FLOAT_EQ(diff.data()[3], 2.0f);
    EXPECT_FLOAT_EQ(product.data()[3], 8.0f);
    EXPECT_FLOAT_EQ(negative.data()[3], -4.0f);
}

TEST(TensorArithmetic, RejectsIncompatibleShapes) {
    auto a = Tensor<>::create(2, 3);
    auto b = Tensor<>::create(3, 2);
    expect_tensor_error([&] { static_cast<void>(a + b); }, TensorError::IncompatibleShapes);
}

TEST(TensorIndexing, PrefixViewsAliasRowMajorStorage) {
    auto tensor = Tensor<>::create(2, 3, 4);
    tensor[1].matrix().setConstant(2.0f);
    tensor[1, 2].vector()[3] = 7.0f;
    EXPECT_FLOAT_EQ((tensor[1, 2, 3].get()), 7.0f);
    EXPECT_FLOAT_EQ(tensor[1][2][3].get(), 7.0f);
    EXPECT_FLOAT_EQ(tensor.data()[23], 7.0f);
    EXPECT_FLOAT_EQ((tensor[0, 2, 3].get()), 0.0f);
    EXPECT_EQ(tensor[1].matrix().rows(), 3);
    EXPECT_EQ(tensor[1].matrix().cols(), 4);
    EXPECT_THROW(tensor[1].vector(), TensorException);
    EXPECT_THROW((tensor[1, 2].matrix()), TensorException);
    const auto& readonly = tensor;
    static_assert(std::is_same_v<decltype(readonly[1, 2, 3].get()), const float&>);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(readonly[1].matrix()(0, 0))>>);
}

TEST(TensorIndexing, EmptySlicesAndScalarTensors) {
    auto empty = Tensor<>::create(2, 0, 3);
    EXPECT_TRUE(empty[1].data().empty());
    EXPECT_EQ(empty[1].matrix().rows(), 0);
    EXPECT_THROW((empty[1, 0]), TensorException);
    auto scalar = Tensor<>::create(Tensor<>::Shape{});
    scalar[].get() = 3.0f;
    EXPECT_FLOAT_EQ(scalar[].get(), 3.0f);
    EXPECT_THROW(scalar[0], TensorException);
}

TEST(TensorLayout, TransposeAliasesStorageAndOwnsMetadata) {
    auto tensor = Tensor<int>::create(2, 3);
    for (std::size_t i = 0; i < tensor.numel(); ++i)
        tensor.data()[i] = static_cast<int>(i);
    auto transposed = tensor.transpose();
    EXPECT_EQ(transposed.shape(), (NanoChat::Shape{3, 2}));
    EXPECT_EQ(transposed.strides(), (NanoChat::Shape{1, 3}));
    EXPECT_EQ(tensor.shape(), (NanoChat::Shape{2, 3}));
    EXPECT_EQ(tensor.strides(), (NanoChat::Shape{3, 1}));
    EXPECT_FALSE(transposed.is_contiguous());
    EXPECT_EQ((transposed[2, 1].get()), 5);
    transposed[2, 1].get() = 42;
    EXPECT_EQ((tensor[1, 2].get()), 42);
    auto row = tensor.transpose()[1]; // Intermediate metadata has already been destroyed.
    EXPECT_EQ(row.vector()[1], 4);
    row.vector()[1] = 17;
    EXPECT_EQ((tensor[1, 1].get()), 17);
    EXPECT_EQ(transposed.matrix()(2, 1), 42);
    transposed.matrix()(0, 1) = 19;
    EXPECT_EQ((tensor[1, 0].get()), 19);
    EXPECT_TRUE(transposed.transpose().is_contiguous());
    EXPECT_THROW(transposed.data(), TensorException);
    EXPECT_THROW(static_cast<void>(tensor.transpose(0, 2)), TensorException);
}

TEST(TensorLayout, ThreeDimensionalStridedSlices) {
    auto tensor = Tensor<int>::create(2, 3, 4);
    for (std::size_t i = 0; i < tensor.numel(); ++i)
        tensor.data()[i] = static_cast<int>(i);
    auto transposed = tensor.transpose(0, 2);
    EXPECT_EQ(transposed.shape(), (NanoChat::Shape{4, 3, 2}));
    EXPECT_EQ((transposed[3, 2, 1].get()), 23);
    EXPECT_EQ(transposed[3].matrix()(2, 1), 23);
    auto vector = transposed[3, 2];
    EXPECT_EQ(vector.vector()[0], 11);
    EXPECT_EQ(vector.vector()[1], 23);
    EXPECT_THROW(static_cast<void>(tensor.transpose()), TensorException);
    auto copy = transposed.contiguous();
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 2; ++k)
                EXPECT_EQ((copy[i, j, k].get()), (tensor[k, j, i].get()));
}

TEST(TensorLayout, ReshapeBorrowsElementsAndOwnsMetadata) {
    auto tensor = Tensor<double>::create(2, 3, 4);
    auto reshaped = tensor.reshape(4, 6);
    EXPECT_EQ(reshaped.shape(), (NanoChat::Shape{4, 6}));
    EXPECT_EQ(reshaped.data().data(), tensor.data().data());
    EXPECT_TRUE(reshaped.is_contiguous());
    reshaped[3, 5].get() = 9.0;
    EXPECT_DOUBLE_EQ((tensor[1, 2, 3].get()), 9.0);
    auto chained = tensor.reshape(4, 6).transpose()[5];
    EXPECT_DOUBLE_EQ(chained.vector()[3], 9.0);
    EXPECT_EQ(tensor.shape(), (NanoChat::Shape{2, 3, 4}));
    EXPECT_THROW(static_cast<void>(tensor.reshape(5, 5)), TensorException);
    EXPECT_THROW(static_cast<void>(tensor.reshape(-1, 24)), TensorException);
    EXPECT_THROW(static_cast<void>(tensor.reshape(std::numeric_limits<std::size_t>::max(), 2)), TensorException);
    EXPECT_THROW(static_cast<void>(tensor.transpose(0, 2).reshape(24)), TensorException);
}

TEST(TensorLayout, ContiguousMaterializesLogicalOrder) {
    auto tensor = Tensor<int>::create(2, 3);
    for (std::size_t i = 0; i < tensor.numel(); ++i)
        tensor.data()[i] = static_cast<int>(i);
    auto copy = tensor.transpose().contiguous();
    EXPECT_EQ(copy.shape(), (NanoChat::Shape{3, 2}));
    EXPECT_EQ(copy.strides(), (NanoChat::Shape{2, 1}));
    EXPECT_TRUE(copy.is_contiguous());
    EXPECT_EQ(std::vector<int>(copy.data().begin(), copy.data().end()), (std::vector<int>{0, 3, 1, 4, 2, 5}));
    copy[0, 0].get() = 99;
    EXPECT_EQ((tensor[0, 0].get()), 0);
    auto rowCopy = tensor.transpose()[1].contiguous();
    EXPECT_EQ(rowCopy.shape(), (NanoChat::Shape{2}));
    EXPECT_EQ(rowCopy.data()[0], 1);
    EXPECT_EQ(rowCopy.data()[1], 4);
    Tensor<int> explicitCopy(tensor.transpose());
    EXPECT_EQ((explicitCopy[2, 1].get()), 5);
    auto ownedCopy = tensor.contiguous();
    EXPECT_NE(ownedCopy.data().data(), tensor.data().data());
    const auto* address = ownedCopy.data().data();
    auto moved = std::move(ownedCopy).contiguous();
    EXPECT_EQ(moved.data().data(), address);
    auto flat = copy.reshape(6);
    EXPECT_EQ(flat.shape(), (NanoChat::Shape{6}));
}

TEST(TensorLayout, ConstViewsAndIndependentOwnerLifetime) {
    const auto tensor = Tensor<double>::create(2, 3);
    static_assert(std::is_same_v<decltype(tensor.view()), NanoChat::TensorView<const double>>);
    static_assert(std::is_same_v<decltype(tensor.transpose()), NanoChat::TensorView<const double>>);
    static_assert(std::is_same_v<decltype(tensor.reshape(6)), NanoChat::TensorView<const double>>);
    static_assert(std::is_same_v<decltype(tensor.transpose().contiguous()), Tensor<double>>);
    static_assert(std::is_same_v<decltype(tensor.transpose()[0, 0].get()), const double&>);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(tensor.transpose().matrix()(0, 0))>>);
    auto copy = [] {
        auto source = Tensor<int>::create(2, 3);
        source[1, 2].get() = 8;
        return source.transpose().contiguous();
    }();
    EXPECT_EQ((copy[2, 1].get()), 8);
}

TEST(TensorLayout, SingletonEmptyAndScalarLayouts) {
    auto singleton = Tensor<int>::create(1, 3);
    EXPECT_TRUE(singleton.transpose().is_contiguous());
    EXPECT_EQ(singleton.transpose().reshape(3).numel(), 3u);
    auto empty = Tensor<int>::create(2, 0, 3);
    auto transposed = empty.transpose(0, 2);
    EXPECT_TRUE(transposed.is_contiguous());
    EXPECT_TRUE(transposed[1].data().empty());
    EXPECT_EQ(transposed.contiguous().shape(), (NanoChat::Shape{3, 0, 2}));
    EXPECT_EQ(transposed.reshape(0, 6).numel(), 0u);
    auto scalar = Tensor<int>::create(NanoChat::Shape{});
    scalar[].get() = 7;
    EXPECT_EQ(scalar.reshape(1).contiguous().data()[0], 7);
    EXPECT_EQ(scalar.reshape(1).reshape({}).get(), 7);
    auto scalarCopy = scalar.view().contiguous();
    EXPECT_EQ(scalarCopy[].get(), 7);
}

TEST(TensorLayout, RejectsInvalidBorrowedLayouts) {
    std::vector<float> storage(6);
    const NanoChat::Shape shape{2, 3};
    const NanoChat::Shape shortStrides{1};
    EXPECT_THROW((NanoChat::TensorView<float>(storage, shape, shortStrides)), TensorException);
    const NanoChat::Shape invalidStrides{4, 1};
    EXPECT_THROW((NanoChat::TensorView<float>(storage, shape, invalidStrides)), TensorException);
    const NanoChat::Shape broadcastStrides{0, 1};
    EXPECT_THROW((NanoChat::TensorView<float>(storage, shape, broadcastStrides)), TensorException);
    const NanoChat::Shape emptyShape{2, 0};
    const NanoChat::Shape emptyStrides{0, 1};
    auto empty = NanoChat::TensorView<float>(storage, emptyShape, emptyStrides);
    EXPECT_TRUE(empty.data().empty());
}

template<typename T>
concept CanBorrowTemporary = requires(T tensor) { std::move(tensor).view(); };
template<typename T>
concept CanTransposeTemporary = requires(T tensor) { std::move(tensor).transpose(); };
template<typename T>
concept CanReshapeTemporary = requires(T tensor) { std::move(tensor).reshape(1); };
static_assert(!CanBorrowTemporary<Tensor<float>>);
static_assert(!CanBorrowTemporary<const Tensor<float>>);
static_assert(!CanTransposeTemporary<Tensor<float>>);
static_assert(!CanTransposeTemporary<const Tensor<float>>);
static_assert(!CanReshapeTemporary<Tensor<float>>);
static_assert(!CanReshapeTemporary<const Tensor<float>>);

TEST(TensorErrors, OwnersAndViewsUseIdenticalErrorCodes) {
    auto tensor = Tensor<float>::create(2, 3);
    auto view = tensor.view();
    const auto check = [&](auto& value) {
        expect_tensor_error([&] { static_cast<void>(value[2, 0]); }, TensorError::IndexOutOfBounds);
        expect_tensor_error([&] { static_cast<void>(value[0, 0, 0]); }, TensorError::InvalidShape);
        expect_tensor_error([&] { static_cast<void>(value.transpose(0, 3)); }, TensorError::IndexOutOfBounds);
        expect_tensor_error([&] { static_cast<void>(value.reshape(5)); }, TensorError::InvalidShape);
        expect_tensor_error([&] { static_cast<void>(value.reshape(-1)); }, TensorError::InvalidDimension);
        expect_tensor_error([&] { static_cast<void>(value.reshape(std::numeric_limits<std::size_t>::max())); },
                            TensorError::SizeOverflow);
        expect_tensor_error([&] { static_cast<void>(value.get()); }, TensorError::InvalidShape);
        expect_tensor_error([&] { static_cast<void>(value.vector()); }, TensorError::InvalidShape);
        expect_tensor_error([&] { static_cast<void>(value.transpose().reshape(6)); }, TensorError::NonContiguous);
        auto other = Tensor<float>::create(6);
        expect_tensor_error([&] { static_cast<void>(value + other); }, TensorError::IncompatibleShapes);
        expect_tensor_error([&] { static_cast<void>(value + other.view()); }, TensorError::IncompatibleShapes);
    };
    check(tensor);
    check(view);
    expect_tensor_error([&] { static_cast<void>(view.transpose().data()); }, TensorError::NonContiguous);
}

TEST(TensorArithmetic, OwnersAndStridedViewsAreInterchangeable) {
    auto tensor = Tensor<double>::create(2, 2);
    tensor.matrix() << 1.0, 2.0, 3.0, 4.0;
    const auto& readonly = tensor;
    auto transposed = readonly.transpose();
    auto bothViews = transposed + transposed;
    auto ownerView = tensor + transposed;
    auto viewOwner = transposed + tensor;
    auto difference = tensor - transposed;
    auto product = transposed * tensor;
    auto negative = -transposed;
    EXPECT_DOUBLE_EQ((bothViews[0, 1].get()), 6.0);
    EXPECT_DOUBLE_EQ((ownerView[0, 1].get()), 5.0);
    EXPECT_DOUBLE_EQ((viewOwner[0, 1].get()), 5.0);
    EXPECT_DOUBLE_EQ((difference[0, 1].get()), -1.0);
    EXPECT_DOUBLE_EQ((product[0, 1].get()), 6.0);
    EXPECT_DOUBLE_EQ((negative[0, 1].get()), -3.0);
    EXPECT_DOUBLE_EQ((tensor[0, 1].get()), 2.0);
    static_assert(std::is_same_v<decltype(bothViews), Tensor<double>>);
    auto empty = Tensor<double>::create(2, 0);
    EXPECT_TRUE((empty + empty.view()).data().empty());
    auto scalar = Tensor<double>::create(NanoChat::Shape{});
    scalar.get() = 2.0;
    auto scalarSum = scalar.view() + scalar;
    EXPECT_DOUBLE_EQ(scalarSum.get(), 4.0);
}

TEST(TensorValidation, RequirementsComposeInOrder) {
    using NanoChat::TensorValidation::validate_input;
    const std::array values{2, 3, 5};
    int calls = 0;
    validate_input(values)
            .require(
                    [&](const auto& input) {
                        ++calls;
                        return std::all_of(input.begin(), input.end(), [](int value) { return value > 0; });
                    },
                    TensorError::InvalidDimension, "All dimensions must be positive")
            .require(
                    [&](const auto& input) {
                        ++calls;
                        return std::accumulate(input.begin(), input.end(), 0) == 10;
                    },
                    TensorError::InvalidShape, "Dimensions must sum to 10");
    EXPECT_EQ(calls, 2);
    EXPECT_EQ(values, (std::array{2, 3, 5}));
}

TEST(TensorValidation, FirstFailureStopsThePipeline) {
    using NanoChat::TensorValidation::validate_input;
    const std::array values{-1, 3, 5};
    bool laterCalled = false;
    try {
        validate_input(values)
                .require(
                        [](const auto& input) {
                            return std::all_of(input.begin(), input.end(), [](int value) { return value > 0; });
                        },
                        TensorError::InvalidDimension, "All dimensions must be positive")
                .require(
                        [&](const auto&) {
                            laterCalled = true;
                            return false;
                        },
                        TensorError::InvalidShape, "Dimensions must sum to 10");
        FAIL() << "Expected validation failure";
    } catch (const TensorException& error) {
        EXPECT_EQ(error.code(), TensorError::InvalidDimension);
        EXPECT_STREQ(error.what(), "All dimensions must be positive");
    }
    EXPECT_FALSE(laterCalled);
}

TEST(TensorValidation, LaterFailureHasItsOwnCodeAndMessage) {
    using NanoChat::TensorValidation::validate_input;
    const std::array values{1, 2, 3};
    try {
        validate_input(values)
                .require(
                        [](const auto& input) {
                            return std::all_of(input.begin(), input.end(), [](int value) { return value > 0; });
                        },
                        TensorError::InvalidDimension, "All dimensions must be positive")
                .require([](const auto& input) { return std::accumulate(input.begin(), input.end(), 0) == 10; },
                         TensorError::InvalidShape, "Dimensions must sum to 10");
        FAIL() << "Expected validation failure";
    } catch (const TensorException& error) {
        EXPECT_EQ(error.code(), TensorError::InvalidShape);
        EXPECT_STREQ(error.what(), "Dimensions must sum to 10");
    }
}

TEST(TensorValidation, BorrowsLvaluesAndOwnsTemporaryInputs) {
    using NanoChat::TensorValidation::validate_input;
    const auto tensor = Tensor<double>::create(2, 3);
    validate_input(tensor).require([&](const auto& input) { return &input == &tensor; }, TensorError::InvalidShape,
                                   "Input should be borrowed");
    auto check = validate_input(std::vector<int>{2, 3, 5});
    check.require([](const auto& input) { return std::accumulate(input.begin(), input.end(), 0) == 10; },
                  TensorError::InvalidShape, "Temporary input must remain alive");
    auto predicate = [state = std::make_unique<int>(10)](const auto& input) {
        return std::accumulate(input.begin(), input.end(), 0) == *state;
    };
    check.require(std::move(predicate), TensorError::InvalidShape, "Move-only predicates are supported");
    EXPECT_NO_THROW(NanoChat::TensorValidation::require(true, TensorError::InvalidShape, "Single check"));
}

TEST(TensorValidation, InvalidIndexFailsBeforeSliceAccess) {
    auto tensor = Tensor<int>::create(2, 3);
    expect_tensor_error([&] { static_cast<void>(tensor[-1, 100]); }, TensorError::IndexOutOfBounds);
    expect_tensor_error([&] { static_cast<void>(tensor[0, -1]); }, TensorError::IndexOutOfBounds);
    expect_tensor_error([&] { static_cast<void>(tensor[0, 0, -1]); }, TensorError::InvalidShape);
    expect_tensor_error([&] { Tensor<int>::create(NanoChat::Shape{0, std::numeric_limits<std::size_t>::max()}); },
                        TensorError::SizeOverflow);
}
