#ifndef TENSOR_H
#define TENSOR_H

#include "tensor_view.h"

#include <Eigen/Dense>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace NanoChat {

    template<typename Value = float>
    class Tensor {
    public:
        using Scalar = Value;
        using Shape = NanoChat::Shape;
        using Storage = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

        /* Factory Methods */
        static Tensor create(const Shape& shape) { return create_from_shape(shape); }
        template<std::integral... Dims>
            requires(sizeof...(Dims) > 0)
        static Tensor create(Dims... dims) {
            TensorValidation::require([&] { return (std::in_range<std::size_t>(dims) && ...); },
                                      TensorError::InvalidDimension,
                                      "Tensor dimensions must be non-negative and representable as size_t");
            return create(Shape{static_cast<std::size_t>(dims)...});
        }

        /* Public getters */
        [[nodiscard]] const Shape& shape() const noexcept { return m_shape; }
        [[nodiscard]] const Shape& strides() const noexcept { return m_strides; }
        [[nodiscard]] std::size_t ndim() const noexcept { return m_shape.size(); }
        [[nodiscard]] std::size_t numel() const noexcept { return static_cast<std::size_t>(m_data.size()); }
        std::span<Scalar> data() noexcept { return {m_data.data(), numel()}; }
        std::span<const Scalar> data() const noexcept { return {m_data.data(), numel()}; }

        Scalar& get() & { return view().get(); }
        const Scalar& get() const& { return view().get(); }
        Scalar& get() && = delete;
        const Scalar& get() const&& = delete;
        auto matrix() & { return view().matrix(); }
        auto matrix() const& { return view().matrix(); }
        auto matrix() && = delete;
        auto matrix() const&& = delete;
        auto vector() & { return view().vector(); }
        auto vector() const& { return view().vector(); }
        auto vector() && = delete;
        auto vector() const&& = delete;

        /* Non-owning views on the data */
        template<std::integral... Indices>
        TensorView<Scalar> operator[](Indices... indices) & {
            return view()[indices...];
        }
        template<std::integral... Indices>
        TensorView<const Scalar> operator[](Indices... indices) const& {
            return view()[indices...];
        }
        template<std::integral... Indices>
        TensorView<Scalar> operator[](Indices...) && = delete;
        template<std::integral... Indices>
        TensorView<const Scalar> operator[](Indices...) const&& = delete;

        // TensorView implements arithmetic for both contiguous and strided operands.
        template<typename Other>
        Tensor sum(const Other& other) const {
            return view().sum(other);
        }
        template<typename Other>
        Tensor product(const Other& other) const {
            return view().product(other);
        }
        template<typename Other>
        Tensor operator+(const Other& other) const {
            return sum(other);
        }
        template<typename Other>
        Tensor operator-(const Other& other) const {
            return view() - other;
        }
        template<typename Other>
        Tensor operator*(const Other& other) const {
            return product(other);
        }
        Tensor operator-() const { return -view(); }

        // Layout operations share TensorView's implementation and never copy elements.
        [[nodiscard]] TensorView<Scalar> view() & { return {data(), m_shape, m_strides}; }
        [[nodiscard]] TensorView<const Scalar> view() const& { return {data(), m_shape, m_strides}; }
        TensorView<Scalar> view() && = delete;
        TensorView<const Scalar> view() const&& = delete;

        [[nodiscard]] TensorView<Scalar> transpose(std::size_t axis1, std::size_t axis2) & {
            return view().transpose(axis1, axis2);
        }
        [[nodiscard]] TensorView<const Scalar> transpose(std::size_t axis1, std::size_t axis2) const& {
            return view().transpose(axis1, axis2);
        }
        TensorView<Scalar> transpose(std::size_t, std::size_t) && = delete;
        TensorView<const Scalar> transpose(std::size_t, std::size_t) const&& = delete;
        [[nodiscard]] TensorView<Scalar> transpose() & { return view().transpose(); }
        [[nodiscard]] TensorView<const Scalar> transpose() const& { return view().transpose(); }
        TensorView<Scalar> transpose() && = delete;
        TensorView<const Scalar> transpose() const&& = delete;

        [[nodiscard]] TensorView<Scalar> reshape(const Shape& shape) & { return view().reshape(shape); }
        [[nodiscard]] TensorView<const Scalar> reshape(const Shape& shape) const& { return view().reshape(shape); }
        TensorView<Scalar> reshape(const Shape&) && = delete;
        TensorView<const Scalar> reshape(const Shape&) const&& = delete;
        template<std::integral... Dims>
            requires(sizeof...(Dims) > 0)
        [[nodiscard]] TensorView<Scalar> reshape(Dims... dims) & {
            return view().reshape(dims...);
        }
        template<std::integral... Dims>
            requires(sizeof...(Dims) > 0)
        [[nodiscard]] TensorView<const Scalar> reshape(Dims... dims) const& {
            return view().reshape(dims...);
        }
        template<std::integral... Dims>
        TensorView<Scalar> reshape(Dims...) && = delete;
        template<std::integral... Dims>
        TensorView<const Scalar> reshape(Dims...) const&& = delete;

        [[nodiscard]] bool is_contiguous() const noexcept { return true; }
        [[nodiscard]] Tensor contiguous() const& { return *this; }
        [[nodiscard]] Tensor contiguous() && { return std::move(*this); }

        // Explicit materialization, analogous to constructing string from string_view.
        template<typename Element>
            requires std::same_as<std::remove_const_t<Element>, Scalar>
        explicit Tensor(const TensorView<Element>& source) :
            m_shape(source.shape()), m_strides(TensorLayout::row_major_strides(m_shape)),
            m_data(static_cast<Eigen::Index>(source.numel())) {
            if (source.numel() == 0)
                return;
            if (source.is_contiguous()) {
                const auto elements = source.data();
                std::copy(elements.begin(), elements.end(), m_data.data());
                return;
            }
            for (std::size_t linear = 0; linear < source.numel(); ++linear) {
                auto remainder = linear;
                std::size_t offset = 0;
                for (std::size_t axis = source.ndim(); axis-- > 0;) {
                    offset += (remainder % m_shape[axis]) * source.m_strides[axis];
                    remainder /= m_shape[axis];
                }
                m_data[static_cast<Eigen::Index>(linear)] = source.m_data[offset];
            }
        }

    private:
        Shape m_shape;
        Shape m_strides;
        Storage m_data;

        Tensor(Shape shape, Shape strides, Storage data) :
            m_shape(std::move(shape)), m_strides(std::move(strides)), m_data(std::move(data)) {}

        static Tensor create_from_shape(Shape shape) {
            const auto count = TensorValidation::checked_numel(shape);
            auto strides = TensorLayout::row_major_strides(shape);
            Storage data = Storage::Zero(static_cast<Eigen::Index>(count));
            return Tensor(std::move(shape), std::move(strides), std::move(data));
        }
    };
    template<typename Element>
    template<typename Other, typename Operation>
    Tensor<typename TensorView<Element>::Scalar> TensorView<Element>::binary_op(const Other& other,
                                                                                Operation operation) const {
        static_assert(std::same_as<typename Other::Scalar, Scalar>, "Tensor operands must have matching scalar types");
        TensorValidation::validate_input(other).require(
                [&](const auto& value) {
                    return shape().size() == value.shape().size() &&
                           std::equal(shape().begin(), shape().end(), value.shape().begin());
                },
                TensorError::IncompatibleShapes, "Tensor arithmetic: operand shapes must match");
        const auto right = other.view();
        Tensor<Scalar> result(*this);
        auto output = result.data();
        if (right.is_contiguous()) {
            const auto input = right.data();
            std::transform(output.begin(), output.end(), input.begin(), output.begin(), operation);
        } else {
            for (std::size_t linear = 0; linear < numel(); ++linear) {
                auto remainder = linear;
                std::size_t offset = 0;
                for (std::size_t axis = right.ndim(); axis-- > 0;) {
                    offset += (remainder % right.m_shape[axis]) * right.m_strides[axis];
                    remainder /= right.m_shape[axis];
                }
                output[linear] = operation(output[linear], right.m_data[offset]);
            }
        }
        return result;
    }

    template<typename Element>
    template<typename Other>
    Tensor<typename TensorView<Element>::Scalar> TensorView<Element>::sum(const Other& other) const {
        return binary_op(other, [](Scalar a, Scalar b) { return a + b; });
    }
    template<typename Element>
    template<typename Other>
    Tensor<typename TensorView<Element>::Scalar> TensorView<Element>::product(const Other& other) const {
        return binary_op(other, [](Scalar a, Scalar b) { return a * b; });
    }
    template<typename Element>
    template<typename Other>
    Tensor<typename TensorView<Element>::Scalar> TensorView<Element>::operator-(const Other& other) const {
        return binary_op(other, [](Scalar a, Scalar b) { return a - b; });
    }
    template<typename Element>
    Tensor<typename TensorView<Element>::Scalar> TensorView<Element>::operator-() const {
        Tensor<Scalar> result(*this);
        for (auto& element: result.data())
            element = -element;
        return result;
    }
    template<typename Element>
    Tensor<typename TensorView<Element>::Scalar> TensorView<Element>::contiguous() const {
        return Tensor<Scalar>(*this);
    }
} // namespace NanoChat
#endif // TENSOR_H
