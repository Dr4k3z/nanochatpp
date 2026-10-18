#ifndef TENSOR_VIEW_H
#define TENSOR_VIEW_H

#include "tensor_layout.h"

#include <Eigen/Core>

#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace NanoChat {
    template<typename Scalar>
    class Tensor;

    template<typename Element>
    class TensorView {
        // TensorView doesn't hold the data, which is delegated to Tensor,
        // but only the required metadata. It borrows elements from the supplied backing storage.
        // Methods which require owning the data are defined here but implemented in the Tensor
        // class, so that the API experience is linear and the user doesn't need to go back
        // and forth between tensors and views.

    public:
        using Scalar = std::remove_const_t<Element>;
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using MatrixMap = Eigen::Map<std::conditional_t<std::is_const_v<Element>, const Matrix, Matrix>,
                                     Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
        using VectorMap = Eigen::Map<std::conditional_t<std::is_const_v<Element>, const Vector, Vector>,
                                     Eigen::Unaligned, Eigen::InnerStride<Eigen::Dynamic>>;

        /* Constructor */
        TensorView(std::span<Element> data, std::span<const std::size_t> shape, std::span<const std::size_t> strides) :
            m_data(data), m_shape(shape.begin(), shape.end()), m_strides(strides.begin(), strides.end()) {
            m_numel = TensorValidation::layout(m_data.size(), m_shape, m_strides);
        }

        [[nodiscard]] TensorView view() const { return *this; }

        /* Public getters */
        [[nodiscard]] const Shape& shape() const noexcept { return m_shape; }
        [[nodiscard]] const Shape& strides() const noexcept { return m_strides; }
        [[nodiscard]] std::size_t ndim() const noexcept { return m_shape.size(); }
        [[nodiscard]] std::size_t numel() const noexcept { return m_numel; }
        [[nodiscard]] bool is_contiguous() const noexcept {
            if (numel() == 0)
                return true;
            std::size_t expected = 1;
            for (std::size_t axis = ndim(); axis-- > 0;) {
                if (m_shape[axis] > 1 && m_strides[axis] != expected)
                    return false;
                expected *= m_shape[axis];
            }
            return true;
        }
        std::span<Element> data() const {
            TensorValidation::require([&] { return is_contiguous(); }, TensorError::NonContiguous,
                                      "TensorView::data() requires contiguous storage; call contiguous() first");
            return m_data.first(numel());
        }

        /* Data access overloaded operator */
        template<std::integral... Indices>
        TensorView operator[](Indices... indices) const {
            TensorValidation::require([&] { return sizeof...(Indices) <= ndim(); }, TensorError::InvalidShape,
                                      "Tensor indexing: too many indices for tensor rank");
            if constexpr (TensorValidation::enabled) {
                // Validate all indices before calculating offsets or constructing a slice.
                std::size_t checkedAxis = 0;
                const auto validateIndex = [&](auto index) {
                    const auto axis = checkedAxis++;
                    TensorValidation::validate_input(index)
                            .require([](auto value) { return std::in_range<std::size_t>(value); },
                                     TensorError::IndexOutOfBounds, "Tensor indexing: index must be non-negative")
                            .require([&](auto value) { return static_cast<std::size_t>(value) < m_shape[axis]; },
                                     TensorError::IndexOutOfBounds, "Tensor indexing: index exceeds axis extent");
                };
                (validateIndex(indices), ...);
            }
            std::size_t axis = 0;
            std::size_t pos = 0;
            const auto consume = [&](auto index) {
                const auto checkedIndex = static_cast<std::size_t>(index);
                if (numel() != 0)
                    pos += checkedIndex * m_strides[axis];
                ++axis;
            };
            (consume(indices), ...);
            const auto remainingShape = std::span<const std::size_t>(m_shape).subspan(sizeof...(Indices));
            std::size_t count = 1;
            for (const auto extent: remainingShape)
                count *= extent;
            auto slice = count == 0 ? m_data.first(0) : m_data.subspan(pos);
            return TensorView(slice, remainingShape,
                              std::span<const std::size_t>(m_strides).subspan(sizeof...(Indices)));
        }

        /* Data access methods */
        Element& get() const {
            TensorValidation::require([&] { return ndim() == 0; }, TensorError::InvalidShape,
                                      "Tensor scalar access via get() requires rank 0");
            return m_data[0];
        }
        MatrixMap matrix() const {
            TensorValidation::require([&] { return ndim() == 2; }, TensorError::InvalidShape,
                                      "Tensor matrix() requires rank 2");
            return MatrixMap(m_data.data(), static_cast<Eigen::Index>(m_shape[0]),
                             static_cast<Eigen::Index>(m_shape[1]),
                             Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(static_cast<Eigen::Index>(m_strides[0]),
                                                                           static_cast<Eigen::Index>(m_strides[1])));
        }
        VectorMap vector() const {
            TensorValidation::require([&] { return ndim() == 1; }, TensorError::InvalidShape,
                                      "Tensor vector() requires rank 1");
            return VectorMap(m_data.data(), static_cast<Eigen::Index>(m_shape[0]),
                             Eigen::InnerStride<Eigen::Dynamic>(static_cast<Eigen::Index>(m_strides[0])));
        }

        /* Public API */
        [[nodiscard]] TensorView transpose(std::size_t axis1, std::size_t axis2) && {
            TensorValidation::require([&] { return axis1 < ndim() && axis2 < ndim(); }, TensorError::IndexOutOfBounds,
                                      "Tensor transpose: axis outside rank");
            std::swap(m_shape[axis1], m_shape[axis2]);
            std::swap(m_strides[axis1], m_strides[axis2]);
            return std::move(*this);
        }
        [[nodiscard]] TensorView transpose(std::size_t axis1, std::size_t axis2) const& {
            return TensorView(*this).transpose(axis1, axis2);
        }
        [[nodiscard]] TensorView transpose() const& {
            TensorValidation::require([&] { return ndim() == 2; }, TensorError::InvalidShape,
                                      "Tensor transpose() without axes requires rank 2");
            return transpose(0, 1);
        }
        [[nodiscard]] TensorView transpose() && {
            TensorValidation::require([&] { return ndim() == 2; }, TensorError::InvalidShape,
                                      "Tensor transpose() without axes requires rank 2");
            return std::move(*this).transpose(0, 1);
        }
        [[nodiscard]] TensorView reshape(const Shape& shape) const {
            TensorValidation::validate_input(shape)
                    .require([&](const auto& value) { return TensorValidation::checked_numel(value) == numel(); },
                             TensorError::InvalidShape, "Tensor reshape: element count must stay unchanged")
                    .require([&](const auto&) { return is_contiguous(); }, TensorError::NonContiguous,
                             "Tensor reshape requires contiguous storage; call contiguous() first");
            const auto strides = TensorLayout::row_major_strides(shape);
            return TensorView(m_data, shape, strides);
        }
        template<std::integral... Dims>
            requires(sizeof...(Dims) > 0)
        [[nodiscard]] TensorView reshape(Dims... dims) const {
            TensorValidation::require([&] { return (std::in_range<std::size_t>(dims) && ...); },
                                      TensorError::InvalidDimension,
                                      "Tensor reshape: dimensions must be non-negative and representable as size_t");
            return reshape(Shape{static_cast<std::size_t>(dims)...});
        }

        // The same arithmetic API works on owners and views, including strided operands.
        template<typename Other>
        Tensor<Scalar> sum(const Other& other) const;
        template<typename Other>
        Tensor<Scalar> product(const Other& other) const;
        template<typename Other>
        Tensor<Scalar> operator+(const Other& other) const {
            return sum(other);
        }
        template<typename Other>
        Tensor<Scalar> operator-(const Other& other) const;
        template<typename Other>
        Tensor<Scalar> operator*(const Other& other) const {
            return product(other);
        }
        Tensor<Scalar> operator-() const;

        /* Owning methods defined here but implemented inside of Tensor */
        [[nodiscard]] Tensor<Scalar> contiguous() const;

    private:
        template<typename>
        friend class Tensor;
        template<typename>
        friend class TensorView;
        template<typename Other, typename Operation>
        Tensor<Scalar> binary_op(const Other& other, Operation operation) const;
        std::span<Element> m_data;
        Shape m_shape;
        Shape m_strides;
        std::size_t m_numel = 0;
    };
} // namespace NanoChat

#endif // TENSOR_VIEW_H
