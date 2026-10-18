#ifndef TENSOR_NEW_H
#define TENSOR_NEW_H

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "errors.h"

namespace NanoChat {
    template<typename Value = float>
    class Tensor {
    public:
        using Shape = std::vector<std::size_t>;
        using Storage = Eigen::Matrix<Value, Eigen::Dynamic, 1>;

        /* Public Constructors */
        explicit Tensor(const Shape& shape) : Tensor(create_tensor_from_shape(shape)) {}

        template<std::integral... Dims>
            requires(sizeof...(Dims) > 0)
        explicit Tensor(Dims... dims) : Tensor(from_dimensions(dims...)) {}

        /* Data Access APIs and Operators */
        template<std::integral... Indices>
            requires(sizeof...(Indices) > 0)
        [[nodiscard]] Tensor operator[](Indices... indices) const {
            constexpr auto n = sizeof...(Indices);
            TensorValidation::require_invariant(n <= get_ndim(), TensorError::IndexOutOfBounds,
                                                "Too many indices for tensor rank");
            TensorValidation::require_invariant((std::in_range<std::size_t>(indices) && ...),
                                                TensorError::IndexOutOfBounds,
                                                "Tensor index cannot be negative or exceed size_t");

            const std::array<std::size_t, n> idx{static_cast<std::size_t>(indices)...};
            auto offset = m_offset;

            for (std::size_t i = 0; i < n; ++i) {
                TensorValidation::require_invariant(idx[i] < m_shape[i], TensorError::IndexOutOfBounds,
                                                    "Tensor index exceeds dimension");
                offset += idx[i] * m_strides[i];
            }

            Shape newShape(m_shape.begin() + n, m_shape.end());
            Shape newStrides(m_strides.begin() + n, m_strides.end());
            return Tensor(m_data, std::move(newShape), std::move(newStrides), offset);
        }

        // NOTE are these needed?
        Value& get() & { return scalar_reference(); }
        const Value& get() const& { return scalar_reference(); }
        Value& get() && { return scalar_reference(); }
        const Value& get() const&& { return scalar_reference(); }

        /* Metadata APIs */
        [[nodiscard]] const Shape& get_shape() const noexcept { return m_shape; }
        [[nodiscard]] const Shape& get_strides() const noexcept { return m_strides; }
        [[nodiscard]] std::size_t get_ndim() const noexcept { return m_shape.size(); }
        [[nodiscard]] std::size_t get_numel() const noexcept { return m_numel; }
        [[nodiscard]] bool is_contiguous() const noexcept {
            if (m_numel == 0)
                return true;
            std::size_t expected = 1;
            for (std::size_t axis = get_ndim(); axis-- > 0;) {
                if (m_shape[axis] > 1 && m_strides[axis] != expected)
                    return false;
                if (m_shape[axis] != 0 && expected > std::numeric_limits<std::size_t>::max() / m_shape[axis])
                    return false;
                expected *= m_shape[axis];
            }
            return true;
        }

        /* Metadata Transformation Methods */
        [[nodiscard]] Tensor transpose(std::size_t axis_1, std::size_t axis_2) const {
            TensorValidation::require_invariant(axis_1 < get_ndim() && axis_2 < get_ndim(),
                                                TensorError::IndexOutOfBounds, "Transpose axes outside rank");

            auto result = *this;
            std::swap(result.m_shape[axis_1], result.m_shape[axis_2]);
            std::swap(result.m_strides[axis_1], result.m_strides[axis_2]);
            return result;
        }
        [[nodiscard]] Tensor permute(std::span<const std::size_t> axes) const {
            TensorValidation::require_invariant(axes.size() == get_ndim(), TensorError::InvalidShape,
                                                "Permutation must include every axis");

            std::vector<bool> seen(get_ndim(), false);
            Shape shape(get_ndim()), strides(get_ndim());
            for (std::size_t i = 0; i < axes.size(); ++i) {
                const auto axis = axes[i];
                TensorValidation::require_invariant(axis < get_ndim(), TensorError::IndexOutOfBounds,
                                                    "Permutation axis outside rank");
                TensorValidation::require_invariant(!seen[axis], TensorError::InvalidShape,
                                                    "Duplicate permutation axis");
                seen[axis] = true;
                shape[i] = m_shape[axis];
                strides[i] = m_strides[axis];
            }
            return Tensor(m_data, std::move(shape), std::move(strides), m_offset);
        }
        [[nodiscard]] Tensor reshape(const Shape& shape) const {
            TensorValidation::require_invariant(checked_numel(shape) == m_numel, TensorError::InvalidShape,
                                                "Reshape must preserve the number of elements");
            if (!is_contiguous())
                return make_contiguous().reshape(shape);
            return Tensor(m_data, shape, row_major_strides(shape), m_offset);
        }
        [[nodiscard]] Tensor make_contiguous() const {
            if (is_contiguous())
                return *this; // shared-storage view, not a copy
            return clone();
        }

        /* Data Ownership Transformations */
        [[nodiscard]] Tensor clone() const {
            Tensor result(m_shape);
            for (std::size_t i = 0; i < m_numel; ++i)
                (*result.m_data)[static_cast<Eigen::Index>(i)] = element_at_linear(i);
            return result;
        }

        /* Math Operations */
        [[nodiscard]] Tensor sum(const Tensor& other) const {
            return elementwise(other, [](const Value& a, const Value& b) { return a + b; });
        }
        [[nodiscard]] Tensor product(const Tensor& other) const {
            return elementwise(other, [](const Value& a, const Value& b) { return a * b; });
        }

        /* Math Overloaded Operators */
        [[nodiscard]] Tensor operator+(const Tensor& other) const { return sum(other); }
        [[nodiscard]] Tensor operator*(const Tensor& other) const { return product(other); }

    private:
        Shape m_shape;
        Shape m_strides;
        std::shared_ptr<Storage> m_data;
        std::size_t m_offset{0};
        std::size_t m_numel{0};

        /* Private Constructors */
        Tensor(std::shared_ptr<Storage> data, Shape shape, Shape strides, std::size_t offset) :
            m_shape(std::move(shape)), m_strides(std::move(strides)), m_data(std::move(data)), m_offset(offset),
            m_numel(checked_numel(m_shape)) {
            validate_layout();
        }

        template<std::integral... Dims>
        static Tensor from_dimensions(Dims... dims) {
            TensorValidation::require_invariant((std::in_range<std::size_t>(dims) && ...),
                                                TensorError::InvalidDimension, "Invalid tensor dimension");
            return create_tensor_from_shape(Shape{static_cast<std::size_t>(dims)...});
        }

        static Shape row_major_strides(std::span<const std::size_t> shape) {
            // TODO: constexpr validation
            Shape strides(shape.size());
            std::size_t stride = 1;
            for (std::size_t axis = shape.size(); axis-- > 0;) {
                strides[axis] = stride;
                TensorValidation::require_invariant(
                        shape[axis] == 0 || stride <= std::numeric_limits<std::size_t>::max() / shape[axis],
                        TensorError::SizeOverflow, "Tensor stride overflow");
                stride *= shape[axis];
            }
            return strides;
        }

        static std::size_t checked_numel(std::span<const std::size_t> shape) {
            // constexpr validation
            /* if constexpr (!enabled) {
                std::size_t count = 1;
                for (auto extent: shape)
                    count *= extent;
                return count;
            } */
            // const auto limit = static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max());
            std::size_t count = 1;
            std::size_t suffix = 1;
            for (std::size_t axis = 0; axis < shape.size(); ++axis) {
                const auto extent = shape[axis];
                const auto reverseExtent = shape[shape.size() - 1 - axis];

                /*require(extent <= limit && (extent == 0 || count <= limit / extent), TensorError::SizeOverflow,
                        "Tensor shape exceeds Eigen index range");
                require(reverseExtent == 0 || suffix <= limit / reverseExtent, TensorError::SizeOverflow,
                        "Tensor strides exceed Eigen index range");
                */
                TensorValidation::require_invariant(extent == 0 ||
                                                            count <= std::numeric_limits<std::size_t>::max() / extent,
                                                    TensorError::SizeOverflow, "Tensor element count overflow");
                TensorValidation::require_invariant(
                        reverseExtent == 0 || suffix <= std::numeric_limits<std::size_t>::max() / reverseExtent,
                        TensorError::SizeOverflow, "Tensor stride overflow");
                count *= extent;
                suffix *= reverseExtent;
            }
            TensorValidation::require_invariant(
                    count <= static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()),
                    TensorError::SizeOverflow, "Tensor exceeds Eigen index range");
            return count;
        }

        static Tensor create_tensor_from_shape(Shape shape) {
            // TODO: constexpr validation
            const auto count = checked_numel(shape);
            auto strides = row_major_strides(shape);
            auto data = std::make_shared<Storage>(Storage::Zero(static_cast<Eigen::Index>(count)));
            return Tensor(std::move(data), std::move(shape), std::move(strides), 0);
        }

        void validate_layout() const {
            TensorValidation::require_invariant(m_data != nullptr && m_shape.size() == m_strides.size(),
                                                TensorError::InvalidShape, "Invalid tensor storage or layout");
            const auto capacity = static_cast<std::size_t>(m_data->size());
            TensorValidation::require_invariant(m_offset <= capacity, TensorError::IndexOutOfBounds,
                                                "Tensor offset exceeds storage");
            if (m_numel == 0)
                return;
            std::size_t maxIndex = m_offset;
            for (std::size_t i = 0; i < m_shape.size(); ++i) {
                if (m_shape[i] == 0)
                    return;
                const auto steps = m_shape[i] - 1;
                TensorValidation::require_invariant(maxIndex < capacity, TensorError::IndexOutOfBounds,
                                                    "Tensor view exceeds storage");
                TensorValidation::require_invariant(steps == 0 || m_strides[i] <= (capacity - 1 - maxIndex) / steps,
                                                    TensorError::IndexOutOfBounds, "Tensor view exceeds storage");
                maxIndex += steps * m_strides[i];
            }
            TensorValidation::require_invariant(maxIndex < capacity, TensorError::IndexOutOfBounds,
                                                "Tensor view exceeds storage");
        }

        Value& scalar_reference() {
            TensorValidation::require_invariant(get_ndim() == 0, TensorError::InvalidShape,
                                                "get() requires a rank-zero tensor");
            return (*m_data)[static_cast<Eigen::Index>(m_offset)];
        }

        const Value& scalar_reference() const {
            TensorValidation::require_invariant(get_ndim() == 0, TensorError::InvalidShape,
                                                "get() requires a rank-zero tensor");
            return (*m_data)[static_cast<Eigen::Index>(m_offset)];
        }

        const Value& element_at_linear(std::size_t linear) const {
            std::size_t offset = m_offset;
            for (std::size_t axis = get_ndim(); axis-- > 0;) {
                offset += (linear % m_shape[axis]) * m_strides[axis];
                linear /= m_shape[axis];
            }
            return (*m_data)[static_cast<Eigen::Index>(offset)];
        }

        template<typename Operation>
        Tensor elementwise(const Tensor& other, Operation op) const {
            TensorValidation::require_invariant(m_shape == other.m_shape, TensorError::IncompatibleShapes,
                                                "Tensor operands must have identical shapes");
            Tensor result(m_shape);
            for (std::size_t i = 0; i < m_numel; ++i)
                (*result.m_data)[static_cast<Eigen::Index>(i)] = op(element_at_linear(i), other.element_at_linear(i));
            return result;
        }
    };
} // namespace NanoChat

#endif // TENSOR_NEW_H
