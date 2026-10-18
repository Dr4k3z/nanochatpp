#ifndef TENSOR_VALIDATION_H
#define TENSOR_VALIDATION_H

#include <Eigen/Core>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include "tensor_error.h"

// Define consistently across all translation units using these headers.
#ifndef NANOCHAT_ENABLE_VALIDATION
#define NANOCHAT_ENABLE_VALIDATION 1
#endif
#if NANOCHAT_ENABLE_VALIDATION != 0 && NANOCHAT_ENABLE_VALIDATION != 1
#error "NANOCHAT_ENABLE_VALIDATION must be 0 or 1"
#endif

namespace NanoChat::TensorValidation {
    inline constexpr bool enabled = NANOCHAT_ENABLE_VALIDATION != 0;

    inline void require(bool condition, TensorError code, std::string_view message) {
        if constexpr (enabled) {
            if (!condition)
                throw TensorException(code, std::string(message));
        }
    }

    // Use a predicate when evaluating the condition itself has a cost or side effect.
    template<typename Predicate>
        requires std::predicate<Predicate>
    inline void require(Predicate&& predicate, TensorError code, std::string_view message) {
        if constexpr (enabled)
            require(std::invoke(std::forward<Predicate>(predicate)), code, message);
    }

    // Lvalues are borrowed; rvalues are owned. Predicates are evaluated immediately,
    // in order, and never run after a failed requirement. No type erasure or registry.
    template<typename Input>
    class InputValidation {
    public:
        explicit InputValidation(Input input) : m_input(std::forward<Input>(input)) {}

        template<typename Predicate>
            requires std::predicate<Predicate, const std::remove_reference_t<Input>&>
        InputValidation& require(Predicate&& predicate, TensorError code, std::string_view message) {
            if constexpr (enabled)
                TensorValidation::require(std::invoke(std::forward<Predicate>(predicate), std::as_const(m_input)), code,
                                          message);
            return *this;
        }

    private:
        Input m_input;
    };

    template<typename Input>
    [[nodiscard]] auto validate_input(Input&& input) {
        return InputValidation<Input>(std::forward<Input>(input));
    }

    // Nontrivial shape validation shared by creation, reshape and borrowed layouts.
    inline std::size_t checked_numel(std::span<const std::size_t> shape) {
        if constexpr (!enabled) {
            std::size_t count = 1;
            for (auto extent: shape)
                count *= extent;
            return count;
        }
        const auto limit = static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max());
        std::size_t count = 1;
        std::size_t suffix = 1;
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            const auto extent = shape[axis];
            const auto reverseExtent = shape[shape.size() - 1 - axis];
            require(extent <= limit && (extent == 0 || count <= limit / extent), TensorError::SizeOverflow,
                    "Tensor shape exceeds Eigen index range");
            require(reverseExtent == 0 || suffix <= limit / reverseExtent, TensorError::SizeOverflow,
                    "Tensor strides exceed Eigen index range");
            count *= extent;
            suffix *= reverseExtent;
        }
        return count;
    }

    inline std::size_t layout(std::size_t storageSize, std::span<const std::size_t> shape,
                              std::span<const std::size_t> strides) {
        if constexpr (enabled) {
            validate_input(shape).require([&](auto value) { return value.size() == strides.size(); },
                                          TensorError::InvalidShape,
                                          "TensorView: shape and strides must have equal rank");
            const auto count = checked_numel(shape);
            validate_input(strides).require(
                    [](auto values) {
                        return std::all_of(values.begin(), values.end(), [](auto stride) {
                            return stride <= static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max());
                        });
                    },
                    TensorError::SizeOverflow, "TensorView: stride exceeds Eigen index range");
            if (count == 0)
                return count;
            require(storageSize != 0, TensorError::InvalidShape, "TensorView: backing storage is empty");
            std::size_t last = 0;
            for (std::size_t axis = 0; axis < shape.size(); ++axis) {
                require(shape[axis] <= 1 || strides[axis] != 0, TensorError::InvalidShape,
                        "TensorView: zero strides on non-singleton axes are unsupported");
                const auto steps = shape[axis] - 1;
                require(steps == 0 || strides[axis] <= (storageSize - 1 - last) / steps, TensorError::InvalidShape,
                        "TensorView: layout exceeds backing storage");
                last += steps * strides[axis];
            }
            return count;
        } else {
            return checked_numel(shape);
        }
    }
} // namespace NanoChat::TensorValidation
#endif
