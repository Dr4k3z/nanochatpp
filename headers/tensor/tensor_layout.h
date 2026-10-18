#ifndef TENSOR_LAYOUT_H
#define TENSOR_LAYOUT_H

#include <span>
#include <vector>
#include "tensor_validation.h"

namespace NanoChat {
    using Shape = std::vector<std::size_t>;
    namespace TensorLayout {
        inline Shape row_major_strides(std::span<const std::size_t> shape) {
            if constexpr (TensorValidation::enabled)
                TensorValidation::checked_numel(shape);
            Shape strides(shape.size());
            std::size_t stride = 1;
            for (std::size_t axis = shape.size(); axis-- > 0;) {
                strides[axis] = stride;
                stride *= shape[axis];
            }
            return strides;
        }
    } // namespace TensorLayout
} // namespace NanoChat
#endif
