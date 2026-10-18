#ifndef TENSOR_ERROR_H
#define TENSOR_ERROR_H

#include <stdexcept>
#include <string>
#include <utility>

namespace NanoChat {
    enum class TensorError {
        InvalidDimension,
        InvalidShape,
        IncompatibleShapes,
        IndexOutOfBounds,
        SizeOverflow,
        NonContiguous
    };

    class TensorException final : public std::runtime_error {
    public:
        TensorException(TensorError code, const std::string& message) : std::runtime_error(message), m_code(code) {}
        [[nodiscard]] TensorError code() const noexcept { return m_code; }

    private:
        TensorError m_code;
    };
} // namespace NanoChat
#endif
