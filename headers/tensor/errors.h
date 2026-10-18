#ifndef TENSOR_ERROR_H
#define TENSOR_ERROR_H

#include <concepts>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#ifndef NANOCHAT_ENABLE_VALIDATION
#define NANOCHAT_ENABLE_VALIDATION 1
#endif
#if NANOCHAT_ENABLE_VALIDATION != 0 && NANOCHAT_ENABLE_VALIDATION != 1
#error "NANOCHAT_ENABLE_VALIDATION must be 0 or 1"
#endif

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

    namespace TensorValidation {
        inline constexpr bool enabled = NANOCHAT_ENABLE_VALIDATION != 0;

        inline void require(bool condition, TensorError code, std::string_view message = "") {
            if constexpr (enabled)
                if (!condition)
                    throw TensorException(code, std::string(message));
        }

        template<typename Predicate>
            requires std::predicate<Predicate>
        void require(Predicate&& predicate, TensorError code, std::string_view message = "") {
            if constexpr (enabled)
                require(std::invoke(std::forward<Predicate>(predicate)), code, message);
        }

        inline void require_invariant(bool condition, TensorError code, std::string_view message = "") {
            if (!condition)
                throw TensorException(code, std::string(message));
        }

        template<typename Predicate>
            requires std::predicate<Predicate>
        void require_invariant(Predicate&& predicate, TensorError code, std::string_view message = "") {
            require_invariant(std::invoke(std::forward<Predicate>(predicate)), code, message);
        }
    } // namespace TensorValidation
} // namespace NanoChat
#endif
