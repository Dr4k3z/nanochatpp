#ifndef NANOCHAT_SCALAR_REFERENCE_H
#define NANOCHAT_SCALAR_REFERENCE_H

#include <Eigen/Core>
#include <memory>
#include <utility>

namespace NanoChat {
    template<typename>
    class Tensor;
    /// A scalar result that retains storage ownership instead of borrowing from a temporary Tensor.
    template<typename Scalar>
    class ScalarReference {
    public:
        using Storage = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        ScalarReference(const ScalarReference&) = default;
        ScalarReference(ScalarReference&&) = default;
        operator Scalar() const { return (*m_storage)[m_offset]; }
        ScalarReference& operator=(Scalar value) {
            (*m_storage)[m_offset] = std::move(value);
            return *this;
        }
        ScalarReference& operator=(const ScalarReference& other) { return *this = static_cast<Scalar>(other); }

    private:
        template<typename>
        friend class Tensor;
        ScalarReference(std::shared_ptr<Storage> storage, std::size_t offset) :
            m_storage(std::move(storage)), m_offset(static_cast<Eigen::Index>(offset)) {}
        std::shared_ptr<Storage> m_storage;
        Eigen::Index m_offset;
    };
} // namespace NanoChat
#endif
