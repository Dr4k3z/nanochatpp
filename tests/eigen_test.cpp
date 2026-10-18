#include <Eigen/Core>
#include <gtest/gtest.h>

TEST(EigenIntegration, MatrixVectorProduct) {
    Eigen::Matrix2d matrix;
    matrix << 1.0, 2.0, 3.0, 4.0;
    const Eigen::Vector2d vector{2.0, 3.0};
    const Eigen::Vector2d result = matrix * vector;

    EXPECT_DOUBLE_EQ(result.x(), 8.0);
    EXPECT_DOUBLE_EQ(result.y(), 18.0);
}
