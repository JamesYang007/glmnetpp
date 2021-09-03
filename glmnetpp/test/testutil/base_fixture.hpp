#pragma once
#include "gtest/gtest.h"
#include <Eigen/Core>

namespace glmnetpp {

struct base_fixture : ::testing::Test
{
protected:
    using value_t = double;
    using index_t = Eigen::Index;

    // Useful tools to test vector equality
    template <class VecType1, class VecType2>
    void expect_double_eq_vec(const VecType1& v1,
                              const VecType2& v2)
    {
        EXPECT_EQ(v1.size(), v2.size());
        for (index_t i = 0; i < v1.size(); ++i) {
            EXPECT_DOUBLE_EQ(v1[i], v2[i]);
        }
    }

    template <class MatType1, class MatType2>
    void expect_double_eq_mat(const MatType1& m1,
                              const MatType2& m2)
    {
        EXPECT_EQ(m1.rows(), m2.rows());
        EXPECT_EQ(m1.cols(), m2.cols());
        for (index_t j = 0; j < m1.cols(); ++j) {
            for (index_t i = 0; i < m1.rows(); ++i) {
                EXPECT_DOUBLE_EQ(m1(i,j), m2(i,j));
            }
        }
    }

    template <class VecType1, class VecType2>
    void expect_near_vec(const VecType1& v1,
                         const VecType2& v2,
                         double tol=0)
    {
        EXPECT_EQ(v1.size(), v2.size());
        for (index_t i = 0; i < v1.size(); ++i) {
            EXPECT_NEAR(v1[i], v2[i], tol);
        }
    }

    template <class MatType1, class MatType2>
    void expect_near_mat(const MatType1& m1,
                         const MatType2& m2,
                         double tol=0)
    {
        EXPECT_EQ(m1.rows(), m2.rows());
        EXPECT_EQ(m1.cols(), m2.cols());
        for (index_t j = 0; j < m1.cols(); ++j) {
            for (index_t i = 0; i < m1.rows(); ++i) {
                EXPECT_NEAR(m1(i,j), m2(i,j), tol);
            }
        }
    }
};

} // namespace glmnetpp
