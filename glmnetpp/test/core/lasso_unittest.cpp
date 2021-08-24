#include <testutil/base_fixture.hpp>
#include <glmnetpp_bits/core/lasso.hpp>
#include <Eigen/Core>

namespace glmnetpp {
namespace core {

struct lasso_fixture : base_fixture
{
protected:
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    Eigen::VectorXd Xty;
    ElasticNetConfig config;
    util::index_t n, p;

    lasso_fixture()
        : X(5,3)
        , y(5)
        , Xty(3)
        , config{}
        , n{X.rows()}
        , p{X.cols()}
    {
        X << 1, 2, 3, 
             4, 5, 6, 
             7, 8, 9, 
             10, 11, 12, 
             13, 14, 15;
        y << -0.2, 3.1, 0.03, 2.3, 10.3;

        X = center_scale(X);
        y = center_scale(y);

        Xty = X.transpose() * y;
    }

    void setup_config_one_step()
    {
        config.max_iter = 1;
        config.nlambda = 2;
    }

    template <class T, int R, int C>
    Eigen::Matrix<T, R, C> 
    center_scale(const Eigen::Matrix<T, R, C>& X)
    {
        Eigen::Matrix<T, R, C> out(X.rows(), X.cols());
        auto n = X.rows();
        for (int i = 0; i < X.cols(); ++i) {
            out.col(i) = X.col(i).array() - X.col(i).mean();
            out.col(i) /= out.col(i).norm() / std::sqrt(n);
        }
        return out;
    }
};

TEST_F(lasso_fixture, center_scale_simple)
{
    Eigen::MatrixXd X_cs = center_scale(X);
    Eigen::MatrixXd expected(X_cs.rows(), X_cs.cols());
    expected << -1.4142135623730951, -1.4142135623730951, -1.4142135623730951,
                -0.7071067811865476, -0.7071067811865476, -0.7071067811865476,
                0.,          0.,          0.,
                0.7071067811865476,  0.7071067811865476,  0.7071067811865476,
                1.4142135623730951, 1.4142135623730951, 1.4142135623730951;
    expect_double_eq_mat(X_cs, expected);
}

TEST_F(lasso_fixture, lasso_path_one_step)
{
    setup_config_one_step();

    auto output = lasso_path(X, y, config);

    // first column should always be 0
    Eigen::VectorXd expected(p);
    expected.setZero();
    expect_double_eq_vec(output.beta.col(0), expected);

    // second column check
    Eigen::VectorXd expected_2(p);
    expected_2 << 0.7486140916829671,
                  0.,
                  0.;
    expect_double_eq_vec(output.beta.col(1), expected_2);
}

} // namespace core
} // namespace glmnetpp
