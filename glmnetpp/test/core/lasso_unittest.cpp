#include <testutil/base_fixture.hpp>
#include <testutil/data_util.hpp>
#include <glmnetpp_bits/core/lasso.hpp>
#include <Eigen/Core>

namespace glmnetpp {
namespace core {

struct lasso_fixture : base_fixture
{
protected:

    enum class init_type
    {
        fixed,
        random
    };

    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    Eigen::VectorXd Xty;
    ElasticNetConfig config;
    util::index_t n, p;

    void setup_config(uint32_t max_iter,
                      uint32_t nlambda)
    {
        config.max_iter = max_iter;
        config.nlambda = nlambda;
    }

    void initialize_data(init_type type)
    {
        n = (type == init_type::fixed) ? 5 : 100;
        p = (type == init_type::fixed) ? 3 : 5;

        if (type == init_type::fixed) {
            X.resize(n, p);
            y.resize(n);
            Xty.resize(p);
            X << 1, 2, 3, 
                 4, 5, 6, 
                 7, 8, 9, 
                 10, 11, 12, 
                 13, 14, 15;
            y << -0.2, 3.1, 0.03, 2.3, 10.3;
        }

        else {
            X = read_csv("data/x_data_1.txt");
            y = read_csv("data/y_data_1.txt");
        }

        X = center_scale(X);
        y = center_scale(y);
        Xty = X.transpose() * y;
    }
};

// Test center_scale helper method
TEST_F(lasso_fixture, center_scale_simple)
{
    X = Eigen::VectorXd::LinSpaced(15, 1, 15);
    X.resize(5,3);
    Eigen::MatrixXd X_cs = center_scale(X);
    Eigen::MatrixXd expected(X_cs.rows(), X_cs.cols());
    expected << -1.4142135623730951, -1.4142135623730951, -1.4142135623730951,
                -0.7071067811865476, -0.7071067811865476, -0.7071067811865476,
                0.,          0.,          0.,
                0.7071067811865476,  0.7071067811865476,  0.7071067811865476,
                1.4142135623730951, 1.4142135623730951, 1.4142135623730951;
    expect_double_eq_mat(X_cs, expected);
}

// Test lasso path one step with 2 lambda
TEST_F(lasso_fixture, lasso_path_one_step_two_lmda)
{
    setup_config(1, 2);
    initialize_data(init_type::fixed);

    auto output = lasso_path(X, y, config);

    // first column should always be 0
    Eigen::MatrixXd expected(p, 2);
    expected << 0., 0.7486140916829671,
                0., 0.,
                0., 0.;
    expect_double_eq_mat(output.beta, expected);
}

// Test lasso path two step with 2 lambda
TEST_F(lasso_fixture, lasso_path_two_step_two_lmda)
{
    setup_config(2, 2);
    initialize_data(init_type::fixed);

    auto output = lasso_path(X, y, config);

    // first column should always be 0
    Eigen::MatrixXd expected(p, 2);
    expected << 0., 0.7486140916829671,
                0., 0.,
                0., 0.;
    expect_double_eq_mat(output.beta, expected);
}

// Test lasso path one step with 5 lambda
TEST_F(lasso_fixture, lasso_path_two_step_five_lmda)
{
    setup_config(2, 5);
    initialize_data(init_type::fixed);

    auto output = lasso_path(X, y, config);

    // first column should always be 0
    Eigen::MatrixXd expected(p, 5);
    expected << 0., 6.738200645211225e-01, 7.412020709732345e-01, 7.479402716184458e-01, 7.486140916829667e-01,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.;
    expect_double_eq_mat(output.beta, expected);
}

// Test lasso path 10 step with 10 lambda
TEST_F(lasso_fixture, lasso_path_ten_step_ten_lmda)
{
    setup_config(10, 10);
    initialize_data(init_type::fixed);

    auto output = lasso_path(X, y, config);

    // first column should always be 0
    Eigen::MatrixXd expected(p, 10);
    expected << 
        // row 1
        0., 4.7962409893216662e-01, 
        6.5199206295540657e-01, 7.1393789738629698e-01,
        7.3620007600564841e-01, 7.4420068817647689e-01,
        7.4707595911030955e-01, 7.4810927790722426e-01, 
        7.4848063342836613e-01, 7.4861409168296689e-01,
        // row 2
        0., 0., 
        0., 0., 
        0., 0.,
        0., 0.,
        0., 0.,
        // row 3
        0., 0., 
        0., 0., 
        0., 0.,
        0., 0.,
        0., 0.
        ;
    expect_double_eq_mat(output.beta, expected);
}

// Test lasso path with larger, random data
TEST_F(lasso_fixture, lasso_path_random_one_step_two_lmda)
{
    setup_config(1, 2);
    initialize_data(init_type::random);

    auto output = lasso_path(X, y, config);

    // first column should always be 0
    Eigen::MatrixXd expected(p, 2);
    expected << 
        0., 5.5067887034115678e-02,
        0., -7.0417822053915891e-02,
        0., -1.2593788819065974e-01,
        0.,  9.3475661735084953e-02,
        0., -5.9438448636644518e-03
        ;
    expect_near_mat(output.beta, expected, 1e-10);
}

TEST_F(lasso_fixture, lasso_path_random_ten_step_five_lmda)
{
    setup_config(10, 5);
    initialize_data(init_type::random);

    auto output = lasso_path(X, y, config);

    // first column should always be 0
    Eigen::MatrixXd expected(p, 5);
    expected << 
        0., 0.0610624678676764, 0.0774467159082272, 0.0791259701722861, 0.0792946055805973,
        0., -0.075259939996844, -0.0928153802529661, -0.0946019913970585, -0.0947813259834667,
        0., -0.104514036204746,  -0.1131601965175158, -0.114002278013854, -0.1140861994489105,
        0., 0.0848896274490425, 0.1005265142391519, 0.1020820396884099, 0.1022376928407224,
        0., 0., -0.0114578420917296, -0.012855502338117, -0.0129954785220482
        ;
    expect_near_mat(output.beta, expected, 1e-6);
}

} // namespace core
} // namespace glmnetpp
