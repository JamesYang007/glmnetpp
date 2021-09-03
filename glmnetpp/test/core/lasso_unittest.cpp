#include <testutil/base_fixture.hpp>
#include <testutil/data_util.hpp>
#include <Eigen/Core>
#include <glmnetpp_bits/core/elastic_net_impl_default.hpp>
#include <glmnetpp_bits/core/as_fit_gaussian.hpp>
#include <glmnetpp_bits/core/update_resource_gaussian.hpp>

namespace glmnetpp {
namespace core {

struct lasso_fixture : base_fixture
{
protected:

    using fit_t = ASFit<::glmnetpp::util::method_type::gaussian_cov,
                        UpdateResource<::glmnetpp::util::method_type::gaussian_cov,
                                       value_t>
                        >;
    using lasso_t = ElasticNetImplDefault<fit_t>;

    enum class init_type
    {
        fixed,
        random
    };

    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    ElasticNetConfig config;
    std::unique_ptr<lasso_t> lasso_ptr;
    ::glmnetpp::util::index_t n, p;

    void setup(uint32_t max_iter,
               uint32_t nlambda)
    {
        config.max_iter = max_iter;
        config.nlambda = nlambda;
        lasso_ptr.reset(new lasso_t(config));
        auto& int_param = lasso_ptr->get_internal();

        // make thresholds more stringent to test accuracy
        int_param.delta_r_sq_prop_thresh_ = 0;  
        int_param.max_r_sq_ = 1;
    }

    void initialize_data(init_type type)
    {
        if (type == init_type::fixed) {
			n = 5;
			p = 3;

            X.resize(n, p);
            y.resize(n);
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
            n = X.rows();
            p = X.cols();
        }

        X = center_scale(X);
        y = center_scale(y);
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
    setup(1, 2);
    initialize_data(init_type::fixed);

    auto output = lasso_ptr->fit_path(X, y);

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
    setup(2, 2);
    initialize_data(init_type::fixed);

    auto output = lasso_ptr->fit_path(X, y);

    // first column should always be 0
    Eigen::MatrixXd expected(p, 2);
    expected << 0., 0.7486140916829671,
                0., 0.,
                0., 0.;
    expect_double_eq_mat(output.beta, expected);
}

// Test lasso path with 5 lambda until convergence
TEST_F(lasso_fixture, lasso_path_five_lmda)
{
    setup(10000, 5);
    initialize_data(init_type::fixed);

    auto output = lasso_ptr->fit_path(X, y);

    // first column should always be 0
    Eigen::MatrixXd expected(p, 5);
    expected << 0., 6.7382006452112231e-01, 7.4120207097323432e-01, 7.4794027161844556e-01, 7.4861409168296678e-01,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.;
    expect_near_mat(output.beta, expected, 1e-13);
}

// Test lasso path with 10 lambda until convergence
TEST_F(lasso_fixture, lasso_path_ten_lmda)
{
    setup(10000, 10);
    initialize_data(init_type::fixed);

    auto output = lasso_ptr->fit_path(X, y);

    // first column should always be 0
    Eigen::MatrixXd expected(p, 10);
    expected << 
        // row 1
        0., 4.7962409893216640e-01,
        6.5199206295540635e-01, 7.1393789738629698e-01,
        7.3620007600564841e-01, 7.4420068817647711e-01,
        7.4707595911030922e-01, 7.4810927790722392e-01,
        7.4848063342836590e-01, 7.4861409168296678e-01,
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
    expect_near_mat(output.beta, expected, 1e-13);
}

// Test lasso path with larger, random data
TEST_F(lasso_fixture, lasso_path_random_one_step_two_lmda)
{
    setup(1, 2);
    initialize_data(init_type::random);

    auto output = lasso_ptr->fit_path(X, y);

    // first column should always be 0
    Eigen::MatrixXd expected(p, 2);
    expected << 
        0., 5.5067887034115678e-02,
        0., -7.0417822053915891e-02,
        0., -1.2593788819065974e-01,
        0.,  9.3475661735084953e-02,
        0., -5.9438448636644518e-03
        ;
    expect_near_mat(output.beta, expected, 1e-13);
}

TEST_F(lasso_fixture, lasso_path_random_five_lmda)
{
    setup(10000, 5);
    initialize_data(init_type::random);

    auto output = lasso_ptr->fit_path(X, y);

    // first column should always be 0
    Eigen::MatrixXd expected(p, 5);
    expected << 
        0., 0.06103714866101418, 0.07743947020576675, 0.07911935666075842, 0.07923996334869325,
        0., -0.07522530657637544, -0.09280840454522671, -0.094595701582309, -0.09472398066060869,
        0., -0.1045255496862108, -0.11316315398017819, -0.11400490736819723, -0.11411600688241703,
        0., 0.08487693282471384, 0.1005254249889087, 0.10208101737055336, 0.1022270133554916,
        0., 0., -0.011455688051035473, -0.012853554326070916, -0.012978194547333968
        ;
    expect_near_mat(output.beta, expected, 1e-13);
}

} // namespace core
} // namespace glmnetpp
