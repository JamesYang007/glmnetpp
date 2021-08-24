#include <testutil/base_fixture.hpp>
#include <glmnetpp_bits/core/elastic_net_config.hpp>

namespace glmnetpp {
namespace core {

struct elastic_net_config_fixture : base_fixture
{
protected:
    Eigen::VectorXd Xty;
    index_t nobs_small;
    index_t nobs_large;
    ElasticNetConfig config;

    elastic_net_config_fixture()
        : Xty(5)
        , nobs_small{2}
        , nobs_large{10}
    {
        Xty << 1,3,5,2,0;
    }
};

// Test lambda_min_ratio setup
TEST_F(elastic_net_config_fixture, 
        setup_lambda_min_ratio_nobs_small)
{
    config.setup(Xty, nobs_small); 
    EXPECT_DOUBLE_EQ(config.lambda_min_ratio, 0.01);
}

TEST_F(elastic_net_config_fixture,
        setup_lambda_min_ratio_nobs_large)
{
    config.setup(Xty, nobs_large); 
    EXPECT_DOUBLE_EQ(config.lambda_min_ratio, 1e-4);
}

// Test lambda vector setup
TEST_F(elastic_net_config_fixture,
        setup_lambda_user_supplied)
{
    Eigen::VectorXd expected(5);
    expected << 1, 3, 0, 2, 3;
    config.lambda = expected;
    config.setup(Xty, nobs_large);
    expect_double_eq_vec(*config.lambda, expected);
}

TEST_F(elastic_net_config_fixture,
        setup_lambda_default)
{
    config.nlambda = 2;

    // important to call this first - sets config.lambda_min_ratio
    config.setup(Xty, nobs_large);

    Eigen::VectorXd expected(config.nlambda);
    double lambda_max = Xty.array().abs().maxCoeff() / nobs_large;
    double lambda_min = config.lambda_min_ratio * lambda_max;
    expected << lambda_max, lambda_min;

    expect_near_vec(*config.lambda, expected, 1e-8);
}

} // namespace core
} // namespace glmnetpp
