#include <testutil/base_fixture.hpp>
#include <glmnetpp_bits/core/elastic_net_config.hpp>

namespace glmnetpp {

struct elastic_net_config_fixture : base_fixture
{
protected:
    Eigen::VectorXd init_grad;
    index_t nobs_small;
    index_t nobs_large;
    ElasticNetConfig config;

    elastic_net_config_fixture()
        : init_grad(5)
        , nobs_small{2}
        , nobs_large{10}
    {
        init_grad << 1,3,5,2,0;
    }
};

// Test lambda_min_ratio setup
TEST_F(elastic_net_config_fixture, 
        setup_lambda_min_ratio_nobs_small)
{
    config.setup(init_grad, nobs_small); 
    EXPECT_DOUBLE_EQ(config.lambda_min_ratio, 0.01);
}

TEST_F(elastic_net_config_fixture,
        setup_lambda_min_ratio_nobs_large)
{
    config.setup(init_grad, nobs_large); 
    EXPECT_DOUBLE_EQ(config.lambda_min_ratio, 1e-4);
}

TEST_F(elastic_net_config_fixture,
        setup_max_active_default)
{
    config.setup(init_grad, nobs_small);
    EXPECT_EQ(config.max_active, init_grad.size());
}

TEST_F(elastic_net_config_fixture,
        setup_max_active_user)
{
    config.max_active = 123;
    config.setup(init_grad, nobs_large);
    EXPECT_EQ(config.max_active, init_grad.size());

    config.max_active = 2;
    config.setup(init_grad, nobs_large);
    EXPECT_EQ(config.max_active, 2);
}

TEST_F(elastic_net_config_fixture,
        setup_max_non_zero_default)
{
    config.setup(init_grad, nobs_small);
    EXPECT_EQ(config.max_non_zero, init_grad.size());
}

TEST_F(elastic_net_config_fixture,
        setup_max_non_zero_user)
{
    config.max_non_zero = 123;
    config.setup(init_grad, nobs_small);
    EXPECT_EQ(config.max_non_zero, init_grad.size());

    config.max_non_zero = 2;
    config.setup(init_grad, nobs_small);
    EXPECT_EQ(config.max_non_zero, 2);
}

// Test lambda vector setup
TEST_F(elastic_net_config_fixture,
        setup_lambda_user_supplied)
{
    std::vector<double> expected({ 1, 3, 0, 2, 3 });
    config.lambda = expected;
    config.setup(init_grad, nobs_large);

    for (size_t i = 0; i < expected.size(); ++i) {
        auto lmda = config.get_lambda(i, 0);
        EXPECT_DOUBLE_EQ(lmda, expected[i]);
    }
}

TEST_F(elastic_net_config_fixture,
        setup_lambda_default)
{
    config.nlambda = 10;

    // important to call this first - sets config.lambda_min_ratio
    config.setup(init_grad, nobs_large);

    double curr_lmda = init_grad.array().abs().maxCoeff();
    double factor = std::pow(config.lambda_min_ratio, 1. / (config.nlambda - 1));
    EXPECT_DOUBLE_EQ(curr_lmda, config.get_lambda(0, 0));

    for (int i = 1; i < config.nlambda; ++i) {
        auto curr_lmda_expected = curr_lmda * factor;
        curr_lmda = config.get_lambda(i, curr_lmda);
        EXPECT_DOUBLE_EQ(curr_lmda, curr_lmda_expected);
    }
}

} // namespace glmnetpp
