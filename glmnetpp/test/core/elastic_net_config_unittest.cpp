#include <testutil/base_fixture.hpp>
#include <glmnetpp_bits/core/elastic_net_config.hpp>

namespace glmnetpp {
namespace core {

struct elastic_net_config_fixture : base_fixture
{
protected:
    value_t lambda_max;
    index_t nobs_small;
    index_t nobs_large;
    index_t nvars;
    ElasticNetConfig config;

    elastic_net_config_fixture()
        : lambda_max{0.2341}
        , nobs_small{2}
        , nobs_large{10}
        , nvars{5}
    {}
};

// Test lambda_min_ratio setup
TEST_F(elastic_net_config_fixture, 
        setup_lambda_min_ratio_nobs_small)
{
    config.setup(lambda_max, nobs_small, nvars); 
    EXPECT_DOUBLE_EQ(config.lambda_min_ratio, 0.01);
}

TEST_F(elastic_net_config_fixture,
        setup_lambda_min_ratio_nobs_large)
{
    config.setup(lambda_max, nobs_large, nvars); 
    EXPECT_DOUBLE_EQ(config.lambda_min_ratio, 1e-4);
}

TEST_F(elastic_net_config_fixture,
        setup_max_active_default)
{
    config.setup(lambda_max, nobs_small, nvars);
    EXPECT_EQ(config.max_active, nvars);
}

TEST_F(elastic_net_config_fixture,
        setup_max_active_user)
{
    config.max_active = 123;
    config.setup(lambda_max, nobs_large, nvars);
    EXPECT_EQ(config.max_active, nvars);

    config.max_active = 2;
    config.setup(lambda_max, nobs_large, nvars);
    EXPECT_EQ(config.max_active, 2);
}

TEST_F(elastic_net_config_fixture,
        setup_max_non_zero_default)
{
    config.setup(lambda_max, nobs_small, nvars);
    EXPECT_EQ(config.max_non_zero, nvars);
}

TEST_F(elastic_net_config_fixture,
        setup_max_non_zero_user)
{
    config.max_non_zero = 123;
    config.setup(lambda_max, nobs_small, nvars);
    EXPECT_EQ(config.max_non_zero, nvars);

    config.max_non_zero = 2;
    config.setup(lambda_max, nobs_small, nvars);
    EXPECT_EQ(config.max_non_zero, 2);
}

// Test lambda vector setup
TEST_F(elastic_net_config_fixture,
        setup_lambda_user_supplied)
{
    std::vector<double> expected({ 1, 3, 0, 2, 3 });
    config.lambda = expected;
    config.setup(lambda_max, nobs_large, nvars);

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
    config.setup(lambda_max, nobs_large, nvars);

    double curr_lmda = lambda_max;
    double factor = std::pow(config.lambda_min_ratio, 1. / (config.nlambda - 1));
    EXPECT_DOUBLE_EQ(curr_lmda, config.get_lambda(0, 0));

    for (int i = 1; i < config.nlambda; ++i) {
        auto curr_lmda_expected = curr_lmda * factor;
        curr_lmda = config.get_lambda(i, curr_lmda);
        EXPECT_DOUBLE_EQ(curr_lmda, curr_lmda_expected);
    }
}

} // namespace core
} // namespace glmnetpp
