#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <glmnetpp_bits/core/lasso.hpp>
#include <testutil/data_util.hpp>

namespace glmnetpp {

struct lasso_stress_fixture : benchmark::Fixture
{
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    ElasticNetConfig config;
};

BENCHMARK_DEFINE_F(lasso_stress_fixture,
                   lasso_large_X_y_vary_p)(benchmark::State& state)
{
    util::index_t n = 100;
    util::index_t p = state.range(0);
    X.resize(n, p);
    y.resize(n);
    X.setRandom();
    y.setRandom();
    X = center_scale(X);
    y = center_scale(y);

    config.nlambda = 6;
    auto model = Lasso(config);

    for (auto _ : state) {
        auto out = model.lasso_path(X, y);
    }
}

BENCHMARK_REGISTER_F(lasso_stress_fixture,
                     lasso_large_X_y_vary_p)
 //->Arg(10)
 //->Arg(50)
 //->Arg(100)
 //->Arg(500)
 //->Arg(1000)
 //->Arg(2000)
 ->Arg(20000)
    ;

} // namespace glmnetpp
