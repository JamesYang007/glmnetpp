#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <glmnetpp_bits/core/lasso.hpp>
#include <testutil/data_util.hpp>
#include <ctime>
#include <string>

namespace glmnetpp {

struct lasso_stress_fixture : benchmark::Fixture
{
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    ElasticNetConfig config;
};

BENCHMARK_DEFINE_F(lasso_stress_fixture,
                   lasso_unif_X_y)(benchmark::State& state)
{
    util::index_t n = state.range(0);
    util::index_t p = state.range(1);

    std::string prefix = "../../../benchmark/data/";
    X = read_csv(prefix + "x_unif_" + std::to_string(n) + 
                 "_" + std::to_string(p) + ".csv");
    y = read_csv(prefix + "y_unif_" + std::to_string(n) +
                 "_" + std::to_string(p) + ".csv");
    X = center_scale(X);
    y = center_scale(y);

    state.counters["n"] = n;
    state.counters["p"] = p;

    for (auto _ : state) {
        auto model = Lasso(config);
        auto out = model.lasso_path(X, y);
    }
}

BENCHMARK_REGISTER_F(lasso_stress_fixture,
                     lasso_unif_X_y)
    ->ArgsProduct({
        {100, 500, 1000, 2000},
        benchmark::CreateRange(2, 1<<14, 2)
        })
    ;

} // namespace glmnetpp
