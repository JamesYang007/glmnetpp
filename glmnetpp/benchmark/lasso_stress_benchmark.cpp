#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <glmnetpp_bits/core/elastic_net_impl_default.hpp>
#include <glmnetpp_bits/core/as_fit_gaussian.hpp>
#include <glmnetpp_bits/core/update_resource_gaussian.hpp>
#include <testutil/data_util.hpp>
#include <ctime>
#include <string>

namespace glmnetpp {
namespace core {

struct lasso_stress_fixture : benchmark::Fixture
{
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
    ElasticNetConfig config;
    using fit_t = ASFit<util::method_type::gaussian_cov,
                        UpdateResource<util::method_type::gaussian_cov,
                                       double> >;
    using lasso_t = ElasticNetImplDefault<fit_t>;
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
        auto model = lasso_t(config);
        auto out = model.fit_path(X, y);
    }
}

BENCHMARK_REGISTER_F(lasso_stress_fixture,
                     lasso_unif_X_y)
    ->ArgsProduct({
        {100, 500, 1000, 2000},
        benchmark::CreateRange(2, 1<<14, 2)
        })
    ;

} // namespace core
} // namespace glmnetpp
