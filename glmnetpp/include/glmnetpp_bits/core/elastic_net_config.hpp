#pragma once
#include <Eigen/Core>
#include <vector>
#include <optional>
#include <numeric>
#include <glmnetpp_bits/util/constant.hpp>
#include <glmnetpp_bits/util/eigen_ext.hpp>
#include <glmnetpp_bits/util/typedefs.hpp>

namespace glmnetpp {
namespace core {

struct ElasticNetConfig
{
    util::index_t max_iter = 100000;
    util::index_t nlambda = 100; // ignored if lambda is active
    double thresh = 1e-7;
    double alpha = 1;
    double lambda_min_ratio;
    std::optional<Eigen::VectorXd> lambda;

    template <class XtyDerived>
    inline void setup(const Eigen::MatrixBase<XtyDerived>& Xty,
                      uint32_t nobs)
    {
        auto n = nobs;
        auto p = Xty.size();

        // setup uninitialized values
        lambda_min_ratio = (n < p) ? 0.01 : 1e-4;

        // setup lambda vector if not user-supplied
        // create an array of lambda values from lambda_max to lambda_min in descending order
        // evenly spaced on a log-scale.
        if (!lambda) {

            Eigen::VectorXd lambda_vec(nlambda);

            double lambda_max = Xty.array().abs().maxCoeff() / n;
            double lambda_min = lambda_min_ratio * lambda_max;
            util::geomspace(lambda_vec, lambda_max, lambda_min, nlambda);

            lambda = std::move(lambda_vec);
        }
    }

};

} // namespace core
} // namespace glmentpp
