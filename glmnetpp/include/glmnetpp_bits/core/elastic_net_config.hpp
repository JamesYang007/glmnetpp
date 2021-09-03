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
    util::index_t nlambda = 100; // reset if lambda is active
    std::optional<std::vector<double>> lambda;
    double thresh = 1e-7;
    double alpha = 1;
    double lambda_min_ratio;     // ignored if lambda is active
    util::index_t max_active = -1;           
    util::index_t max_non_zero = -1;

    inline void setup(double lambda_max,
                      size_t nobs,
                      size_t nvars)
    {
        auto n = nobs;
        auto p = nvars;

        // setup uninitialized values
        lambda_min_ratio = (n < p) ? 0.01 : 1e-4;

        // if user did not initialize, by default use all the columns.
        // else, cap it at p.
        if (max_active == -1) max_active = p;
        else max_active = std::min(static_cast<size_t>(max_active), p);
        if (max_non_zero == -1) max_non_zero = p;
        else max_non_zero = std::min(static_cast<size_t>(max_non_zero), p);

        // setup lambda vector if not user-supplied
        // create an array of lambda values from lambda_max to lambda_min in descending order
        // evenly spaced on a log-scale.
        if (!lambda) {
            lambda_max_ = lambda_max;
            double lambda_min = lambda_min_ratio * lambda_max_;
		    lambda_factor_ = std::exp(
                (std::log(lambda_min) - std::log(lambda_max_))/(nlambda-1.)
            );
        }
        else {
            nlambda = lambda->size();
        }
    }

    double get_lambda(util::index_t l, double curr_lambda) const
    {
        static_cast<void>(curr_lambda);
        if (lambda) { return lambda.value()[l]; }
        if (l == 0) { return lambda_max_; }
        if (l == 1) { return lambda_max_ * lambda_factor_ / std::max(alpha, 1e-3); }
        return curr_lambda * lambda_factor_;
    }

private:
    // To be initialized during setup:

    // Only initialized when lambda is not active.
    double lambda_factor_;       // factor to multiply to current lambda to get next lambda.
                                 // ignored if lambda active.
    double lambda_max_;          // max lambda to start the sequence

};

} // namespace core
} // namespace glmentpp
