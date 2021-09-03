#pragma once
#include <glmnetpp_bits/core/elastic_net_config.hpp>
#include <glmnetpp_bits/core/elastic_net_output.hpp>

namespace glmnetpp {
namespace core {

/*
 * The default implementation class of elastic net.
 * Solves for the whole path of lambda.
 *
 * @tparam  FitType     policy for fitting at a particular lambda.
 */
template <class FitType>
class ElasticNetImplDefault
{
    struct InternalParams_;
    using fit_t = FitType;

public:
    using value_t = typename fit_t::value_t;
    using index_t = typename fit_t::index_t;

    ElasticNetImplDefault(const ElasticNetConfig& config = ElasticNetConfig())
        : config_{ config }
    {}

    template <class XType, class YType>
    inline auto
    fit_path(const XType& X,
             const YType& y);

    InternalParams_& get_internal() { return internal_; }
    InternalParams_ get_internal() const { return internal_; }

private:
    // Internal parameters
    struct InternalParams_
    {
        double delta_r_sq_prop_thresh_ = 1e-5;
        double max_r_sq_ = 0.999;
    };

    ElasticNetConfig config_;
    InternalParams_ internal_;
};

template <class FitType>
template <class XType, class YType>
inline auto
ElasticNetImplDefault<FitType>::fit_path(
    const XType& X,
    const YType& y)
{
    // convenience variables
    auto n = X.rows();
    auto p = X.cols();

    fit_t fit_obj(p);

    // TODO: need some kind of initialization/processing?
    ElasticNetOutput output;
    using state_t = typename ElasticNetOutput::state_t;

    // if max_iter is <= 0, yeet outta here
    if (config_.max_iter <= 0) {
        output.state = state_t::max_iter_reached;
        return output;
    }

    fit_obj.init_grad(X, y);

    // setup rest of configuration
    config_.setup(fit_obj.lambda_max(), n, p);

    // initialize miscellaneous variables
    index_t iter = 0;
    value_t curr_lambda = config_.get_lambda(0, std::numeric_limits<value_t>::infinity());
    value_t curr_r_sq = 0;

    // initialize output
    output.beta.resize(p, config_.nlambda);
    output.beta.setZero();
    output.lambda.push_back(curr_lambda);

    // iterate over each lambda
    for (index_t l = 1; l < config_.nlambda; ++l) {

        value_t prev_lambda = curr_lambda;
        curr_lambda = config_.get_lambda(l, curr_lambda);
        value_t prev_r_sq = curr_r_sq;
        auto curr_beta = output.beta.col(l);
        curr_beta = output.beta.col(l - 1);

        auto state = fit_obj.fit(
            curr_lambda, prev_lambda, iter, config_.max_iter,
            config_.max_active, config_.thresh, curr_beta, curr_r_sq, X);

        if (state == state_t::max_active_size_reached ||
            state == state_t::max_iter_reached)
        {
            output.state = state;
            break;
        }

        // no need to compute for further lambdas
        output.lambda.push_back(curr_lambda);

        // Count number of current non-zero coefficients.
        // By definition, suffices to look in active set.
        // In general, must look at the full active set,
        // since coefficients that were non-zero before could now be zero.
        auto non_zero_count = fit_obj.n_non_zero_beta(curr_beta);

        if ((non_zero_count > config_.max_non_zero)
            || (curr_r_sq - prev_r_sq < 
                   internal_.delta_r_sq_prop_thresh_ * curr_r_sq)
            || (curr_r_sq > internal_.max_r_sq_)) break;

    }

    return output;
}

} // namespace core
} // namespace glmnetpp
