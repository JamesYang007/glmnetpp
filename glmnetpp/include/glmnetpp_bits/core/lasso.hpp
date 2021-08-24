#pragma once
#include <Eigen/Core>
#include <glmnetpp_bits/core/elastic_net_config.hpp>
#include <glmnetpp_bits/core/elastic_net_output.hpp>

#include <iostream>

namespace glmnetpp {
namespace core {
 
/*
 * Computes the lasso path given X, y, and a configuration object.
 * Assumes that X and y have been centered and scaled.
 */
template <class XDerived, class YDerived>
inline ElasticNetOutput lasso_path(const Eigen::MatrixBase<XDerived>& X,
                                   const Eigen::MatrixBase<YDerived>& y,
                                   ElasticNetConfig config)
                                   
{
    using value_t = typename Eigen::MatrixBase<XDerived>::value_type;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;

    // auxillary variables
    auto n = X.rows();
    auto p = X.cols();

    // cache variables
    vec_t Xty = X.transpose() * y;
    
    // setup rest of configuration
    config.setup(Xty, n);

    // initialization
    ElasticNetOutput output;
    output.beta.resize(p, config.nlambda);
    output.beta.col(0).setZero();

    vec_t resid = Xty / n;                      // jth component = residual from fitting 
                                                // on current beta without jth comp.
    mat_t X_cov(p, p);                          // lazily caches X^T*X (only column vectors)
    std::vector<bool> X_cov_flags(p, false);    // contains whether column j of X_cov is set or not

    for (int l = 1; l < config.lambda->size(); ++l) {
        
        const value_t lambda = (*config.lambda)[l];
        output.beta.col(l) = output.beta.col(l-1);

        for (int i = 0; i < config.max_iter; ++i) {

            value_t max_abs_diff = 0;

            for (int j = 0; j < p; ++j) {

                value_t old_beta_j = output.beta(j, l);

                // check soft-max threshold
                output.beta(j,l) = [lambda, j, &resid]() {
                    if (lambda >= std::abs(resid[j])) return 0.;
                    else if (resid[j] > 0) return resid[j] - lambda;
                    else return resid[j] + lambda;
                }();

                // update residuals only if beta_j changed
                value_t beta_diff = old_beta_j - output.beta(j,l);
                if (beta_diff != 0) {
                    value_t cache_j = resid[j]; // for vectorization, we update everything including jth component,
                                                // then we replace the jth component with the old value.
                    
                    // update X covariance matrix column j if not set before
                    if (!X_cov_flags[j]) {
                        X_cov_flags[j] = true;
                        X_cov.col(j) = X.transpose() * X.col(j);
                    }

                    resid += X_cov.col(j) * (beta_diff / n);
                    resid[j] = cache_j;
                }

                // update max abs diff of betas
                max_abs_diff = std::max(max_abs_diff, std::abs(beta_diff));
            }

            // eager exit if all coefficient (abs) differences are below tolerance 
            if (max_abs_diff < config.thresh) break;

        }
    }

    // move over set-up config into output
    // important to do this after everything above
    // so that config remains valid.
    output.config = std::move(config);

    return output;
}

} // namespace core
} // namespace glmnetpp
