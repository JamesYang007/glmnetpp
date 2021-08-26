#pragma once
#include <Eigen/Core>
#include <glmnetpp_bits/core/elastic_net_config.hpp>
#include <glmnetpp_bits/core/elastic_net_output.hpp>
#include <unordered_set>

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

    // Try putting everything on one contiguous memory
    mat_t buffer(p, 2 + p);
    Eigen::Map<vec_t> Xty(buffer.col(0).data(), p);
    Eigen::Map<vec_t> loo_beta(buffer.col(1).data(), p);
    Eigen::Map<mat_t> X_cov(buffer.col(2).data(), p, p);

    // cache variables
    Xty = X.transpose() * y;
    
    // setup rest of configuration
    config.setup(Xty, n);

    // initialization
    ElasticNetOutput output;
    output.beta.resize(p, config.nlambda);
    output.beta.col(0).setZero();

    loo_beta = Xty / n;                     // jth component = current beta without jth comp (leave-one-out).
                                                
    std::unordered_set<int> X_cov_flags;    // contains j if column j of X_cov is set 

    for (int l = 1; l < config.lambda->size(); ++l) {
        
        const value_t lambda = (*config.lambda)[l];
        output.beta.col(l) = output.beta.col(l-1);

        for (int i = 0; i < config.max_iter; ++i) {

            value_t max_abs_diff = 0;

            for (int j = 0; j < p; ++j) {

                value_t old_beta_j = output.beta(j, l);

                // check soft-max threshold
                output.beta(j,l) = [lambda, j, &loo_beta]() {
                    if (lambda >= std::abs(loo_beta[j])) return 0.;
                    else if (loo_beta[j] > 0) return loo_beta[j] - lambda;
                    else return loo_beta[j] + lambda;
                }();

                // update residuals only if beta_j changed
                value_t beta_diff = old_beta_j - output.beta(j,l);
                if (beta_diff != 0) {
                    value_t cache_j = loo_beta[j]; // for vectorization, we update everything including jth component,
                                                // then we replace the jth component with the old value.
                    
                    // update X covariance matrix column j if not set before
                    if (X_cov_flags.find(j) == X_cov_flags.end()) {
                        X_cov_flags.insert(j);
                        X_cov.col(j) = X.transpose() * X.col(j);
                    }

                    loo_beta += X_cov.col(j) * (beta_diff / n);
                    loo_beta[j] = cache_j;

					// update max abs diff of betas
					max_abs_diff = std::max(max_abs_diff, std::abs(beta_diff));
                }
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
