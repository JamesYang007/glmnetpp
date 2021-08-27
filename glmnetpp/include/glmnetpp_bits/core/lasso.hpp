#pragma once
#include <Eigen/Core>
#include <glmnetpp_bits/core/elastic_net_config.hpp>
#include <glmnetpp_bits/core/elastic_net_output.hpp>
#include <unordered_set>

#include <iostream>

namespace glmnetpp {
namespace core {

/*
 * Checks the KKT condition for optimizing lasso at b_j.
 * Optimization function:
 *
 * 1/2 ||y - Xb||^2 + lmda * ||b||_1
 *
 * TODO: unsure about this - glmnet just checks case 3
 *
 * KKT condition violated if and only if 
 * 1. (abs_grad_j < lambda and b_j != 0) or
 * 2. (abs_grad_j == lambda and b_j == 0) or
 * 3. (abs_grad_j > lambda)
 *
 * @param   grad_j      jth partial of negative loss: x_j^T (y-Xb)
 * @param   lambda      lmda in the above formula
 * @param   beta_j      jth coordinate of b vector
 *
 * @return  true if KKT condition is not violated.
 */
template <class T>
inline bool check_kkt_lasso(T grad_j, T b_j, T lambda)
{
    auto abs_grad_j = std::abs(grad_j);
                
    return !((abs_grad_j < lambda && b_j != 0) ||
             (abs_grad_j == lambda && b_j == 0) ||
             (abs_grad_j > lambda));
}
 
/*
 * Computes the lasso path given X, y, and a configuration object.
 * Assumes that X and y have been centered and scaled.
 */
template <class XDerived, class YDerived>
inline ElasticNetOutput lasso_path(const Eigen::MatrixBase<XDerived>& X,
                                   const Eigen::MatrixBase<YDerived>& y,
                                   ElasticNetConfig config)
                                   
{

    // TODO: 
    // 1. lambda vector doesn't need to be a vector if default (generate as we go) (use std::variant)
    // 2. generalize to arbitrary weights
    // 3. this is covariance method. 
    // 4.

    using value_t = typename Eigen::MatrixBase<XDerived>::value_type;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;

    // convenience variables
    auto n = X.rows();
    auto p = X.cols();

    // Put all allocations on one contiguous buffer
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
    output.beta.setZero();

    // jth component = current beta without jth comp (leave-one-out).
    loo_beta = Xty / n;                     
    
    // contains j if column j of X_cov is set     
    std::unordered_set<int> X_cov_flags;    

    // contains whether index j has ever been active 
    std::vector<bool> ever_active(p, false); 
    size_t ever_active_max_idx = 0; // technically, one after the max idx

    // contains indices that survived the strong rule criterion at each lambda
    // see (https://doi.org/10.1111/j.1467-9868.2011.01004.x).
    // Note that at step lambda_l, the strong rule is applied with lambda_{l-1}
    std::vector<int> strong_set;

    // At each lmda, contains indices that are currently considered for update.
    // It will be initialized to the ever_active set and 
    // be further updated when KKT condition fails on strong set.
    std::vector<int> proposal_set(p, false);

    // heuristic to ease memory allocation/copies
    // if p/4 > 500 (heuristically, large p), then allocate p/4 amount
    // if p/4 <= 500 (heuristically, small-moderate p), allocate 500 but only if p >= 500
    // otherwise, p < 500 anyways, so just allocate p.
    auto opt_amt = std::min(std::max(500l, p/4), p);
    strong_set.reserve(opt_amt);

    // iterate over each lambda
    for (int l = 1; l < config.lambda->size(); ++l) {
        
        const value_t lambda = (*config.lambda)[l];
        output.beta.col(l) = output.beta.col(l-1);

        // initialize proposal set to current ever-active set
        // only need to copy up to the max idx of ever_active.
        std::copy(ever_active.begin(),
                  std::next(ever_active.begin(), ever_active_max_idx),
                  proposal_set.begin());

        // compute the strong set indices using previous beta
        // |x_j^T (y - Xb(l-1))| < 2lmda_l - lmda_{l-1}
        // TODO: gradient computation should subtract ||x_j||^2 * output.beta...
        // TODO: in general, elastic net gradient should take into account the l2 regularization
        strong_set.clear();
        for (int j = 0; j < loo_beta.size(); ++j) {
            auto abs_grad_j = std::abs(loo_beta[j] - output.beta(j,l-1));
            if (abs_grad_j >= 2 * lambda - (*config.lambda)[l-1]) {
                strong_set.push_back(j);
            }
        }

        // keep optimizing until KKT condition holds for all predictors
        while (1) {

            // keep applying coordinate descent over proposal set
            // until KKT condition for strong set is met.
            while (1) {

                // apply coordinate descent over proposal set
                for (int i = 0; i < config.max_iter; ++i) {

                    // compute threshold criterion for stopping early
                    // max_j |sum_i w_i x_ij y_i|
                    value_t max_abs_diff = 0;

                    // update beta for predictors in proposal set
                    for (int j = 0; j < proposal_set.size(); ++j) {

                        // if not a proposed predictor, continue
                        if (!proposal_set[j]) continue;

                        value_t old_beta_j = output.beta(j, l);

                        // compute new beta_j
                        output.beta(j,l) = [lambda, j, &loo_beta]() {
                            auto abs_grad_j = std::abs(loo_beta[j]);
                            if (lambda >= abs_grad_j) return 0.;
                            else return std::copysign(1., loo_beta[j]) * (abs_grad_j - lambda);
                        }();

                        // update gradient only if beta_j changed
                        value_t beta_diff = old_beta_j - output.beta(j,l);
                        if (beta_diff != 0) {
                            value_t cache_j = loo_beta[j]; // for vectorization, we update everything including jth component,
                                                           // then we replace the jth component with the old value.
                            
                            // update X covariance matrix column j if not set before
                            if (X_cov_flags.find(j) == X_cov_flags.end()) {
                                X_cov_flags.insert(j);
                                X_cov.col(j) = X.transpose() * X.col(j);
                            }

                            // update gradient component
                            // TODO: after generalizing to arbitrary weights, 
                            // division by n should go away
                            loo_beta += X_cov.col(j) * (beta_diff / n);
                            loo_beta[j] = cache_j;

                            // update max abs diff of betas
                            max_abs_diff = std::max(max_abs_diff, std::abs(beta_diff));
                        }
                    } // end for - beta update

                    // eager exit if all coefficient difference squared are below tolerance 
                    if (max_abs_diff * max_abs_diff < config.thresh) break;

                } // end for - coordinate descent

                // check the KKT conditions on strong_set using the current solution 
                //
                // x_j^T (y - X beta_hat(lmda)) 
                //  - lmda * penalty(j) * (1-alpha) * beta_hat_j(lmda) 
                // = 
                // lmda * penalty(j) * alpha * gamma_j
                //
                // where gamma_j is the subgradient of the sgn(beta_hat_j(lmda))
                //
                // TODO: loo_grad_j should be updated with ||x_j||^2 beta_j in general
                // TODO: need lmda * penalty(j) * (1-alpha) subtracted from grad_j
                // TODO: comparison should be with lmda * penalty(j) * alpha
                
                // flag indicating whether strong set kkt is met
                bool strong_set_kkt_met = true;

                for (auto j : strong_set) {

                    auto grad_j = loo_beta[j] - output.beta(j,l);
                    bool kkt_met = check_kkt_lasso(grad_j, output.beta(j,l), lambda); 
                    if (!kkt_met) {
                        strong_set_kkt_met = false;
                        proposal_set[j] = true;
                    }

                } // end for - strong_set
                
                // if kkt is met for all strong set indices, we're done.
                if (strong_set_kkt_met) break;

            } // end while - kkt on strong set met

            // check KKT condition on all predictors.
            bool all_kkt_met = true;

            for (size_t j = 0; j < p; ++j) {

                // can skip the strong set indices
                if (std::binary_search(strong_set.begin(), strong_set.end(), j)) continue;

                auto grad_j = loo_beta[j] - output.beta(j,l);
                bool kkt_met = check_kkt_lasso(grad_j, output.beta(j,l), lambda);
                if (!kkt_met) {
                    all_kkt_met = false;
                    ever_active[j] = true;
                    ever_active_max_idx = std::max(ever_active_max_idx, j+1);
                }

            } // end for - check KKT on all predictors

            if (all_kkt_met) break;

        } // end while - all kkt met

    } // end for - iterate through lambda

    // move over set-up config into output
    // important to do this after everything above
    // so that config remains valid.
    output.config = std::move(config);

    return output;
}

} // namespace core
} // namespace glmnetpp
