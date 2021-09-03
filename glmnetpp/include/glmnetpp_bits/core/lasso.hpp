#pragma once
#include <Eigen/Core>
#include <glmnetpp_bits/core/elastic_net_config.hpp>
#include <glmnetpp_bits/core/elastic_net_output.hpp>
#include <glmnetpp_bits/util/compressed_matrix.hpp>
#include <glmnetpp_bits/util/math.hpp>
#include <glmnetpp_bits/util/iterator/counting_iterator.hpp>
#include <algorithm>
#include <unordered_set>
#include <cmath>

namespace glmnetpp {
namespace core {
class Lasso
{

private:

    /*
     * Checks KKT for elements from isub_begin to isub_end(non - inclusive).
     * It does not check for indices j where esub_f(j) evaluates to true.
     * If KKT is violated, f(j) is invoked.
     *
     * @param   isub_begin      begin iterator of inclusive subset of indices to check KKT.
     * @param   isub_end        end iterator of inclusive subset.
     * @param   esub_f          exclusive subset predicate.
     * @param   grad            current gradient.
     * @param   lambda          regularization parameter.
     * @param   f               violation action.
     *
     * @return  true if all indices whose KKT are checked pass.
     */ 
    template <class IncludeSubsetIterType
            , class ExcludeActionType
            , class GradVecType
            , class ActionType>
    static inline bool check_kkt_(IncludeSubsetIterType isub_begin,
                                  IncludeSubsetIterType isub_end,
                                  ExcludeActionType esub_f,
                                  const GradVecType& grad,
                                  double lambda,
                                  ActionType f);


};
 
/*
 * Computes the lasso path given X, y, and a configuration object.
 * Assumes that X and y have been centered and scaled.
 */
template <class XDerived, class YDerived>
inline ElasticNetOutput& 
Lasso::lasso_path(const Eigen::MatrixBase<XDerived>& X,
                  const Eigen::MatrixBase<YDerived>& y)
{
    // iterate over each lambda
    for (index_t l = 1; l < config_.nlambda; ++l) {

        value_t prev_lambda = curr_lambda;
        curr_lambda = config_.get_lambda(l, curr_lambda);
        value_t prev_r_sq = curr_r_sq;
        auto curr_beta = output_.beta.col(l);
        curr_beta = output_.beta.col(l-1);

        // compute the strong set indices using previous beta
        // |x_j^T (y - Xb(l-1))| < 2lmda_l - lmda_{l-1}
        // TODO: gradient computation should subtract ||x_j||^2 * output_.beta...
        // TODO: in general, elastic net gradient should take into account the l2 regularization
        upd_rsrc.init_strong(curr_lambda, prev_lambda);

        // keep optimizing until KKT condition holds for all predictors
        while (1) {

            // First few iterations: the main goal is to get a good approximation of active_set at current lambda.
            // Since strong_set only relies on the previous beta, assuming the previous beta converged,
            // strong_set should have converged as well, i.e. 
            // it is a very good _approximated_ superset of the true new members to the active_set.
            // Hence, it suffices to only iterate through strong_set.
            // We will correct for any errors when we check KKT.
            bool converged_all_kkt_met = false;
            while (1) {
                if (iter == config_.max_iter) {
                    curr_state_ = lasso_state_::max_iter_reached_;
                    return output_;
                }
                ++iter;

                value_t max_abs_diff = 0;
                for (auto j : inv_strong_set) {

                    auto prev_beta_j = curr_beta[j];

                    update_beta_(curr_beta[j], grad[j], curr_lambda);

                    if (prev_beta_j == curr_beta[j]) continue;
                    
                    update_x_cov_(j, X_cov, X);

                    auto vec = X_cov.col(j);
                    auto beta_diff = prev_beta_j - curr_beta[j];

                    // important to update gradient before updating curr_r_sq!
                    update_grad_(grad, vec, beta_diff / n);
                    update_r_sq_(curr_r_sq, grad[j], beta_diff);
                    update_max_abs_diff_(max_abs_diff, beta_diff);

                    if (!active_set[j]) {
                        if (inv_active_set.size() == static_cast<size_t>(config_.max_active)) {
                            curr_state_ = lasso_state_::max_inv_active_set_size_reached_;
                            break;
                        }
                        inv_active_set.push_back(j);
                        active_set[j] = true;
                    }
                }

                if (curr_state_ == lasso_state_::max_inv_active_set_size_reached_) break;

                // Slight optimization: if the one iteration converged,
                // check KKT for the complement of strong_set.
                // If KKT passes, we're done!
                // Otherwise, add to strong_set; most likely it was violated
                // because the strong_set hasn't converged fully yet,
                // so we incorrectly labeled the violator as not being in strong_set.
                if (convg_thresh_reached_(max_abs_diff)) {
                    bool all_kkt_met = check_kkt_(
                        util::counting_iterator<index_t>(0),
                        util::counting_iterator<index_t>(p),
                        [&](index_t j) {
                            return strong_set[j];
                        }, 
                        grad, curr_lambda,
                        [&](index_t j) {
                            if (!strong_set[j]) inv_strong_set.push_back(j);
                            strong_set[j] = true;
                        });
                    if (all_kkt_met) {
                        converged_all_kkt_met = true;
                        break;
                    }
                }
                else break;
            }
            if (curr_state_ == lasso_state_::max_inv_active_set_size_reached_ ||
                converged_all_kkt_met) break;
            
            // initialize proposal set to current ever-active set
            proposal_set = active_set;
            inv_proposal_set = inv_active_set;

            // keep applying coordinate descent over proposal set
            // until KKT condition for strong set is met.
            while (1) {

                // save old proposed beta
                // TODO: can probably optimize by only copying betas
                // that were newly added to inv_proposal_set.
                compressed_beta(Eigen::seqN(0, inv_proposal_set.size())) =
                    curr_beta(inv_proposal_set);

                // apply coordinate descents over proposal set
                while (1) {

                    if (iter == config_.max_iter) {
                        curr_state_ = lasso_state_::max_iter_reached_;
                        return output_;
                    }
                    ++iter;

                    // update beta for predictors in proposal set
                    value_t max_abs_diff = 0;
                    for (auto j : inv_proposal_set) {
                        auto prev_beta_j = curr_beta[j];

                        update_beta_(curr_beta[j], grad[j], curr_lambda);

                        if (prev_beta_j == curr_beta[j]) continue;
                        
                        update_x_cov_(j, X_cov, X);

                        auto vec = X_cov.col(j);
                        auto beta_diff = prev_beta_j - curr_beta[j];

                        // important to update gradient before updating curr_r_sq!
                        // important to only update on proposal set!
                        update_grad_(grad(inv_proposal_set), vec(inv_proposal_set), beta_diff / n);
                        update_r_sq_(curr_r_sq, grad[j], beta_diff);
                        update_max_abs_diff_(max_abs_diff, beta_diff);
                    }

                    if (convg_thresh_reached_(max_abs_diff)) break;

                } // end for - coordinate descent

                // update gradient on all other non-proposed components
                compressed_beta(Eigen::seqN(0, inv_proposal_set.size())) -=
                    curr_beta(inv_proposal_set);

                // TODO: maybe optimize if proposal_set is small enough,
                // then just cache grad components for proposal and 
                // update grad entirely (vectorization), then replace at proposal.
                for (index_t j = 0; j < p; ++j) {
                    if (proposal_set[j]) continue;
                    for (index_t k = 0; static_cast<size_t>(k) < inv_proposal_set.size(); ++k) {
                        if (compressed_beta[k] == 0) continue;
                        auto vec = X_cov.col(inv_proposal_set[k]);
                        update_grad_(grad[j], vec(j), compressed_beta(k) / n);
                    }
                }

                // Check KKT on strong set (ignoring proposal set).
                bool strong_set_kkt_met = check_kkt_(
                    inv_strong_set.begin(),
                    inv_strong_set.end(),
                    [&](index_t j) {
                        return proposal_set[j];
                    },
                    grad, curr_lambda,
                    [&](index_t j) {
                        if (!proposal_set[j]) inv_proposal_set.push_back(j);
                        proposal_set[j] = true;
                    });

                if (strong_set_kkt_met) break;

            } // end for - kkt on strong set met

            // Check KKT condition on all predictors (ignoring proposal_set and strong_set).
            bool all_kkt_met = check_kkt_(
                util::counting_iterator<index_t>(0),
                util::counting_iterator<index_t>(p),
                [&](index_t j) {
                    return proposal_set[j] || strong_set[j];
                },
                grad, curr_lambda,
                [&](index_t j) {
                    if (!strong_set[j]) inv_strong_set.push_back(j);
                    strong_set[j] = true;
                });

            if (all_kkt_met) break;

        } // end while - all kkt met

        // no need to compute for further lambdas
        if (curr_state_ == lasso_state_::max_inv_active_set_size_reached_) break;

        output_.lambda.push_back(curr_lambda);

        // Count number of current non-zero coefficients.
        // By definition, suffices to look in active set.
        // In general, must look at the full active set,
        // since coefficients that were non-zero before could now be zero.
        index_t non_zero_count = 0;
        for (auto j : inv_active_set) {
            if (curr_beta(j) != 0) ++non_zero_count;
        }

        if ((non_zero_count > config_.max_non_zero)
            || (curr_r_sq - prev_r_sq < 
                   internal_.delta_r_sq_prop_thresh_ * curr_r_sq)
            || (curr_r_sq > internal_.max_r_sq_)) break;

    } // end for - iterate through lambda

    return output_;
}

} // namespace core
} // namespace glmnetpp
