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

/*
 * Linear model trained with L1 prior as regularizer.
 * Optimization function:
 *
 * 1/2 ||y - Xb||^2 + lmda * ||b||_1
 * 
 * Currently, the model assumes that y (n-vector) and X (n-by-p matrix) 
 * are centered and scaled such that ||y|| = ||x_j|| = 1 for all j = 1,...,p
 * and x_j are column vectors of X.
 *
 * Notes: 
 * - glmnet computes R^2 differently (seems to be incorrect).
 */
class Lasso
{
    struct ElasticNetInternal;

public:
    Lasso(const ElasticNetConfig& config = ElasticNetConfig())
        : config_{ config }
    {}

    template <class XDerived, class YDerived>
    inline ElasticNetOutput&
    lasso_path(const Eigen::MatrixBase<XDerived>& X,
               const Eigen::MatrixBase<YDerived>& y);

    ElasticNetInternal& get_internal() { return internal_; }
    ElasticNetInternal get_internal() const { return internal_; }

private:
    using index_t = util::index_t;

    // Internal parameters
    struct ElasticNetInternal
    {
        double delta_r_sq_prop_thresh_ = 1e-5;
        double max_r_sq_ = 0.999;
    };

    // Flags to indicate state of lasso.
    enum class lasso_state_
    {
        noop_,
        max_iter_reached_,
        max_inv_active_set_size_reached_,
        beta_no_change_
    };

	/*
	 * Checks the KKT condition for optimizing lasso at b_j.
	 *
	 * KKT condition violated if and only if (|x_j^T (y-Xb)| > lambda)
	 *
	 * @param   grad_j      jth partial of negative loss: x_j^T (y-Xb)
	 * @param   lambda      lmda in the above formula
	 *
	 * @return  true if KKT condition is not violated.
	 */
	template <class T>
	static inline bool check_kkt_lasso(T grad_j, T lambda)
	{
		auto abs_grad_j = std::abs(grad_j);
		return abs_grad_j <= lambda;
	}

    /*
     * Computes whether convergence threshold is reached.
     * 
     * @param   v       maximum absolute difference of the coordinates 
     *                  between previous beta and current beta.
     * @return  true if threshold reached.
     */ 
    bool convg_thresh_reached_(double v) const { return v * v < config_.thresh; }

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

    /*
     * Updates beta via coordinate descent rule.
     */
    template <class ValueType>
    static inline void update_beta_(ValueType& beta,
								    ValueType grad,
								    ValueType lambda)
    {
		// TODO: loo_grad should be computed using + ||x_j||^2 beta[j]
		auto loo_grad = grad + beta;
		beta = util::soft_threshold(loo_grad, lambda);
    }

    /*
     * Updates X_cov, if necessary.
     */
    template <class XCovType, class XType>
    static inline void update_x_cov_(index_t idx,
                                     XCovType& X_cov,
                                     const XType& X)
    {
		if (!X_cov.is_set(idx)) {
			X_cov.allocate(idx);
			X_cov.col(idx) = X.transpose() * X.col(idx);
		}
    }

    /*
     * Updates gradient using an inner product vector of the form X^Tx_j
     * and beta_diff = (beta_old - beta_new) / n, where n is the number of data points.
     */
    template <class GradType, class XInnerProdType, class ValueType>
    static inline void update_grad_(GradType&& grad,
                                    XInnerProdType&& x_inner_prod,
                                    ValueType beta_diff)
    {
	    // TODO: n should actually go away and be absorbed into x_inner_prod.
        grad += x_inner_prod * beta_diff;
    }

    /*
     * Updates r_sq (R^2) using a gradient value at index j
     * and beta_diff = beta_old_j - beta_new_j.
     */
    template <class ValueType>
    static inline void update_r_sq_(ValueType& r_sq,
                                    ValueType grad,
                                    ValueType beta_diff)
    {
		// TODO: should be -beta_diff * ||x_j||^2 at the end
        r_sq -= beta_diff * (2.0 * grad - beta_diff);
    }

    /*
     * Updates maximum absolute beta difference.
     */
    template <class ValueType>
    static inline void update_max_abs_diff_(ValueType& max_abs_diff,
                                            ValueType beta_diff)
    {
		max_abs_diff = std::max(max_abs_diff, std::abs(beta_diff));
    }

    ElasticNetConfig config_;
    ElasticNetInternal internal_;
    ElasticNetOutput output_;
    lasso_state_ curr_state_ = lasso_state_::noop_;
};

template <class IncludeSubsetIterType
		, class ExcludeActionType
        , class GradVecType
		, class ActionType>
inline bool Lasso::check_kkt_(IncludeSubsetIterType isub_begin,
							  IncludeSubsetIterType isub_end,
							  ExcludeActionType esub_f,
							  const GradVecType& grad,
							  double lambda,
							  ActionType f) 
{
	bool all_kkt_met = true;

    std::for_each(isub_begin, isub_end,
        [=, &grad, &all_kkt_met](auto j) {
            if (esub_f(j)) return;
            // TODO: check_kkt_lasso should probably change later to a lambda parameter
            bool kkt_met = check_kkt_lasso(grad[j], lambda);
            if (!kkt_met) {
                all_kkt_met = false;
                f(j);
            }
        });

    return all_kkt_met;
}
 
/*
 * Computes the lasso path given X, y, and a configuration object.
 * Assumes that X and y have been centered and scaled.
 */
template <class XDerived, class YDerived>
inline ElasticNetOutput& 
Lasso::lasso_path(const Eigen::MatrixBase<XDerived>& X,
                  const Eigen::MatrixBase<YDerived>& y)
{
    // TODO: 
    // 1. generalize to arbitrary weights.
    // 2. generalize policy that defines how to perform the coordinate descent at index j. 
    // 3. add a check that active set never exceeds a user-specified number of predictors.
    using value_t = typename Eigen::MatrixBase<XDerived>::value_type;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;

    // if max_iter is <= 0, yeet outta here
    if (config_.max_iter <= 0) { 
        curr_state_ = lasso_state_::max_iter_reached_;
        return output_; 
    }

    // convenience variables
    auto n = X.rows();
    auto p = X.cols();

    vec_t grad = (X.transpose() * y) / n ;  // stores current gradient
    util::CompressedMatrix<value_t> X_cov(p, p); // caches X^t X
    
    // setup rest of configuration
    config_.setup(grad, n);
    
    // active_set contains whether index j has ever been active 
    std::vector<bool> active_set(p, false);

    // inv_active_set contains at position k, the kth index that became active.
    std::vector<index_t> inv_active_set;
    inv_active_set.reserve(p);

    // proposal_set contains whether index j is currently considered for update.
    std::vector<bool> proposal_set(p, false);
    
    // inv_proposal_set contains at position k, the kth index that became active.
    std::vector<index_t> inv_proposal_set;
    inv_proposal_set.reserve(p);

    // contains indices that survived the strong rule criterion at each lambda
    // see (https://doi.org/10.1111/j.1467-9868.2011.01004.x).
    // Note that at step lambda_l, the strong rule is applied with lambda_{l-1}
    std::vector<bool> strong_set(p, false);
    std::vector<index_t> inv_strong_set;
    inv_strong_set.reserve(p);

    // compressed beta when iterating on a subset of predictors
    // if there are k proposed beta, the first k components will be the compressed version.
    vec_t compressed_beta(p);

	index_t iter = 0;
    value_t curr_lambda = config_.get_lambda(0, std::numeric_limits<value_t>::infinity());
    value_t curr_r_sq = 0;

    // initialize output
    output_.beta.resize(p, config_.nlambda);
    output_.beta.setZero();
    output_.lambda.push_back(curr_lambda);

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
        inv_strong_set.clear();
        std::fill(strong_set.begin(), strong_set.end(), false);
        for (index_t j = 0; j < grad.size(); ++j) {
            auto abs_grad_j = std::abs(grad[j]);
            if (abs_grad_j >= 2 * curr_lambda - prev_lambda) {
                strong_set[j] = true;
                inv_strong_set.push_back(j);
            }
        }

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

} // namespace glmnetpp
