#pragma once
#include <Eigen/Core>
#include <glmnetpp_bits/core/elastic_net_config.hpp>
#include <glmnetpp_bits/core/elastic_net_output.hpp>
#include <glmnetpp_bits/util/compressed_matrix.hpp>
#include <glmnetpp_bits/util/math.hpp>
#include <unordered_set>
#include <cmath>

#include <iostream>

namespace glmnetpp {
namespace details {

/*
 * Checks the KKT condition for optimizing lasso at b_j.
 * Optimization function:
 *
 * 1/2 ||y - Xb||^2 + lmda * ||b||_1
 *
 * KKT condition violated if and only if (|x_j^T (y-Xb)| > lambda)
 *
 * @param   grad_j      jth partial of negative loss: x_j^T (y-Xb)
 * @param   lambda      lmda in the above formula
 *
 * @return  true if KKT condition is not violated.
 */
template <class T>
inline bool check_kkt_lasso(T grad_j, T lambda)
{
    auto abs_grad_j = std::abs(grad_j);
    return abs_grad_j <= lambda;
}

} // namespace details

struct Lasso
{
    Lasso(const ElasticNetConfig& config = ElasticNetConfig())
        : config_{ config }
    {}

    template <class XDerived, class YDerived>
    inline ElasticNetOutput&
    lasso_path(const Eigen::MatrixBase<XDerived>& X,
               const Eigen::MatrixBase<YDerived>& y);

private:
    using index_t = typename Eigen::Index;

    enum class lasso_state_
    {
        noop_,
        converged_,
        max_iter_reached_
    };

    bool thresh_reached_(double v) const { return v * v < config_.thresh; }

	template <class IncludeSubsetType
			, class ExcludeActionType
			, class GradVecType
			, class ActionType>
	static inline bool check_kkt_(const IncludeSubsetType& isub,
						          const ExcludeActionType& esub_f,
						          const GradVecType& grad,
						          double lambda,
						          ActionType f);

	template <class ExcludeActionType
			, class GradVecType
			, class ActionType>
	static inline bool check_kkt_(index_t p,
                                  const ExcludeActionType& esub_f,
						          const GradVecType& grad,
						          double lambda,
						          ActionType f);

	// at a given lambda, first iteration: go through all predictors,
	// and update beta, X_cov, active_set/inv_active_set, and grad.
	template <class BetaVecType
			, class GradVecType
			, class XType
			, class XCovType
			, class ActiveSetType
			, class InvActiveSetType
			, class InvStrongSetType>
	inline lasso_state_ update_approx_active_set_(
		index_t n,
		const BetaVecType& old_beta,
		BetaVecType& new_beta,
		GradVecType& grad,
		XType& X,
		XCovType& X_cov,
		ActiveSetType& active_set,
		InvActiveSetType& inv_active_set,
		const InvStrongSetType& inv_strong_set,
		double lambda);

    ElasticNetConfig config_;
    ElasticNetOutput output_;
    lasso_state_ curr_state_;
};

template <class IncludeSubsetType
		, class ExcludeActionType
        , class GradVecType
		, class ActionType>
inline bool Lasso::check_kkt_(const IncludeSubsetType& isub,
					          const ExcludeActionType& esub_f,
                              const GradVecType& grad,
                              double lambda,
					          ActionType f)
{
	// flag indicating whether kkt is met for all isub members
	bool all_kkt_met = true;

	for (auto j : isub) {
        // continue if exclude
        if (esub_f(j)) continue;
        // TODO: check_kkt_lasso should probably change later to a lambda parameter
		bool kkt_met = details::check_kkt_lasso(grad[j], lambda); 
		if (!kkt_met) {
			all_kkt_met = false;
            f(j);
		}
	}

    return all_kkt_met;
}

template <class ExcludeActionType
		, class GradVecType
		, class ActionType>
inline bool Lasso::check_kkt_(index_t p,
							  const ExcludeActionType& esub_f,
							  const GradVecType& grad,
							  double lambda,
							  ActionType f)
{
	// flag indicating whether kkt is met for all isub members
	bool all_kkt_met = true;

    for (index_t j = 0; j < p; ++j) {
        // continue if exclude
        if (esub_f(j)) continue;
		auto grad_j = grad[j];
        // TODO: check_kkt_lasso should probably change later to a lambda parameter
		bool kkt_met = details::check_kkt_lasso(grad_j, lambda); 
		if (!kkt_met) {
			all_kkt_met = false;
            f(j);
		}
	}

    return all_kkt_met;
}

template <class BetaVecType
        , class GradVecType
        , class XType
        , class XCovType
        , class ActiveSetType
        , class InvActiveSetType
        , class InvStrongSetType>
inline Lasso::lasso_state_ Lasso::update_approx_active_set_(
    index_t n,
	const BetaVecType& old_beta,
    BetaVecType& new_beta,
    GradVecType& grad,
    XType& X,
    XCovType& X_cov,
    ActiveSetType& active_set,
    InvActiveSetType& inv_active_set,
    const InvStrongSetType& inv_strong_set,
    double lambda)
{
    double max_abs_diff = 0.;

	for (auto j : inv_strong_set) {

        auto old_beta_j = old_beta[j];

		// compute new beta_j
		// TODO: loo_grad should be computed using + ||x_j||^2 beta[j]
		auto loo_grad_j = grad[j] + old_beta_j;
		new_beta[j] = util::soft_threshold(loo_grad_j, lambda);

		// update gradient only if beta_j changed
        if (old_beta_j == new_beta[j]) continue;

		// update X covariance matrix column j if not set before
		auto [vec, was_set_before] = X_cov.col(j);
		if (!was_set_before) vec = X.transpose()* X.col(j);

        if (!active_set[j]) {
            inv_active_set.push_back(j);
            active_set[j] = true;
        }

		auto beta_diff = old_beta_j - new_beta[j];

		// update gradient component
		// TODO: after generalizing to arbitrary weights, 
		// division by n should go away
		grad += vec * (beta_diff / n);

		// update max abs diff of betas
		max_abs_diff = std::max(max_abs_diff, std::abs(beta_diff));

	}

	if (thresh_reached_(max_abs_diff)) {
		// do some lambda checking...
		return lasso_state_::converged_;
	}

    return lasso_state_::noop_;
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
    // 1. lambda vector doesn't need to be a vector if default (generate as we go) (use std::variant?)
    //      - also apply a similar early stopping method.
    // 2. generalize to arbitrary weights
    // 3. mark this is covariance method. 
    using value_t = typename Eigen::MatrixBase<XDerived>::value_type;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;

    // convenience variables
    auto n = X.rows();
    auto p = X.cols();

    vec_t grad = (X.transpose() * y) / n ;  // stores current gradient
    util::CompressedMatrix<value_t> X_cov(p, p); // caches X^t X
    
    // setup rest of configuration
    config_.setup(grad, n);

    // initialization
    output_.beta.resize(p, config_.nlambda);
    output_.beta.setZero();

    // if max_iter is <= 0, yeet outta here
    if (config_.max_iter <= 0) { return output_; }
    
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

	size_t iter = 0;

    // iterate over each lambda
    for (int l = 1; l < config_.lambda->size(); ++l) {
        
        const value_t lambda = (*config_.lambda)[l];

        // compute the strong set indices using previous beta
        // |x_j^T (y - Xb(l-1))| < 2lmda_l - lmda_{l-1}
        // TODO: gradient computation should subtract ||x_j||^2 * output_.beta...
        // TODO: in general, elastic net gradient should take into account the l2 regularization
        inv_strong_set.clear();
        std::fill(strong_set.begin(), strong_set.end(), false);
        for (int j = 0; j < grad.size(); ++j) {
            auto abs_grad_j = std::abs(grad[j]);
            if (abs_grad_j >= 2 * lambda - (*config_.lambda)[l-1]) {
                strong_set[j] = true;
                inv_strong_set.push_back(j);
            }
        }

        // keep optimizing until KKT condition holds for all predictors
        while (1) {

			// First few iterations: the main goal is to get a good approximation of active_set at current lambda.
			// Since strong_set only relies on the previous beta, assuming the previous beta converged,
			// strong_set should have converged as well, i.e. 
			// it is a very good approximation of the set of true new members to the active_set.
			// Hence, it suffices to only iterate through strong_set.
			bool converged_all_kkt_met = false;
			while (1) {
				if (iter == config_.max_iter) {
					curr_state_ = lasso_state_::max_iter_reached_;
					return output_;
				}
				++iter;
				auto update_state = update_approx_active_set_(
					n, output_.beta.col(l-1), output_.beta.col(l), 
					grad, X, X_cov, 
					active_set, inv_active_set, inv_strong_set, lambda);

				// Slight optimization: if the one iteration converged,
				// check KKT for the complement of strong_set.
                // If KKT passes, we're done!
                // Otherwise, add to strong_set; most likely it was violated
                // because the strong_set hasn't converged fully yet,
                // so we incorrectly labeled the violator as not being in strong_set.
				if (update_state == lasso_state_::converged_) {
					bool all_kkt_met = check_kkt_(p, 
						[&](index_t j) {
							return strong_set[j];
						}, 
						grad, lambda,
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
			if (converged_all_kkt_met) break;
			
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
                    output_.beta(inv_proposal_set, l);

                // apply coordinate descents over proposal set
                while (1) {

                    if (iter == config_.max_iter) {
                        curr_state_ = lasso_state_::max_iter_reached_;
                        return output_;
                    }
                    ++iter;

                    // compute threshold criterion for stopping early
                    // max_j |sum_i w_i x_ij y_i|
                    value_t max_abs_diff = 0;

                    // update beta for predictors in proposal set
                    for (auto j : inv_proposal_set) {

                        value_t old_beta_j = output_.beta(j, l);

                        // compute new beta_j
                        // TODO: loo_grad should be computed using + ||x_j||^2 beta[j]
                        auto loo_grad_j = grad[j] + old_beta_j;
                        output_.beta(j, l) = util::soft_threshold(loo_grad_j, lambda);

                        // update gradient only if beta_j changed
                        if (old_beta_j == output_.beta(j, l)) continue;

                        // have to check X_cov again and update since 
                        // new variables from strong set could have been added.
                        auto [vec, was_set_before] = X_cov.col(j);
                        if (!was_set_before) vec = X.transpose() * X.col(j);

                        // update gradient component
                        // TODO: after generalizing to arbitrary weights, 
                        // division by n should go away
                        value_t beta_diff = old_beta_j - output_.beta(j, l);
                        grad(inv_proposal_set) += vec(inv_proposal_set) * (beta_diff / n);

                        // update max abs diff of betas
                        max_abs_diff = std::max(max_abs_diff, std::abs(beta_diff));

                    } // end for - beta update

                    if (thresh_reached_(max_abs_diff)) break;

                } // end for - coordinate descent

                // update gradient on all other non-proposed components
                compressed_beta(Eigen::seqN(0, inv_proposal_set.size())) -=
                    output_.beta(inv_proposal_set, l);

                // TODO: maybe optimize if proposal_set is small enough,
                // then just cache grad components for proposal and 
                // update grad entirely (vectorization), then replace at proposal.
                for (int j = 0; j < p; ++j) {
                    if (proposal_set[j]) continue;
                    for (int k = 0; k < inv_proposal_set.size(); ++k) {
                        if (compressed_beta[k] == 0.) continue;
                        auto [vec, _] = X_cov.col(inv_proposal_set[k]);
                        grad[j] += (compressed_beta(k) / n) * vec(j);
                    }
                }

                bool strong_set_kkt_met = check_kkt_(
                    inv_strong_set,
                    [&](index_t j) {
                        return proposal_set[j];
                    },
                    grad, lambda,
                    [&](index_t j) {
                        if (!proposal_set[j]) inv_proposal_set.push_back(j);
                        proposal_set[j] = true;
                    });

                if (strong_set_kkt_met) break;

            } // end for - kkt on strong set met

            // check KKT condition on all predictors.
            bool all_kkt_met = check_kkt_(p,
                [&](index_t j) {
                    return proposal_set[j] || strong_set[j];
                },
                grad, lambda,
				[&](index_t j) {
                    if (!strong_set[j]) inv_strong_set.push_back(j);
                    strong_set[j] = true;
                });

            if (all_kkt_met) break;

        } // end while - all kkt met

    } // end for - iterate through lambda

    return output_;
}

} // namespace glmnetpp
