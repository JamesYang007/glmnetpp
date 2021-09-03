#pragma once
#include <glmnetpp_bits/core/as_fit.hpp>
#include <glmnetpp_bits/util/typedefs.hpp>
#include <glmnetpp_bits/util/iterator/counting_iterator.hpp>

namespace glmnetpp {
namespace core {
namespace details {

/*
 * CRTP base class such that all derived class will have the same boilerplate fit function.
 * Implements the general structure of the active/strong principle algorithm.
 */
template <class ValueType
        , class IndexType
        , class StateType
        , class Derived>
struct ASFitBase
{
    using value_t = ValueType;
    using index_t = IndexType;
    using state_t = StateType;

    template <class BetaType, class XType>
    state_t fit(
        value_t curr_lambda,
        value_t prev_lambda,
        index_t& iter,
        index_t max_iter,
        index_t max_active,
        value_t thresh,
        BetaType& beta,
        value_t& r_sq,
        const XType& X)
    {
        auto& derived = this->self();

        derived.init_strong_set(curr_lambda, prev_lambda);
        while (1) {
            bool converged_all_kkt_met = false;
            auto upd_state = derived.update_active_set(
                converged_all_kkt_met, iter, max_iter,
                max_active, thresh, curr_lambda, beta, r_sq, X);
            if (upd_state == state_t::max_active_size_reached ||
                upd_state == state_t::max_iter_reached) { 
                return upd_state;
            }

            if (converged_all_kkt_met) break;

            derived.init_proposal();

            while (1) {
                auto state = derived.solve_proposal(
                    iter, max_iter, max_active, 
                    thresh, curr_lambda, beta, r_sq, X);
                if (state == state_t::max_iter_reached) return state;
                if (derived.check_kkt_update_strong_set(curr_lambda)) break;
            }

            if (derived.check_kkt_update_all(curr_lambda)) break;
        }

        return state_t::noop;
    }

protected:

    Derived& self() 
    { return static_cast<Derived&>(*this); }
    const Derived& self() const 
    { return static_cast<const Derived&>(*this); }
};

} // namespace details

/*
 * Fit using covariance method.
 */
template <class ResourceType>
struct ASFit<::glmnetpp::util::method_type::gaussian_cov,
             ResourceType>
    : details::ASFitBase<typename ResourceType::value_t,
                         typename ResourceType::index_t,
                         ::glmnetpp::util::elnet_state,
                         ASFit<::glmnetpp::util::method_type::gaussian_cov,
                               ResourceType> >
{
private:
    using base_t = details::ASFitBase<typename ResourceType::value_t,
                         typename ResourceType::index_t,
                         ::glmnetpp::util::elnet_state,
                         ASFit<::glmnetpp::util::method_type::gaussian_cov,
                               ResourceType> >;

public:
    using value_t = typename base_t::value_t;
    using index_t = typename base_t::index_t;
    using state_t = ::glmnetpp::util::elnet_state;

    ASFit(size_t p)
        : compressed_beta_(p)
        , rsrc_(p)
    {}

    // TODO: optimize (probably don't need clearing)!
    void init_strong_set(value_t curr_lambda,
                         value_t prev_lambda)
    {
        value_t thrsh = 2 * curr_lambda - prev_lambda; 
        rsrc_.clear_strong_set();
        for (index_t j = 0; j < rsrc_.n_coords(); ++j) {
            auto abs_grad_j = std::abs(rsrc_.grad(j));
            if (abs_grad_j >= thrsh) {
                rsrc_.update_strong_set(j);
            }
        }
    }

    template <class BetaType
            , class XType>
    state_t update_active_set(
        bool& converged_all_kkt_met,
        index_t& iter,
        index_t max_iter,
        index_t max_active,
        value_t thresh, 
        value_t lambda,
        BetaType& beta,
        value_t& r_sq,
        const XType& X)
    {
        while (1) {
            if (iter == max_iter) return state_t::max_iter_reached;
            ++iter;

            value_t max_abs_diff = 0;
            auto state = coord_descent_<0>(
                rsrc_.strong_begin(),
                rsrc_.strong_end(),
                max_active, lambda, beta, r_sq, X, max_abs_diff);

            if (state == state_t::max_active_size_reached) return state;

            // Slight optimization: if the one iteration converged,
            // check KKT for the complement of strong_set.
            // If KKT passes, we're done!
            // Otherwise, add to strong_set; most likely it was violated
            // because the strong_set hasn't converged fully yet,
            // so we incorrectly labeled the violator as not being in strong_set.
            if (convg_thresh_reached_(max_abs_diff, thresh)) {
                bool all_kkt_met = rsrc_.check_kkt(
                    util::counting_iterator<index_t>(0),
                    util::counting_iterator<index_t>(rsrc_.n_coords()),
                    [&](index_t j) { return rsrc_.is_strong(j); }, 
                    lambda,
                    [&](index_t j) { rsrc_.update_strong_set(j); });
                if (all_kkt_met) {
                    converged_all_kkt_met = true;
                    break;
                }
            }
            else break;
        }

        return state_t::noop;
    }

    void init_proposal() { rsrc_.init_proposal(); }

    template <class BetaType, class XType>
    state_t solve_proposal(
        index_t& iter,
        index_t max_iter,
        index_t max_active,
        value_t thresh,
        value_t lambda,
        BetaType& beta,
        value_t& r_sq,
        const XType& X)
    {
        // save old proposed beta
        index_t k = 0;
        for (auto it = rsrc_.proposal_begin();
             it != rsrc_.proposal_end();
             ++it, ++k)
        {
            auto j = *it;
            compressed_beta_[k] = beta[j];
        }

        // apply coordinate descents over proposal set
        while (1) {

            if (iter == max_iter) return state_t::max_iter_reached;
            ++iter;

            // update beta for predictors in proposal set
            value_t max_abs_diff = 0;
            coord_descent_<1>(
                rsrc_.proposal_begin(),
                rsrc_.proposal_end(),
                max_active, lambda, beta, r_sq, X, max_abs_diff);

            if (convg_thresh_reached_(max_abs_diff, thresh)) break;

        } // end for - coordinate descent

        // update gradient on all other non-proposed components
        k = 0;
        for (auto it = rsrc_.proposal_begin();
             it != rsrc_.proposal_end();
             ++it, ++k)
        {
            auto j = *it;
            compressed_beta_[k] -= beta[j];
        }

        // TODO: maybe optimize if proposal_set is small enough,
        // then just cache grad components for proposal and 
        // update grad entirely (vectorization), then replace at proposal.
        for (index_t j = 0; j < rsrc_.n_coords(); ++j) {
            if (rsrc_.is_proposal(j)) continue;
            for (index_t k = 0; static_cast<size_t>(k) < rsrc_.n_proposal(); ++k) {
                if (compressed_beta_[k] == 0) continue;
                rsrc_.update_invariant_compressed(j, k, compressed_beta_[k] / X.rows());
            }
        }

        return state_t::noop;
    }

    bool check_kkt_update_strong_set(value_t lambda)
    {
        // Check KKT on strong set (ignoring proposal set).
        return rsrc_.check_kkt(
            rsrc_.strong_begin(),
            rsrc_.strong_end(),
            [&](index_t j) { return rsrc_.is_proposal(j); },
            lambda,
            [&](index_t j) { rsrc_.update_proposal_set(j); } );
    }

    bool check_kkt_update_all(value_t lambda) 
    {
        // Check KKT on all coordinates (ignoring proposal set and strong set).
        return rsrc_.check_kkt(
            util::counting_iterator<index_t>(0),
            util::counting_iterator<index_t>(rsrc_.n_coords()),
            [&](index_t j) { return rsrc_.is_proposal(j) || rsrc_.is_strong(j); },
            lambda,
            [&](index_t j) { rsrc_.update_strong_set(j); } );
    }

    template <class XType, class YType>
    void init_grad(const XType& X, const YType& y)
    { rsrc_.init_grad(X, y); }

    auto lambda_max() const { return rsrc_.lambda_max(); }

    template <class BetaType>
    auto n_non_zero_beta(const BetaType& beta) const 
    { return rsrc_.n_non_zero_beta(beta); }

private:
    using resource_t = ResourceType;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;

    static bool convg_thresh_reached_(value_t v, value_t thresh) 
    { return v * v < thresh; }

    template <size_t version
            , class IterType
            , class BetaType
            , class XType>
    state_t coord_descent_(IterType begin,
                           IterType end,
                           index_t max_active,
                           value_t lambda,
                           BetaType& beta,
                           value_t& r_sq,
                           const XType& X,
                           value_t& max_abs_diff)
    {
        for (auto it = begin; it != end; ++it) 
        {
            auto j = *it;
            auto prev_beta_j = beta[j];

            beta[j] = rsrc_.next_beta(j, beta[j], lambda);

            if (prev_beta_j == beta[j]) continue;
            
            auto beta_diff = prev_beta_j - beta[j];

            // called from update_active_set
            if constexpr (version == 0) {
                if (!rsrc_.is_active(j)) {
                    if (rsrc_.n_active() == static_cast<size_t>(max_active)) {
                        return state_t::max_active_size_reached;
                    }
                    rsrc_.update_active_set(j);
                }
                rsrc_.update_invariant(j, beta_diff, X);
            }

            // called from solve_proposal
            else {
                rsrc_.update_invariant_proposal(j, beta_diff, X);
            }

            r_sq = rsrc_.next_r_sq(r_sq, j, beta_diff);
            max_abs_diff = std::max(max_abs_diff, std::abs(beta_diff));
        }

        return state_t::noop;
    }

    vec_t compressed_beta_;
    resource_t rsrc_; 
};

} // namespace core
} // namespace glmnetpp