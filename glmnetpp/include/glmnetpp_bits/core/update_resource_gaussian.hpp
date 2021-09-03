#pragma once
#include <glmnetpp_bits/core/elastic_net_output.hpp>
#include <glmnetpp_bits/core/update_resource.hpp>
#include <glmnetpp_bits/util/compressed_matrix.hpp>
#include <glmnetpp_bits/util/typedefs.hpp>
#include <Eigen/Core>
#include <vector>

namespace glmnetpp {
namespace core {
namespace details {

// Base for Gaussian update resource.
template <class ValueType>
struct UpdateResourceGaussianBase
{
    using value_t = ValueType;
    using index_t = ::glmnetpp::util::index_t;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;

    UpdateResourceGaussianBase(size_t p)
        : active_set_(p, false)
        , inv_active_set_{}
        , proposal_set_(p, false)
        , inv_proposal_set_{}
        , strong_set_(p, false)
        , inv_strong_set_{}
        , grad_()
    {
        inv_active_set_.reserve(p);
        inv_proposal_set_.reserve(p);
        inv_strong_set_.reserve(p);
    }

    void clear_strong_set()
    {
        inv_strong_set_.clear();
        std::fill(strong_set_.begin(), strong_set_.end(), false);
    }

    value_t next_beta(index_t j, value_t beta_j, value_t lambda)
    {
        // TODO: loo_grad should be computed using + ||x_j||^2 beta[j]
        auto loo_grad = grad_[j] + beta_j;
        return util::soft_threshold(loo_grad, lambda);
    }

    void init_proposal()
    {
        proposal_set_ = active_set_;
        inv_proposal_set_ = inv_active_set_;
    }
    
    template <class XType, class YType>
    void init_grad(const XType& X, const YType& y)
    {
        grad_ = (X.transpose() * y) / X.rows(); 
    }

    value_t lambda_max() const { return grad_.array().abs().maxCoeff(); }

    bool is_active(index_t j) const { return active_set_[j]; }
    bool is_strong(index_t j) const { return strong_set_[j]; }
    bool is_proposal(index_t j) const { return proposal_set_[j]; }
    auto strong_begin() { return inv_strong_set_.begin(); }
    auto strong_end() { return inv_strong_set_.end(); }
    auto proposal_begin() { return inv_proposal_set_.begin(); }
    auto proposal_end() { return inv_proposal_set_.end(); }
    auto n_coords() const { return active_set_.size(); }
    auto n_active() const { return inv_active_set_.size(); }
    auto n_proposal() const { return inv_proposal_set_.size(); }

    template <class BetaType>
    auto n_non_zero_beta(const BetaType& beta) const {
        index_t non_zero_count = 0;
        for (auto j : inv_active_set_) {
            if (beta[j] != 0) ++non_zero_count;
        }
        return non_zero_count;
    }

    auto grad(index_t j) const { return grad_[j]; }

    void update_active_set(index_t j)
    {
        inv_active_set_.push_back(j);
        active_set_[j] = true;
    }

    void update_strong_set(index_t j) 
    {
        inv_strong_set_.push_back(j);
        strong_set_[j] = true;
    }

    void update_proposal_set(index_t j)
    {
        inv_proposal_set_.push_back(j);
        proposal_set_[j] = true;
    }

    /*
     * Updates r_sq (R^2) using a gradient value at index j
     * and beta_diff = beta_old_j - beta_new_j.
     */
    value_t next_r_sq(value_t r_sq,
                      index_t j,
                      value_t beta_diff)
    {
        // TODO: should be -beta_diff * ||x_j||^2 at the end
        return r_sq - beta_diff * (2.0 * grad_[j] - beta_diff);
    }

    template <class IncludeSubsetIterType
            , class ExcludeActionType
            , class ActionType>
    bool check_kkt(IncludeSubsetIterType isub_begin,
                   IncludeSubsetIterType isub_end,
                   ExcludeActionType esub_f,
                   double lambda,
                   ActionType f) 
    {
        bool all_kkt_met = true;

        std::for_each(isub_begin, isub_end,
            [&, esub_f, lambda, f](auto j) {
                if (esub_f(j)) return;
                // TODO: check_kkt_lasso should probably change later to a lambda parameter
                bool kkt_met = (std::abs(grad_[j]) <= lambda);
                if (!kkt_met) {
                    all_kkt_met = false;
                    f(j);
                }
            });

        return all_kkt_met;
    }

protected:

    // contains whether index j has ever been active 
    std::vector<bool> active_set_;

    // contains at position k, the kth index that became active.
    std::vector<index_t> inv_active_set_;

    // similar definitions for the following:
    std::vector<bool> proposal_set_;
    std::vector<index_t> inv_proposal_set_;
    std::vector<bool> strong_set_;
    std::vector<index_t> inv_strong_set_;

    // current R^2
    value_t r_sq_ = 0;

    // current gradient
    vec_t grad_;
};

} // namespace details

// Specialization: Gaussian covariance method.
template <class ValueType>
struct UpdateResource<
    ::glmnetpp::util::method_type::gaussian_cov, 
    ValueType>
    : details::UpdateResourceGaussianBase<ValueType>
{
private:
    using base_t = details::UpdateResourceGaussianBase<ValueType>;

public:
    using typename base_t::value_t;
    using typename base_t::index_t;

    UpdateResource(size_t p)
        : base_t(p)
        , X_cov_(p, p)
    {}

    template <class XType>
    void update_invariant(index_t j,
                          value_t beta_diff,
                          const XType& X)
    {
        update_x_cov_(j, X);
        auto vec = X_cov_.col(j);

        // important to update gradient before updating curr_r_sq!
        // TODO: n should actually go away and be absorbed into vec.
        this->grad_ += vec * (beta_diff / X.rows());
    }

    template <class XType>
    void update_invariant_proposal(
        index_t j,
        value_t beta_diff,
        const XType& X)
    {
        update_x_cov_(j, X);
        auto vec = X_cov_.col(j);

        // important to update gradient before updating curr_r_sq!
        // TODO: n should actually go away and be absorbed into vec.
        auto beta_diff_scaled = (beta_diff / X.rows());
        for (auto j : this->inv_proposal_set_) {
            this->grad_[j] += vec[j] * beta_diff_scaled;
        }
    }

    void update_invariant_compressed(
        index_t j,
        index_t k,
        value_t beta_diff_scaled)
    {
        auto vec = X_cov_.col(this->inv_proposal_set_[k]);
        this->grad_[j] += vec(j) * beta_diff_scaled;
    }

private:

    template <class XType>
    void update_x_cov_(index_t idx,
                       const XType& X)
    {
        if (!X_cov_.is_set(idx)) {
            X_cov_.allocate(idx);
            X_cov_.col(idx) = X.transpose() * X.col(idx);
        }
    }

    util::CompressedMatrix<value_t> X_cov_; // caches X^t X
};

// Specialization: Gaussian naive method.
template <class ValueType>
struct UpdateResource<
    ::glmnetpp::util::method_type::gaussian_naive, 
    ValueType>
    : details::UpdateResourceGaussianBase<ValueType>
{
 
};

} // namespace core
} // namespace glmnetpp