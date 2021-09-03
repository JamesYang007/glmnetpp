#pragma once
#include <glmnetpp_bits/core/elastic_net_base.hpp>
#include <glmnetpp_bits/core/elastic_net.hpp>

namespace glmnetpp {
namespace core {

/*
 * Elastic net solver for Gaussian loss.
 * Optimization function:
 *
 * 1/(2n) * ||y - Xb||^2 + lmda * (alpha * ||b||_1 + (1-alpha)/2 ||b||_2^2)
 * 
 * Currently, the model assumes that y (n-vector) and X (n-by-p matrix) 
 * are centered and scaled such that ||y||^2 = ||x_j||^2 = n for all j = 1,...,p
 * and x_j are column vectors of X.
 *
 * TODO: 
 * - generalize to arbitrary weights.
 * - add a check that active set never exceeds a user-specified number of predictors.
 */
template <class UpdateResourceType>
struct ElasticNet<util::loss_type::gaussian,
                  UpdateResourceType>
    : ElasticNetBase<UpdateResourceType>
{

};

} // namespace core
} // namespace glmnetpp