#pragma once
#include <Eigen/Core>
#include <vector>
#include <glmnetpp_bits/util/typedefs.hpp>

namespace glmnetpp {
namespace core {

struct ElasticNetOutput
{
    using state_t = util::elnet_state;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> beta;
    std::vector<double> lambda;
    state_t state = state_t::noop;
};

} // namespace core
} // namespace glmnetpp
