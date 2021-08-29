#pragma once
#include <Eigen/Core>
#include <glmnetpp_bits/core/elastic_net_config.hpp>

namespace glmnetpp {

struct ElasticNetOutput
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> beta;

};

} // namespace glmnetpp
