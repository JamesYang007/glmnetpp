#pragma once
#include <Eigen/Core>
#include <glmnetpp_bits/core/elastic_net_config.hpp>

namespace glmnetpp {
namespace core {

struct ElasticNetOutput
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> beta;
    ElasticNetConfig config;
};

} // namespace core
} // namespace glmnetpp
