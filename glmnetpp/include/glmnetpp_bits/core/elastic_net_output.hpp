#pragma once
#include <Eigen/Core>
#include <glmnetpp_bits/core/elastic_net_config.hpp>
#include <vector>

namespace glmnetpp {

struct ElasticNetOutput
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> beta;
    std::vector<double> lambda;
};

} // namespace glmnetpp
