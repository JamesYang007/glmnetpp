#pragma once
#include <Eigen/Core>

namespace glmnetpp {
namespace util {

using index_t = Eigen::Index;
    
// Flags to indicate state of elastic net.
enum class elnet_state
{
    noop,
    max_iter_reached,
    max_active_size_reached,
    beta_no_change
};

// loss types
enum class loss_type
{
    gaussian 
};

// update method types
enum class method_type
{
    gaussian_naive,
    gaussian_cov
};

} // namespace util
} // namespace glmnetpp
