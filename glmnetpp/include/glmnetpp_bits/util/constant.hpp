#pragma once
#include <limits>

namespace glmnetpp {
namespace util {

inline constexpr double neg_inf =
    std::numeric_limits<double>::is_iec559 ?
    -std::numeric_limits<double>::infinity() :
    std::numeric_limits<double>::lowest();

} // namespace util
} // namespace glmnetpp
