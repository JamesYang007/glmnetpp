#pragma once
#include <Eigen/Core>

namespace glmnetpp {
namespace util {

/*
 * Creates an evenly-spaced (log scale) vector of values from start to stop and stores into dest.
 * Size of dest is num after the call.
 * num must be greater than 1 for this call to be well-defined.
 * dest may be resized.
 */
template <class VectorType, class T>
inline void geomspace(VectorType& dest,
                      T start, T stop, 
                      uint32_t num)
{
    assert(num > 1);
    dest.resize(num);

    auto log_factor = (std::log(stop) - std::log(start))/(num-1);
    dest[0] = 0;
#pragma omp simd
    for (int i = 1; i < dest.size(); ++i)
    {
        dest[i] = dest[i-1] + log_factor;
    }

    dest = dest.array().exp();
    dest *= start;
}

} // namespace util
} // namespace glmnetpp
