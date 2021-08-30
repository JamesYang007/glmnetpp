#pragma once
#include <cstddef>
#include <climits>
#include <type_traits>
#include <cmath>

namespace glmnetpp {
namespace util {

/* 
 * Computes integer log_2.
 * For an integer-like input x, it computes j where 
 * 2^j <= x < 2^{j+1}
 * Undefined behavior for x <= 0.
 */
template <class IntType>
inline constexpr IntType ilog2(IntType x)
{
	constexpr int n_bits = sizeof(IntType) * CHAR_BIT;
    std::make_unsigned_t<IntType> ux = x;
	std::make_unsigned_t<IntType> bit_mask = 1;
	int i = 0;
	for (; i < n_bits; ++i) {
		if (bit_mask > ux) break;
		bit_mask <<= 1;
	}
	if (bit_mask == 0) return n_bits-1;
	return i-1;
}

// Computes the soft-threshold function:
// sign(z) * max(|z| - l, 0)
template <class T>
inline constexpr T soft_threshold(T z, T l)
{
	auto abs_z = std::abs(z);
	if (l >= abs_z) return 0.;
	return std::copysign(1., z) * (abs_z - l);
}

} // namespace util
} // namespace glmnetpp
