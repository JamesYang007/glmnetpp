#include "gtest/gtest.h"
#include <glmnetpp_bits/util/math.hpp>
#include <tuple>

namespace glmnetpp {
namespace util {

template <class IntType>
struct ilog2_fixture :
	testing::TestWithParam<std::pair<IntType, IntType>>
{
	void SetUp() override
	{
		std::tie(in, expected) = this->GetParam();
	}

protected:
	using int_t = IntType;
	int_t in, expected;
	
	void test() 
	{
		int_t out = ilog2(in);
		EXPECT_EQ(out, expected);
	}
};

#define ILOG2_TEST_SUITE_GEN(name, type) \
	struct ilog2_##name##_fixture : ilog2_fixture<type>	{}; \
	TEST_P(ilog2_##name##_fixture, ilog2_out) { test(); } \
	INSTANTIATE_TEST_SUITE_P( \
		ilog2_suite, \
		ilog2_##name##_fixture, \
		testing::Values( \
			std::make_pair<int>( 1, 0 ),\
			std::make_pair<int>( 2, 1 ),\
			std::make_pair<int>( 3, 1 ),\
			std::make_pair<int>( 4, 2 ),\
			std::make_pair<int>( 5, 2 ),\
			std::make_pair<int>( 7, 2 ),\
			std::make_pair<int>( 8, 3 ),\
			std::make_pair<int>( 14, 3 ),\
			std::make_pair<int>( 257, 8 ),\
			std::make_pair<int>( 329, 8 ),\
			std::make_pair<int>( 511, 8 ),\
			std::make_pair<int>( 512, 9 )\
		))

ILOG2_TEST_SUITE_GEN(int, int);
ILOG2_TEST_SUITE_GEN(uint, unsigned int);
ILOG2_TEST_SUITE_GEN(uint64, uint64_t);
ILOG2_TEST_SUITE_GEN(size_t, size_t);

#undef ILOG2_TEST_SUITE_GEN


} // namespace util
} // namespace glmnetpp
