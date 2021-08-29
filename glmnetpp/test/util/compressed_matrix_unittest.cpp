#include "gtest/gtest.h"
#include <glmnetpp_bits/util/compressed_matrix.hpp>
#include <memory>

namespace glmnetpp {
namespace util {

struct compressed_matrix_fixture : 
	testing::TestWithParam<std::tuple<size_t, size_t, size_t>>
{

	void SetUp() override 
	{
		std::tie(n_rows, n_cols, cap) = GetParam();
		matrix_ptr.reset(new matrix_t(n_rows, n_cols, cap));
	}

protected:
	using value_t = double;
	using matrix_t = CompressedMatrix<value_t>;
	std::unique_ptr<matrix_t> matrix_ptr;
	size_t n_rows, n_cols, cap;
};

#ifndef NDEBUG
TEST_P(compressed_matrix_fixture, matrix_zero_row)
{
	ASSERT_DEATH(matrix_t(0, n_cols, cap), 
				 "Assertion failed: n_rows > 0");
}
#endif

TEST_P(compressed_matrix_fixture, matrix_one_buffer_amount)
{
	size_t alloc_amt = cap;

	// if test case is not valid
	if (n_cols < alloc_amt) return;

	// allocate amount
	for (size_t i = 0; i < alloc_amt; ++i) {
		matrix_ptr->col(i);
	}

	// all allocation pointers should be n_rows apart
	auto [vec, was_stored_before] = matrix_ptr->col(0);
	if (alloc_amt == 0) EXPECT_FALSE(was_stored_before);
	else EXPECT_TRUE(was_stored_before);
	auto prev_ptr = vec.data();
	auto curr_ptr = prev_ptr;

	for (size_t i = 1; i < alloc_amt; ++i) {
		auto [vec, was_stored_before] = matrix_ptr->col(i);
		EXPECT_TRUE(was_stored_before);
		curr_ptr = vec.data();
		EXPECT_EQ(curr_ptr - prev_ptr, n_rows);
		prev_ptr = curr_ptr;
	}
}

// tests whether second buffer allocated the correct amount
TEST_P(compressed_matrix_fixture, matrix_two_buffer_amount)
{
	auto alloc_amt = 3 * cap;

	// if test case is not valid
	if (n_cols < alloc_amt) return;

	// allocate amount
	for (size_t i = 0; i < alloc_amt; ++i) {
		matrix_ptr->col(i);
	}

	// suffices to check starting from the second buffer 
	auto [vec, was_stored_before] = matrix_ptr->col(cap);
	if (alloc_amt == 0) EXPECT_FALSE(was_stored_before);
	else EXPECT_TRUE(was_stored_before);
	auto prev_ptr = vec.data();
	auto curr_ptr = prev_ptr;

	for (size_t i = cap+1; i < alloc_amt; ++i) {
		auto [vec, was_stored_before] = matrix_ptr->col(i);
		EXPECT_TRUE(was_stored_before);
		curr_ptr = vec.data();
		EXPECT_EQ(curr_ptr - prev_ptr, n_rows);
		prev_ptr = curr_ptr;
	}
}

INSTANTIATE_TEST_SUITE_P(
	CompressedMatrixAllocSuite,
	compressed_matrix_fixture,
	testing::Combine(
		testing::Values(1, 5, 23, 152),
		testing::Values(1, 5, 31, 58),
		testing::Values(0, 1, 5, 20)));

} // namespace util
} // namespace glmnetpp