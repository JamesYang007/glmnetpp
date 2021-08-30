#pragma once
#include <Eigen/Core>
#include <vector>
#include <cassert>
#include <cmath>
#include <memory>
#include <glmnetpp_bits/util/math.hpp>

namespace glmnetpp {
namespace util {
namespace details {

/*
 * This is a custom allocator specifically for CompressedMatrix.
 * For now, it does not need to adhere to std::allocator<T> interface,
 * so we make this a separate class.
 */
template <class ValueType>
class CompressedMatrixAlloc
{
public:
	using value_type = ValueType;

	CompressedMatrixAlloc(size_t n_rows,
						  size_t max_cols,
						  size_t cap,
						  size_t init_pool_size)
		: buffer_pool_()
		, n_rows_{n_rows}
		, n_cols_{0}
		, curr_idx_{0}
		, col_capacity_{std::max(cap, static_cast<size_t>(1))}
		, curr_col_size_{col_capacity_}
		, max_cols_{max_cols}
	{
		assert(n_rows > 0);
		buffer_pool_.reserve(init_pool_size);
		buffer_pool_.emplace_back(new value_type[curr_col_size_ * n_rows_]);
	}

	value_type* allocate()
	{
		++n_cols_;
		if (n_cols_ > col_capacity_) {
			curr_col_size_ *= 2;
			// if doubling current col size is too much
			if (col_capacity_ + curr_col_size_ > max_cols_) {
				curr_col_size_ = max_cols_ - col_capacity_;
			}
			buffer_pool_.emplace_back(new value_type[curr_col_size_ * n_rows_]);
			curr_idx_ = 0;
			col_capacity_ += curr_col_size_;
		}
		value_type* data_ptr = buffer_pool_.back().get() + curr_idx_ * n_rows_;
		++curr_idx_;
		return data_ptr;
	}

	auto rows() const { return n_rows_; }

private:
	std::vector<std::unique_ptr<value_type>> buffer_pool_; // each element is a pointer to a buffer
	const size_t n_rows_;		// column vector length.
	size_t n_cols_;				// number of occupied columns.
	size_t curr_idx_;			// current index to the next available part of buffer.
	size_t col_capacity_;		// maximum number of columns that can be added with current allocations.
	size_t curr_col_size_;		// current largest buffer size.
	size_t max_cols_;			// maximum number of columns ever.
};

} // namespace details

/*
 * This class represents a matrix where the column vectors
 * are compressed into a semi-contiguous buffer in the order that the user first initializes them.
 * It should not be used like an Eigen matrix to perform matrix operations as the memory is not fully contiguous.
 * However, each column vector is guaranteed to lie in a contiguous chunk, 
 * so one could view a column vector as an Eigen vector class and perform matrix operations.
 * It uses a custom allocator that amortizes cache misses while saving a lot of memory than
 * constructing a naive m by n matrix.
 * The matrix currently cannot be resized once initialized.
 */
template <class ValueType
		, class IndexType = size_t>
class CompressedMatrix
{
public:	
	using value_type = ValueType;
	using index_type = IndexType;

private:
	using alloc_t = details::CompressedMatrixAlloc<value_type>;
	using vec_t = Eigen::Matrix<value_type, Eigen::Dynamic, 1>;

public:

	CompressedMatrix(size_t n_rows,
					 size_t n_cols,
					 size_t init_cols = 16)
		: alloc_(n_rows, 
				 n_cols,
				 init_cols,
				 static_cast<size_t>(std::log2(n_cols)))
		, col_to_ptr_(n_cols, nullptr)
	{}

	value_type* allocate(index_type c)
	{
		auto ptr = col_to_ptr_[c];
		if (ptr) return ptr;
		return (col_to_ptr_[c] = alloc_.allocate());
	}

	Eigen::Map<vec_t> col(index_type c) const
	{
		return Eigen::Map<vec_t>(col_to_ptr_[c], alloc_.rows());
	}

	bool is_set(index_type c) const { return col_to_ptr_[c] != nullptr; }
	
private:
	alloc_t alloc_;
	// index j contains a pointer to its location in allocated memory 
	std::vector<value_type*> col_to_ptr_;	
};

} // namespace util
} // namespace glmnetpp
