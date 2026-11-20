#pragma once

#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <utility>
#include "matrix.hpp"
#include "compiler_macros.hpp"

namespace preprocess {
	using Matrix = utils::Matrix;
	/*
		We will split the given data into two sets, one to train and other to test.
		This function accepts X and y. X is the number of features or input variables. y is the computed/to be predicted label.
		The function will return a tuple of four matrices: X_train, X_test, y_train, y_test.
	*/
	struct SplitResult {
		Matrix X_train;
		Matrix X_test;
		Matrix y_train;
		Matrix y_test;
	};

	// We inline the only method (not inside a struct) in this hpp to avoid multiple copies when hpp is included multiple times.
	inline SplitResult train_test_split(const Matrix& X, const Matrix& y, double test_size = 0.2, uint32_t r_seed = 42) {
		if (UNLIKELY(X.t_r != y.t_r)) 
			throw std::invalid_argument("Number of rows in X and y must be the same.");

		const size_t n_samples = X.t_r;
		const size_t n_test = static_cast<size_t>(n_samples * test_size);
		const size_t n_train = n_samples - n_test;

		std::vector<size_t> ind(n_samples);
		std::iota(ind, begin(), ind.end(), 0);

		std::mt19937 rng(r_seed);
		std::shuffle(ind.begin(), ind.end(), rng);

		Matrix X_train(n_train, X.t_c);
		Matrix X_test(n_test, X.t_c);
		Matrix y_train(n_train, y.t_c);
		Matrix y_test(n_test, y.t_c);

		/*
			ind has shuffled indices of rows. For 10 rows, it contains something like
				7, 3, 4, 9, 0, 8, 5, 6, 1, 2
			X_train takes first n_train rows of X, same with y_train for y.
			Similarly for X_test, y_test
		*/

		// std::copy_n(source_ptr, count, dest_ptr), contiguous bulk memory copying (memcpy).
		for (size_t i = 0; i < n_train; i++) {
			const size_t src_row = ind[i];
			std::copy_n(&X.data[src_row * X.t_c], X.t_c, &X_train.data[i * X.t_c]);
			std::copy_n(&y.data[src_row * y.t_c], y.t_c, &y_train.data[i * y.t_c]);
		}

		for (size_t i = 0; i < n_test; i++) {
			const size_t src_row = ind[n_train + i];
			std::copy_n(&X.data[src_row * X.t_c], X.t_c, &X_test.data[i * X.t_c]);
			std::copy_n(&y.data[src_row * y.t_c], y.t_c, &y_test.data[i * y.t_c]);
		}

		return { std::move(X_train), std::move(X_test), std::move(y_train), std::move(y_test) };
	}

	/*
		For each feature column j, compute its mean uj and standard deviation sj, then transform every entry, or convert to standard normal
		distribution by subtracting the mean and dividing by the standard deviation:
			
			Z = (X - u) / s

		Thus, collapse along column to compute mean.
	*/
	struct StandardScaler {
		Matrix mn_vc;
		Matrix std_vc;

		void fit(const Matrix& x) {
			mn_vc = mean(x, 0);

			// Get X - u, for every element in column j, subtract mean uj
			Matrix sq_diff(x.t_r, x.t_c);
			const double* RESTRICT x_ptr = x.data.data();
			const double* RESTRICT mn_ptr = mn_vc.data.data();
			double* RESTRICT diff_ptr = sq_diff.data.data();

			for (size_t r = 0; r < x.t_r; ++r) {
				const size_t offset = r * x.t_c;
				PRAGMA_IVDEP
					for (size_t c = 0; c < x.t_c; ++c) diff_ptr[offset + c] = x_ptr[offset + c] - mn_ptr[c];
			}

			// Compute (X - u)^2
			hadamard(sq_diff, sq_diff, sq_diff);
			std_vc = mean(sq_diff, 0);
			size_t lt_e = std_vc.t_e;
			for (size_t i = 0; i < lt_e; ++i) {
				double s = std::sqrt(std_vc.data[i]);
				std_vc.data[i] = (s > 1e-12) ? s : 1.0;
			}
		}

		// Transform the data using the fitted mean and standard deviation
		// For every element in column j, subtract mean uj and divide by standard deviation sj, to convert to standard normal distribution.
		Matrix transform(const Matrix& x) const {
			if (UNLIKELY(x.t_c != mn_vc.t_c || x.t_c != std_vc.t_c)) 
				throw std::invalid_argument("Input matrix must have the same number of columns as the fitted mean and standard deviation vectors.");
			if (UNLIKELY(mn_vc.t_r != 1 || std_vc.t_r != 1)) 
				throw std::invalid_argument("Mean and standard deviation vectors must be row vectors.");

			Matrix res(x.t_r, x.t_c);
			const double* RESTRICT x_ptr = x.data.data();
			const double* RESTRICT mn_ptr = mn_vc.data.data();
			const double* RESTRICT std_ptr = std_vc.data.data();
			double* RESTRICT res_ptr = res.data.data();

			for (size_t r = 0; r < x.t_r; ++r) {
				const size_t offset = r * x.t_c;
				PRAGMA_IVDEP
					for (size_t c = 0; c < x.t_c; ++c) res_ptr[offset + c] = (x_ptr[offset + c] - mn_ptr[c]) / std_ptr[c];
			}

			return res;
		}

		Matrix fit_transform(const Matrix& x) {
			fit(x);
			return transform(x);
		}
 	};

	/*
		This is min-max scaling, which rescales the features to a fixed range, usually [0, 1]. The formula is:
		x(i, j) = (x(i, j) - min_j) / (max_j - min_j)
		where min_j and max_j are the minimum and maximum values of feature j, respectively. (feature j is simply column j of the matrix)
	*/
	struct MinMaxScaler {
		Matrix min_vc;
		Matrix rng_vc;

		void fit(const Matrix& x) {
			min_vc = min(x, 0);
			Matrix max_vc = max(x, 0);
			rng_vc = max_vc;
			for (size_t i = 0; i < rng_vc.t_e; ++i) {
				double range = rng_vc.data[i] - min_vc.data[i];
				rng_vc.data[i] = (range > 1e-12) ? range : 1.0;
			}
		}

		Matrix transform(const Matrix& x) const {
			if (UNLIKELY(x.t_c != min_vc.t_c || x.t_c != rng_vc.t_c)) 
				throw std::invalid_argument("Input matrix must have the same number of columns as the fitted min and range vectors.");
			if (UNLIKELY(min_vc.t_r != 1 || rng_vc.t_r != 1)) 
				throw std::invalid_argument("Min and range vectors must be row vectors.");

			Matrix res(x.t_r, x.t_c);
			const double* RESTRICT x_ptr = x.data.data();
			const double* RESTRICT min_ptr = min_vc.data.data();
			const double* RESTRICT rng_ptr = rng_vc.data.data();
			double* RESTRICT res_ptr = res.data.data();
			for (size_t r = 0; r < x.t_r; ++r) {
				const size_t offset = r * x.t_c;
				PRAGMA_IVDEP
					for (size_t c = 0; c < x.t_c; ++c) res_ptr[offset + c] = (x_ptr[offset + c] - min_ptr[c]) / rng_ptr[c];
			}
			return res;
		}

		Matrix fit_transform(const Matrix& x) {
			fit(x);
			return transform(x);
		}
	};
}
