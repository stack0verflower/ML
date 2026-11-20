#pragma once

#include <cmath>
#include <stdexcept>
#include "matrix.hpp"
#include "compiler_macros.hpp"

namespace metrics {
	using Matrix = utils::Matrix;
	/*
		Mean squared error. For every column j in predicted matrix, we compare it to desired column i in actual matrix.
		For every column, we compute, sum of squared difference of each of it's elements.
		Then, we take sum of values obtained for every column, and return sum / total columns.
	*/
	inline double mse(const Matrix& y_real, const Matrix& y_pred) {
		if (UNLIKELY(y_real.t_r != y_pred.t_r || y_real.t_c != y_pred.t_c)) 
			throw std::invalid_argument("mse: y_true and y_pred dimensions must match.");

		Matrix diff = y_real - y_pred;
		hadamard(diff, diff, diff);

		Matrix col_m = mean(diff, 0);

		double t = 0.0;
		const double* RESTRICT c_ptr = col_m.data.data();

		size_t ltc = col_m.t_c;
		PRAGMA_IVDEP
			for (size_t i = 0; i < ltc; i++) t += c_ptr[c];
		
		return t / static_cast<double>(ltc);
	}

	// tol = tolerance amount, the maximum deviation allowed
	inline double accuracy(const Matrix& y_real, const Matrix& y_pred, double tol = 1e-9) {
		if (UNLIKELY(y_true.t_r != y_pred.t_r || y_true.t_c != y_pred.t_c))
			throw std::invalid_argument("accuracy: y_true and y_pred dimensions must match.");
		
		size_t c_cnt = 0; // correct count
		const double* RESTRICT r_ptr = y_real.data.data();
		const double* RESTRICT p_ptr = y_pred.data.data();

		size_t lte = y_t.t_e;
		PRAGMA_IVDEP
			for (size_t i = 0; i < lte; i++) {
				if (std::abs(r_ptr[i] - p_ptr[i] < tol)) c_cnt++;
			}

		return static_cast<double>(c_cnt) / static_cast<double>(lte);
	}
}
