#include "matrix.hpp"
#include "compiler_macros.hpp"
#include <stdexcept>
#include <algorithm>
#include <random>

namespace utlis {
	Matrix Matrix::operator+(const Matrix& other) const {
		if (UNLIKELY(t_r != other.t_r || t_c != other.t_c))
			throw std::invalid_argument("Matrix dimensions must match for addition.");
		
		Matrix result(t_r, t_c);
		
		double* RESTRICT res = result.data.data();
		const double* RESTRICT lhs = this->data.data();
		const double* RESTRICT rhs = other.data.data();

		PRAGMA_IVDEP
			for (size_t i = 0; i < t_e; ++i) res[i] = lhs[i] + rhs[i];
		return result;
	}

	Matrix& Matrix::operator+=(const Matrix& other) {
		if (UNLIKELY(t_r != other.t_r || t_c != other.t_c)) 
			throw std::invalid_argument("Matrix dimensions must match for addition.");

		double* RESTRICT lhs = this->data.data();
		const double* RESTRICT rhs = other.data.data();

		PRAGMA_IVDEP
			for (size_t i = 0; i < t_e; ++i) lhs[i] = lhs[i] + rhs[i];
		return *this;
	}

	Matrix Matrix::operator-(const Matrix& other) const {
		if (UNLIKELY(t_r != other.t_r || t_c != other.t_c)) 
			throw std::invalid_argument("Matrix dimensions must match for subtraction.");
		
		Matrix result(t_r, t_c);

		double* RESTRICT res = result.data.data();
		const double* RESTRICT lhs = this->data.data();
		const double* RESTRICT rhs = other.data.data();

		PRAGMA_IVDEP
			for (size_t i = 0; i < t_e; ++i) res[i] = lhs[i] - rhs[i];
		return result;
	}

	Matrix Matrix::operator-() const {
		Matrix result(t_r, t_c);

		double* RESTRICT res = result.data.data();
		const double* RESTRICT lhs = this->data.data();

		PRAGMA_IVDEP
			for (size_t i = 0; i < t_e; ++i) res[i] = -lhs[i];
		return result;
	}

	Matrix& Matrix::operator-=(const Matrix& other) {
		if (UNLIKELY(t_r != other.t_r || t_c != other.t_c)) 
			throw std::invalid_argument("Matrix dimensions must match for subtraction.");

		double* RESTRICT lhs = this->data.data();
		const double* RESTRICT rhs = other.data.data();

		PRAGMA_IVDEP
			for (size_t i = 0; i < t_e; ++i) lhs[i] = lhs[i] - rhs[i];
		return *this;
	}

	Matrix Matrix::operator*(const Matrix& other) const {
		if (UNLIKELY(t_c != other.t_r)) 
			throw std::invalid_argument("Matrix dimensions must match for multiplication.");

		Matrix result(t_r, other.t_c);

		double* RESTRICT res = result.data.data();
		const double* RESTRICT lhs = this->data.data();
		const double* RESTRICT rhs = other.data.data();

		constexpr size_t b_size = L1_CACHE_BLOCK_SIZE;
		const size_t M = t_r;
		const size_t K = t_c;
		const size_t N = other.t_c;

		for (size_t io = 0; io < t_r; io++) {
			const size_t i_e = std::min(io + b_size, t_r);

			for (size_t ko = 0; ko < t_c; ko++) {
				const size_t k_e = std::min(ko + b_size, t_c);

				for (size_t jo = 0; jo < other.t_c; jo++) {
					const size_t j_e = std::min(jo + b_size, other.t_c);

					for (size_t i = io; i < i_e; i++) {
						const size_t c_row = i * N;
						const size_t a_row = i * K;

						for (size_t k = ko; k < k_e; k++) {
							const double a_ik = lhs[a_row + k];
							const size_t b_row = k * N;

							PRAGMA_IVDEP
								for (size_t j = jo; j < j_e; j++) res[c_row + j] += a_ik * rhs[b_row + j];
						}
					}
				}
			}
		}
		return result;
	}

	// Custom implementation of in-place is messy, as we will need to resize dimensions and copy data.
	// This defeats my purpose of writing custom, as there, I need not worry about copy overload.
	Matrix& Matrix::operator*=(const Matrix& other) {
		*this = *this * other;
		return *this;
	}

	Matrix Matrix::transpose() const {
		Matrix result(t_c, t_r);

		double* RESTRICT res = result.data.data();
		const double* RESTRICT lhs = this->data.data();

		constexpr size_t b_size = L1_CACHE_BLOCK_SIZE;
		
		for (size_t ro = 0; ro < t_r; ro += b_size) {
			for (size_t co = 0; co < t_c; co += b_size) {
				const size_t r_e = std::min(ro + b_size, t_r);
				const size_t c_e = std::min(co + b_size, t_c);

				for (size_t r = ro; r < r_e; ++r) 
					PRAGMA_IVDEP
						for (size_t c = co; c < c_e; ++c)
							res[c * t_r + r] = lhs[r * t_c + c];
			}
		}
		return result;
	}

	Matrix Matrix::operator+(const double other) const {
		if (UNLIKELY(t_c != t_r))
			throw std::invalid_argument("Matrix must be square for scalar addition/subtraction.");

		Matrix result(*this);

		double* RESTRICT res = result.data.data();
		const size_t jmp = t_c + 1;

		for (size_t i = 0; i < t_r; ++i) res[i * jmp] += other;
		return result;
	}

	Matrix& Matrix::operator+=(const double other) {
		if (UNLIKELY(t_c != t_r))
			throw std::invalid_argument("Matrix must be square for scalar addition/subtraction.");

		double* RESTRICT lhs = this->data.data();
		const size_t jmp = t_c + 1;

		for (size_t i = 0; i < t_r; ++i) lhs[i * jmp] += other;
		return *this;
	}
	
	Matrix Matrix::operator-(const double other) const {
		return *this + (-other);
	}

	Matrix& Matrix::operator-=(const double other) {
		return *this += -other;
	}

	Matrix Matrix::operator*(const double other) const {
		Matrix result(t_r, t_c);
		const size_t t_e = t_r * t_c;

		double* RESTRICT res = result.data.data();
		const double* RESTRICT lhs = this->data.data();

		PRAGMA_IVDEP
			for (size_t i = 0; i < t_e; ++i) res[i] = lhs[i] * other;
		return result;
	}

	Matrix& Matrix::operator*=(const double other) {
		const size_t t_e = t_r * t_c;
		double* RESTRICT lhs = this->data.data();

		PRAGMA_IVDEP
			for (size_t i = 0; i < t_e; ++i) lhs[i] *= other;
		return *this;
	}

	Matrix Matrix::operator/(const double other) const {
		if (UNLIKELY(other == 0.0))
			throw std::invalid_argument("Division by zero in matrix scalar division.");
		
		double inv = 1.0 / other;
		return *this * inv;
	}

	Matrix& Matrix::operator/=(const double other) {
		if (UNLIKELY(other == 0.0)) 
			throw std::invalid_argument("Division by zero in matrix scalar division.");
		
		double inv = 1.0 / other;
		return *this *= inv;
	}

	double sum(const Matrix& A) {
		const size_t lt_e = A.t_e;
		const double* RESTRICT a = A.data.data();

		double res = 0.0;
		PRAGMA_IVDEP
			for (size_t i = 0; i < lt_e; ++i) res += a[i];
		return res;
	}

	Matrix sum(const Matrix& A, int axis) {
		if (UNLIKELY(axis != 0 && axis != 1)) 
			throw std::invalid_argument("Axis must be 0 (columns) or 1 (rows).");

		if (axis == 0) {
			Matrix result(1, A.t_c);
			double* RESTRICT res = result.data.data();
			const double* RESTRICT a = A.data.data();
			
			size_t r_ul = A.t_r;
			size_t c_ul = A.t_c;
			for (size_t r = 0; r < r_ul; ++r) {
				const size_t offset = r * A.t_c;
				PRAGMA_IVDEP
					for (size_t c = 0; c < c_ul; ++c) res[c] += a[offset + c];
			}
			return result;
		} else {
			Matrix result(A.t_r, 1);
			double* RESTRICT res = result.data.data();
			const double* RESTRICT a = A.data.data();

			size_t r_ul = A.t_r;
			size_t c_ul = A.t_c;
			for (size_t r = 0; r < r_ul; ++r) {
				const size_t offset = r * A.t_c;
				double r_sum = 0.0;
				PRAGMA_IVDEP
					for (size_t c = 0; c < c_ul; ++c) r_sum += a[offset + c];
				res[r] = r_sum;
			}
			return result;
		}
	}

	double mean(const Matrix& A) {
		if (UNLIKELY(A.t_e == 0)) 
			throw std::invalid_argument("Cannot compute mean of an empty matrix.");

		return sum(A) / static_cast<double>(A.t_e);
	}

	Matrix mean(const Matrix& A, int axis) {
		if (UNLIKELY(axis != 0 && axis != 1)) 
			throw std::invalid_argument("Axis must be 0 (columns) or 1 (rows).");

		Matrix num = sum(A, axis);
		return (axis == 0) ? num / static_cast<double>(A.t_r) : num / static_cast<double>(A.t_c);
	}

	double max(const Matrix& A) {
		if (UNLIKELY(A.t_e == 0)) 
			throw std::invalid_argument("Cannot compute max of an empty matrix.");

		const size_t lt_e = A.t_e;
		const double* RESTRICT a = A.data.data();
		double res = A.data[0];

		PRAGMA_IVDEP
			for (size_t i = 1; i < lt_e; ++i) res = (a[i] > res) ? a[i] : res;
		return res;
	}

	Matrix max(const Matrix& A, int axis) {
		if (UNLIKELY(axis != 0 && axis != 1)) 
			throw std::invalid_argument("Axis must be 0 (columns) or 1 (rows).");

		if (UNLIKELY(A.t_e == 0)) 
			throw std::invalid_argument("Cannot compute max of an empty matrix.");

		if (axis == 1) {
			Matrix result(A.t_r, 1);
			double* RESTRICT res = result.data.data();
			const double* RESTRICT a = A.data.data();

			for (size_t r = 0; r < A.t_r; ++r) {
				const size_t offset = r * A.t_c;
				double r_max = a[offset];

				PRAGMA_IVDEP
					for (size_t c = 1; c < A.t_c; ++c) {
						const double val = a[offset + c];
						if (val > r_max) r_max = val;
					}
				res[r] = r_max;
			}
			return result;
		} else {
			Matrix result(1, A.t_c);
			double* RESTRICT res = result.data.data();
			const double* RESTRICT a = A.data.data();

			for (size_t c = 0; c < A.t_c; ++c) res[c] = a[c];

			for (size_t r = 1; r < A.t_r; ++r) {
				const size_t offset = r * A.t_c;

				PRAGMA_IVDEP
					for (size_t c = 0; c < A.t_c; ++c) {
						const double val = a[offset + c];
						if (val > res[c]) res[c] = val;
					}
			}
			return result;
		}
	}

	double min(const Matrix& A) {
		if (UNLIKELY(A.t_e == 0))
			throw std::invalid_argument("Cannot compute min of an empty matrix.");

		const size_t lt_e = A.t_e;
		const double* RESTRICT a = A.data.data();
		double res = A.data[0];
		PRAGMA_IVDEP
			for (size_t i = 1; i < lt_e; ++i) res = (a[i] < res) ? a[i] : res;
		return res;
	}

	Matrix min(const Matrix& A, int axis) {
		if (UNLIKELY(axis != 0 && axis != 1))
			throw std::invalid_argument("Axis must be 0 (columns) or 1 (rows).");

		if (UNLIKELY(A.t_e == 0))
			throw std::invalid_argument("Cannot compute min of an empty matrix.");

		if (axis == 1) {
			Matrix result(A.t_r, 1);
			double* RESTRICT res = result.data.data();
			const double* RESTRICT a = A.data.data();

			for (size_t r = 0; r < A.t_r; ++r) {
				const size_t offset = r * A.t_c;
				double r_min = a[offset];

				PRAGMA_IVDEP
					for (size_t c = 1; c < A.t_c; ++c) {
						const double val = a[offset + c];
						if (val > r_min) r_min = val;
					}
				res[r] = r_min;
			}
			return result;
		} else {
			Matrix result(1, A.t_c);
			double* RESTRICT res = result.data.data();
			const double* RESTRICT a = A.data.data();

			for (size_t c = 0; c < A.t_c; ++c) res[c] = a[c];
			for (size_t r = 1; r < A.t_r; ++r) {
				const size_t offset = r * A.t_c;

				PRAGMA_IVDEP
					for (size_t c = 0; c < A.t_c; ++c) {
						const double val = a[offset + c];
						if (val > res[c]) res[c] = val;
					}
			}
			return result;
		}
	}

	void hadamard(const Matrix& A, const Matrix& B, Matrix& dest) {
		if (UNLIKELY(A.t_r != B.t_r || A.t_c != B.t_c || A.t_r != dest.t_r || A.t_c != dest.t_c))
			throw std::invalid_argument("Matrix dimensions must match for Hadamard product.");

		const size_t lt_e = A.t_e;
		double* d = dest.data.data();
		const double* RESTRICT a = A.data.data();
		const double* RESTRICT b = B.data.data();

		PRAGMA_IVDEP
			for (size_t i = 0; i < lt_e; ++i) d[i] = a[i] * b[i];
	}

	void add_broadcast(const Matrix& A, const Matrix& B, Matrix& dest) {
		if (A.t_r == B.t_r && A.t_c == B.t_c) {
			if (UNLIKELY(dest.t_r != A.t_r || dest.t_c != A.t_c)) 
				throw std::invalid_argument("Destination shape mismatch.");
			
			const size_t total = A.t_e;
			double* d = dest.data.data();
			const double* RESTRICT a = A.data.data();
			const double* RESTRICT b = B.data.data();

			PRAGMA_IVDEP
				for (size_t i = 0; i < total; ++i) d[i] = a[i] + b[i];
			return;
		}

		double* d = dest.data.data();
		const double* RESTRICT a = A.data.data();
		const double* RESTRICT b = B.data.data();

		if (B.t_r == 1 && B.t_c == A.t_c) {
			if (UNLIKELY(dest.t_r != A.t_r || dest.t_c != A.t_c)) 
				throw std::invalid_argument("Destination shape mismatch.");

			for (size_t r = 0; r < A.t_r; ++r) {
				const size_t offset = r * A.t_c;
				PRAGMA_IVDEP
					for (size_t c = 0; c < A.t_c; ++c) d[offset + c] = a[offset + c] + b[c];
			}
			return;
		}

		if (B.t_c == 1 && B.t_r == A.t_r) {
			if (UNLIKELY(dest.t_r != A.t_r || dest.t_c != A.t_c))
				throw std::invalid_argument("Destination shape mismatch.");

			for (size_t r = 0; r < A.t_r; ++r) {
				const size_t offset = r * A.t_c;
				const double b_val = b[r];
				PRAGMA_IVDEP
					for (size_t c = 0; c < A.t_c; ++c) d[offset + c] = a[offset + c] + b_val;
			}
			return;
		}

		throw std::invalid_argument("Incompatible shapes for broadcasting addition.");
	}

	Matrix add_broadcast(const Matrix& A, const Matrix& B) {
		Matrix result(A.t_r, A.t_c);
		add_broadcast(A, B, result);
		return result;
	}

	void mul_broadcast(const Matrix& A, const Matrix& B, Matrix& dest) {
		if (A.t_r == B.t_r && A.t_c == B.t_c) {
			hadamard(A, B, dest);
			return;
		}

		double* d = dest.data.data();
		const double* RESTRICT a = A.data.data();
		const double* RESTRICT b = B.data.data();

		if (B.t_r == 1 && B.t_c == A.t_c) {
			if (UNLIKELY(dest.t_r != A.t_r || dest.t_c != A.t_c))
				throw std::invalid_argument("Destination shape mismatch.");

			for (size_t r = 0; r < A.t_r; ++r) {
				const size_t offset = r * A.t_c;
				PRAGMA_IVDEP
					for (size_t c = 0; c < A.t_c; ++c) d[offset + c] = a[offset + c] * b[c];
			}
			return;
		}

		if (B.t_c == 1 && B.t_r == A.t_r) {
			if (UNLIKELY(dest.t_r != A.t_r || dest.t_c != A.t_c))
				throw std::invalid_argument("Destination shape mismatch.");

			for (size_t r = 0; r < A.t_r; ++r) {
				const size_t offset = r * A.t_c;
				const double b_val = b[r];
				PRAGMA_IVDEP
					for (size_t c = 0; c < A.t_c; ++c) d[offset + c] = a[offset + c] * b_val;
			}
			return;
		}

		throw std::invalid_argument("Incompatible shapes for broadcasting addition.");
	}

	Matrix mul_broadcast(const Matrix& A, const Matrix& B) {
		Matrix result(A.t_r, A.t_c);
		mul_broadcast(A, B, result);
		return result;
	}

	Matrix I(size_t n) {
		Matrix result(n, n);
		double* RESTRICT res = result.data.data();
		
		const size_t jmp = n + 1;
		for (size_t i = 0; i < n; ++i) res[i * jmp] = 1.0;
		return result;
	}

	Matrix Uniform(size_t r, size_t c, double min, double max) {
		if (UNLIKELY(min >= max)) 
			throw std::invalid_argument("Minimum value must be less than maximum value for uniform distribution.");

		Matrix result(r, c);
		size_t tle = result.t_e;
		double* RESTRICT res = result.data.data();

		static thread_local std::mt19937_64 gen(std::random_device{}());
		std::uniform_real_distribution<double> dis(min, max);

		/*
			To compute random number i+1, the Mersenne Twister algorithm must know the internal 2.5 KB state
			updated by step i. Because every step depends strictly on the output of the preceding step,
			the CPU cannot compute 4 or 8 states simultaneously using standard auto-vectorization (PRAGMA_IVDEP fails here).
		*/

		for (size_t i = 0; i < tle; ++i) res[i] = dis(gen);
		return result;
	}

	Matrix Normal(size_t r, size_t c, double mean, double stddev) {
		if (UNLIKELY(stddev <= 0.0)) 
			throw std::invalid_argument("Standard deviation must be strictly positive for normal distribution.");
		
		Matrix result(r, c);
		size_t tle = result.t_e;
		double* RESTRICT res = result.data.data();

		static thread_local std::mt19937_64 gen(std::random_device{}());
		std::normal_distribution<double> dis(mean, stddev);

		for (size_t i = 0; i < tle; ++i) res[i] = dis(gen);
		return result;
	}

	void Inverse(const Matrix& A, Matrix& dest) {
		if (UNLIKELY(A.t_r != A.t_c))
			throw std::invalid_argument("Matrix must be square for inversion.");

		const size_t n = A.t_r;
		const size_t jmp = 2 * n;

		std::vector<double> aug(n * jmp, 0.0);
		double* RESTRICT a = aug.data();

		double m_entry = A.data[0];
		for (size_t r = 0; r < n; ++r) {
			size_t off = r * jmp;
			size_t a_off = r * n;
			for (size_t c = 0; c < n; ++c) {
				const double val = A.data[a_off + c];
				a[off + c] = val;
				const double abs_val = std::abs(val);
				if (abs_val > m_entry) m_entry = abs_val;
			}
			a[off + n + r] = 1.0;  // Identity matrix
		}

		if (m_entry <= 1e-12) 
			throw std::runtime_error("Matrix is singular (det == 0).");

		const double tol = 1e-12 * m_entry;

		for (size_t c = 0; c < n; ++c) {
			size_t cr = c;
			double m_abs_v = 0.0;
			for (size_t r = c; r < n; ++r) {
				const double abs_val = std::abs(a[r * jmp + c]);
				if (abs_val > m_abs_v) {
					m_abs_v = abs_val;
					cr = r;
				}
			}

			if (m_abs_v <= tol) 
				throw std::runtime_error("Matrix is singular (det == 0).");

			if (cr != c) {
				double* RESTRICT r_c = &a[c * jmp];
				double* RESTRICT r_cr = &a[cr * jmp];
				for (size_t j = 0; j < jmp; ++j) std::swap(r_c[j], r_cr[j]);
			}

			const double pvt = a[c * jmp + c];
			const double inv_pvt = 1.0 / pvt;

			a[c * jmp + c] = 1.0;
			double* RESTRICT r_c = &a[c * jmp];

			PRAGMA_IVDEP
				for (size_t j = c + 1; j < jmp; ++j) r_c[j] *= inv_pvt;

			for (size_t r = 0; r < n; ++r) {
				if (r == c) continue;
				double f = a[r * jmp + c];
				double* RESTRICT r_r = &a[r * jmp];
				PRAGMA_IVDEP
					for (size_t j = 0; j < jmp; ++j) r_r[j] -= f * r_c[j];
			}
		}

		double* RESTRICT d = dest.data.data();
		for (size_t r = 0; r < n; ++r) {
			const double* RESTRICT r_aug = &a[r * jmp + n];
			double* RESTRICT r_dest = &d[r * n];

			PRAGMA_IVDEP
				for (size_t c_idx = 0; c_idx < n; ++c_idx) r_dest[c_idx] = r_aug[c_idx];
		}
	}

	double det(Matrix A) {
		if (UNLIKELY(A.t_r != A.t_c)) 
			throw std::invalid_argument("Matrix must be square for determinant calculation.");

		const size_t n = A.t_r;
		if (UNLIKELY(n == 0)) 
			throw std::invalid_argument("Cannot compute determinant of an empty matrix.");

		if (n == 1) return A.data[0];
		if (n == 2) return A.data[0] * A.data[3] - A.data[1] * A.data[2];

		double* RESTRICT m = A.data.data();
		double m_entry = A.data[0];
		for (size_t r = 0; r < n; ++r) {
			const double abs_val = std::abs(m[r * n + r]);
			if (abs_val > m_entry) m_entry = abs_val;
		}

		const double tol_lm = 1e-12 * m_entry;
		const double tol = tol_lm > 1e15 ? 1e15 : tol_lm;

		double res = 1.0;
		for (size_t c = 0; c < n; ++c) {
			size_t cr = c;
			double m_abs_v = 0.0;
			for (size_t r = c; r < n; ++r) {
				const double abs_val = std::abs(m[r * n + c]);
				if (abs_val > m_abs_v) {
					m_abs_v = abs_val;
					cr = r;
				}
			}

			if (m_abs_v <= tol) return 0.0;
			if (cr != c) {
				double* RESTRICT r_c = &m[c * n];
				double* RESTRICT r_cr = &m[cr * n];
				for (size_t j = 0; j < n; ++j) std::swap(r_c[j], r_cr[j]);
				res *= -1.0;
			}

			const double pvt = m[c * n + c];
			const double inv_pvt = 1.0 / pvt;
			double* RESTRICT r_c = &m[c * n];
			res *= pvt;

			for (size_t r = c + 1; r < n; ++r) {
				double* RESTRICT r_r = &m[r * n];
				if (std::abs(r_r[c]) <= tol) continue;

				const double f = r_r[c] * inv_pvt;
				r_r[c] = 0.0;
				PRAGMA_IVDEP
					for (size_t j = c + 1; j < n; ++j) r_r[j] -= f * r_c[j];
			}
		}
		return res;
	}
}