#pragma once

#include <vector>
#include <cstddef>

namespace utlis {
	class Matrix;
	Matrix Uniform(size_t r, size_t c, double min = 0.0, double max = 1.0);
	Matrix Normal(size_t r, size_t c, double mean = 0.0, double stddev = 1.0);

	class Matrix {
	public:
		std::vector<double> data;
		size_t t_r, t_c, t_e;

	public:
		Matrix(size_t r, size_t c) : t_r(r), t_c(c), t_e(r * c), data(r * c, 0.0) {};
		Matrix(size_t r, size_t c, double i_v) : t_r(r), t_c(c), t_e(r* c), data(r* c, i_v) {};

		// Copy Constructor
		Matrix(const Matrix& other)
			: t_r(other.t_r), t_c(other.t_c), t_e(other.t_e), data(other.data) {}
		
		// Move Constructor (noexcept for std::vector optimization), READ this
		Matrix(Matrix&& other) noexcept
			: t_r(std::exchange(other.t_r, 0)), t_c(std::exchange(other.t_c, 0)), t_e(std::exchange(other.t_e, 0)), data(std::move(other.data)) {}

		// Copy Assignment Operator
		Matrix& operator=(const Matrix& other) {
			if (this != &other) {
				t_r = other.t_r;
				t_c = other.t_c;
				t_e = other.t_e;
				data = other.data;
			}
			return *this;
		}

		// Move Assignment Operator
		Matrix& operator=(Matrix&& other) noexcept {
			if (this != &other) {
				t_r = std::exchange(other.t_r, 0);
				t_c = std::exchange(other.t_c, 0);
				t_e = std::exchange(other.t_e, 0);
				data = std::move(other.data);
			}
			return *this;
		}

		~Matrix() = default;

		inline double& operator()(size_t r, size_t c) { return data[r * t_c + c]; };
		inline const double& operator()(size_t r, size_t c) const { return data[r * t_c + c]; };

		Matrix operator+(const Matrix& other) const;
		Matrix& operator+=(const Matrix& other);
		Matrix operator-(const Matrix& other) const;
		Matrix operator-() const;
		Matrix& operator-=(const Matrix& other);
		Matrix operator*(const Matrix& other) const;
		Matrix& operator*=(const Matrix& other);
		Matrix transpose() const;

		Matrix operator+(const double other) const;
		Matrix& operator+=(const double other);
		Matrix operator-(const double other) const;
		Matrix& operator-=(const double other);
		Matrix operator*(const double other) const;
		Matrix& operator*=(const double other);
		Matrix operator/(const double other) const;
		Matrix& operator/=(const double other);

		friend double sum(const Matrix& A);
		friend Matrix sum(const Matrix& A, int axis);
		friend double mean(const Matrix& A);
		friend Matrix mean(const Matrix& A, int axis);
		friend double max(const Matrix& A);
		friend Matrix max(const Matrix& A, int axis);
		friend double min(const Matrix& A);
		friend Matrix min(const Matrix& A, int axis);

		friend void hadamard(const Matrix& A, const Matrix& B, Matrix& dest);
		friend void add_broadcast(const Matrix& A, const Matrix& B, Matrix& dest);
		friend Matrix add_broadcast(const Matrix& A, const Matrix& B);
		friend void mul_broadcast(const Matrix& A, const Matrix& B, Matrix& dest);
		friend Matrix mul_broadcast(const Matrix& A, const Matrix& B);
		friend Matrix I(size_t n);
		friend Matrix Uniform(size_t r, size_t c, double min, double max);
		friend Matrix Normal(size_t r, size_t c, double mean, double stddev);
		friend void Inverse(const Matrix& A, Matrix& dest);
		friend double det(Matrix A);
	};

	inline Matrix Zero(size_t r, size_t c) { return Matrix(r, c); }
	inline Matrix One(size_t r, size_t c) { return Matrix(r, c, 1.0); }
	Matrix I(size_t n);
	Matrix Uniform(size_t r, size_t c, double min, double max);
	Matrix Normal(size_t r, size_t c, double mean, double stddev);
}
