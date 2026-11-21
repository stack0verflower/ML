#pragma once 

#include "base.hpp"
#include "matrix.hpp"
#include "compiler_macros.hpp"

namespace models {
	enum class Solver {
		NormalEquation,
		GradientDescent
	};

	class LinearRegression {
	private:
		Matrix W_;
		Matrix b_;
		bool fitted_ = false;

		Solver solver_;
		double lr_;
		size_t mx_iter_;
		double tol;

		void fit_gradient_descent(const Matrix& X, const Matrix& y) {
			const size_t n = X.t_r;
			const size_t d = X.t_c;
			const double inv_n = 1.0 / static_cast<double>(n);

			W_ = Matrix(d, 1, 0.0);
			b_ = Matrix(1, 1, 0.0);

			const Matrix X_t = X.transpose();
			for (size_t it = 0; it < mx_iter_; it++) {
				Matrix E = X * W_;
				add_broadcast(E, b_, E);
				E -= y;

				Matrix grad_w = (X_t * E) * inv_n, grad_b = mean(E, 0);
				W_ -= grad_w * lr_;
				b_ -= grad_b * lr_;

				double max_grad = std::abs(grad_b(0, 0));
				for (size_t i = 0; i < d; ++i) 	max_grad = std::max(max_grad, std::abs(grad_w(i, 0)));
				if (max_grad < tol) break;
			}
			std::cerr << "LinearRegression::fit_gradient_descent: A gradient of " << tol << " was not reached." << std::endl;
			fitted_ = true;
		}

		void fit_normal_equation(const Matrix& X, const Matrix& y) {
			/*
				We have X.t_r equations with X.t_c features.
				This gives us line of the form, y = aX1 + bX2 + cX3 + ... The constant term is missing.

				But, we know that the set {y_mean, x1_mean, x2_mean .. } lies on this plane.
				Thus, we obtain a new set of equations, where each column now holds (val - Xi_mean).

				This transforms a single equation in there to:
					y - y_mean = a(x1 - x1_mean) + b(x2 - x2_mean) +...
				Our val is just the numeric value of x1. Since we are determining the values of a, b, c..., the coeff changes from mere x1 to
				x1 - x1_mean.
			*/
			Matrix X_m = mean(X, 0);
			Matrix X_c = X;
			add_broadcast(X_c, -X_m, X_c);


			Matrix y_m = mean(y, 0);
			Matrix y_c = y;
			add_broadcast(y_c, -y_m, y_c);

			/*
				We know, that for n lines, we calculate coeff by getting E = (y - y_mean - a(X1 - X1_mean) - b(X2 - X2_mean)....)^2.
				Then we get equations by putting partial derivate of E against a, b, c .. and so on to 0. We get equations as many as features.
				Somehow, (see proof and derive this), this resolves to
					XT.X . O  =  XT . y
				Here, XT is transpose of X. O is coeff matrix, a [n, 1] matrix where n is coeff count, and it houses coefficient.
				Solving this gives us value of matrix O, which we need.
			*/
			Matrix lhs = X_c.transpose();
			Matrix rhs = lhs;
			lhs *= X_c;
			rhs *= y_c;

			Matrix lhs_inv(lhs.t_r, lhs.t_c);
			Inverse(lhs, lhs_inv);
			W_ = lhs_inv * rhs;
			b_ = y_m - (X_m * W_);
			fitted_ = true;
		}

	public:
		explicit LinearRegression(Solver solver = Solver::NormalEquation, double lr = 0.01, size_t mx_iter_ = 1000, double tol = 1e-6)
			: solver_(solver), lr_(lr), mx_iter_(mx_iter_), tol(tol) {};

		void fit(const Matrix& X, const Matrix& y) {
			if (UNLIKELY(X.t_r == 0 || X.t_c == 0 || X.t_r != y.t_r || y.t_c == 0)) 
				throw std::invalid_argument("LinearRegression::fit: invalid input dimensions.");
			
			if (solver_ == Solver::NormalEquation) fit_gradient_descent(X, y);
			else fit_normal_equation(X, y);
		}

		/*
			This function computes and returns a value without modifying anything; 
			if the caller ignores/discards that return value, emit a compiler warning.

			model.predict(X_test); // Forgot to assign it: Matrix y_hat = model.predict(X_test);

			This leads to a warning in MSVC
				C4834: discarding return value of function with 'nodiscard' attribute
		*/

		[[nodiscard]] Matrix predict(const Matrix& X) const {
			if (UNLIKELY(!fitted_)) 
				throw std::runtime_error("LinearRegression::predict: model not fitted.");
			
			if (UNLIKELY(X.t_c != W_.t_r)) 
				throw std::invalid_argument("LinearRegression::predict: feature dimension mismatch.");
			
			Matrix res = X * W_;
			add_broadcast(res, b_, res);
			return res;
		}

		[[nodiscard]] const Matrix& weights() const noexcept { return W_; }
		[[nodiscard]] const Matrix& bias() const noexcept { return b_; }
	};

	static_assert(SupervisedModel<LinearRegression>);
}
