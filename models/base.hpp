#pragma once

#include <concepts>
#include "matrix.hpp"

namespace models {
	using Matrix = utils::Matrix;

	template <typename T>
	concept SupervisedModel = requires(T model, const T const_model, const Matrix& X, const Matrix& y) {
		{ model.fit(X, y) } -> std::same_as<void>;
		{ const_model.predict(X) } -> std::same_as<Matrix>;
	};

	template <typename T>
	concept UnsupervisedModel = requires(T model, const T const_model, const Matrix& X) {
		{ model.fit(X) } -> std::same_as<void>;
		{ const_model.predict(X) } -> std::same_as<Matrix>;
	};
}
