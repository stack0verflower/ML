#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "matrix.hpp"
#include "compiler_macros.hpp"

namespace utils {
	inline Matrix load_csv(const std::string& filename, bool has_header = true) {
		std::ifstream file(filename);
		if (!file.is_open()) throw std::runtime_error("Could not open file: " + filename);

		std::string line;
		if (has_header) std::getline(file, line); // Skip header

		std::vector<double> td;
		size_t col = 0, row = 0;

		while (std::getline(file, line)) {
			if (line.empty()) continue;

			std::stringstream ss(line);
			std::string blk;

			size_t c_col = 0;
			while (std::getline(ss, blk, ',')) {
				td.push_back(std::stod(blk));
				c_col++;
			}

			if (row == 0) col = c_col;
			else if (c_col != col) throw std::runtime_error("Inconsistent number of columns in CSV file.");
			row++;
		}

		Matrix M(row, col);
		M.data = std::move(td);
		return M;
	}

    // Peels the target column (default: last column) out of a loaded [X | y] dataset matrix
    inline std::pair<Matrix, Matrix> split_xy(const Matrix& dataset, size_t target_col = SIZE_MAX) {
        if (UNLIKELY(dataset.t_r == 0 || dataset.t_c < 2)) 
            throw std::invalid_argument("split_xy: dataset must have at least 1 row and 2 columns.");
        

        const size_t t_col = (target_col == SIZE_MAX) ? (dataset.t_c - 1) : target_col;
        if (UNLIKELY(t_col >= dataset.t_c)) 
            throw std::out_of_range("split_xy: target_col out of bounds.");
        
        const size_t n_rows = dataset.t_r;
        const size_t n_features = dataset.t_c - 1;

        Matrix X(n_rows, n_features);
        Matrix y(n_rows, 1);

        const double* RESTRICT src = dataset.data.data();
        double* RESTRICT x_dst = X.data.data();
        double* RESTRICT y_dst = y.data.data();

        for (size_t r = 0; r < n_rows; ++r) {
            const size_t src_offset = r * dataset.t_c;
            const size_t x_offset = r * n_features;

            // Fast path when target is the last column (single contiguous block copy per row)
            if (LIKELY(t_col == n_features)) {
                std::copy_n(&src[src_offset], n_features, &x_dst[x_offset]);
                y_dst[r] = src[src_offset + t_col];
            } else {
                std::copy_n(&src[src_offset], t_col, &x_dst[x_offset]);
                std::copy_n(&src[src_offset + t_col + 1], n_features - t_col, &x_dst[x_offset + t_col]);
                y_dst[r] = src[src_offset + t_col];
            }
        }

        return { std::move(X), std::move(y) };
    }
}
