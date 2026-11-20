#include <iostream>
#include <filesystem>
#include "linear_regression.hpp"
#include "matrix.hpp"
#include "loader.hpp"

int main() {
    auto print_mat = [](const std::string& label, const utils::Matrix& M) {
        std::cout << label << " (" << M.t_r << "x" << M.t_c << "):\n";
        for (size_t i = 0; i < M.t_r; ++i) {
            for (size_t j = 0; j < M.t_c; ++j) {
                std::cout << M.data[i * M.t_c + j] << (j + 1 < M.t_c ? "\t" : "");
            }
            std::cout << "\n";
        }
        std::cout << "-------------------\n";
        };

    std::filesystem::path csv_path = std::filesystem::path(PROJECT_ROOT_DIR) / "data.csv";
    auto [X, y] = utils::split_xy(utils::load_csv(csv_path.string()));

    models::LinearRegression lr;
    lr.fit(X, y);

    print_mat("Weights", lr.weights());   // Expect: 2.5, -1.2, 0.8
    print_mat("Bias", lr.bias());         // Expect: 4.0

    utils::Matrix preds = lr.predict(X);
    print_mat("Predictions", preds);      // Should match y column
}
