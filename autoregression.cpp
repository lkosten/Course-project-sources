#include <iostream>
#include <fstream>
#include <algorithm>

#include "Matrix.h"

struct AutoregressiveModel {
    size_t degree;
    long double c_hat;
    std::vector<long double> phi_hat;
};

std::vector<long double> ReadData(const std::string &file_name) {
    std::ifstream input(file_name);
    if (!input) {
        throw std::runtime_error("Can't open file!");
    }

    std::vector<long double> time_series;
    while (!input.eof()) {
        long double cur_val;
        input >> cur_val;

        time_series.push_back(cur_val);
    }

    std::cout << "File " << file_name << " contains time series of length " << time_series.size()
              << std::endl;

    return time_series;
}

long double GetMean(const std::vector<long double> time_series) {
    long double mean = 0;
    for (const auto &item : time_series) {
        mean += item;
    }

    return mean / time_series.size();
}

AutoregressiveModel BuildAutoregressiveModel(std::vector<long double> time_series,
                                             const size_t degree) {
    AutoregressiveModel model;
    model.degree = degree;
    model.phi_hat.assign(degree, 0);

    Matrix r_series(degree, 1);

    auto mean = GetMean(time_series);
    for (auto &item : time_series) {
        item -= mean;
    }

    std::vector<double> coefficients(degree, 0);
    std::vector<std::vector<double>> mat(degree, std::vector<double>(degree, 0));
    for (size_t i = degree - 1; i < time_series.size() - 1; i++) {
        size_t hi = i + 1;
        for (size_t j = 0; j < degree; j++) {
            size_t hj = i - j;
            coefficients[j] += (time_series[hi] * time_series[hj]);
            for (size_t k = j; k < degree; ++k) {
                mat[j][k] += time_series[hj] * time_series[i - k];
            }
        }
    }
    for (int i = 0; i < degree; i++) {
        coefficients[i] /= (time_series.size() - degree);
        for (size_t j = i; j < degree; ++j) {
            mat[i][j] /= (time_series.size() - degree);
            mat[j][i] = mat[i][j];
        }
    }

    for (size_t cur_degree = 1; cur_degree <= degree; ++cur_degree) {
        r_series.SetElement(cur_degree - 1, 0, coefficients[cur_degree - 1]);
    }

    Matrix r_matrix(degree, degree);
    for (int row = 0; row < degree; ++row) {
        for (int column = row; column < degree; ++column) {
            r_matrix.SetElement(row, column, mat[row][column]);
            r_matrix.SetElement(column, row, mat[column][row]);
        }
    }

    auto phi_matrix = r_matrix.GaussianElimination(r_series);
    for (size_t ind = 0; ind < degree; ++ind) {
        model.phi_hat[ind] = phi_matrix.GetElement(ind, 0);
    }

    model.c_hat = mean;
    long double buffer = 1;
    for (const auto item : model.phi_hat) {
        buffer -= item;
    }
    model.c_hat *= buffer;

    return model;
}

long double Func(long double x) {
    return std::sin(x / 25) - 0.69 * std::sin(x / 81) + 1.26 * std::sin(x / 19) -
           0.18 * std::sin(x / 17) + 3.12 * std::sin(x / 34);
}
long double GetRootSquareError(const AutoregressiveModel &model,
                               std::vector<long double> time_series, size_t degree) {
    long double error = 0;

    auto mean = GetMean(time_series);
    for (auto &item : time_series) {
        item -= mean;
    }

    auto series_hat = time_series;

    for (size_t ind = 0; ind < time_series.size(); ++ind) {
        if (ind >= model.degree) {
            series_hat[ind] = 0;
            for (size_t p = 0; p < model.degree; ++p) {
                series_hat[ind] += model.phi_hat[p] * time_series[ind - p - 1];
            }
        }

        error += (series_hat[ind] - time_series[ind]) * (series_hat[ind] - time_series[ind]);
    }
    error /= time_series.size() - degree;
    error = std::sqrt(error);

    /*std::ofstream out("bike_users_approx" + std::to_string(degree));
    for (const auto &item : series_hat) {
        out << item  + mean << '\n';
    }*/

    return error;
}

std::vector<long double> GenerateRandomAutoregressiveModel(size_t length, size_t degree,
                                                           long double noise_variance) {
    const double max_coef = 1;
    std::mt19937 gen(clock());
    std::normal_distribution<double> noise_distribution(0, noise_variance);
    std::uniform_real_distribution<double> coef_distribution(-max_coef, max_coef);

    std::vector<long double> coefficients(degree);
    for (auto &item : coefficients) {
        item = coef_distribution(gen);
    }

    coefficients = {1.02569, 0.551164, -0.85639, 0.219587};
    degree = 4;

    auto mean = 3;
    std::vector<long double> time_series(length);
    for (size_t index = 0; index < length; ++index) {
        if (index >= degree) {
            time_series[index] = 0;
            for (size_t prev_ind = 0; prev_ind < degree; ++prev_ind) {
                time_series[index] +=
                    coefficients[prev_ind] * (time_series[index - prev_ind - 1] - mean);
            }

        } else {
            time_series[index] = coef_distribution(gen);
        }

        time_series[index] += mean;
        time_series[index] += noise_distribution(gen);
    }

    return time_series;
}

std::vector<long double> GenerateFunctionSeries(size_t length) {
    std::vector<long double> time_series(length);
    for (size_t point = 0; point < length; ++point) {
        time_series[point] = Func(point);
    }

    std::ofstream out("bike_users_approx");
    for (const auto &item : time_series) {
        out << std::setprecision(10) << std::fixed << item << '\n';
    }

    return time_series;
}

int main() {
    auto time_series = ReadData("bike_users_data");

    std::ofstream out("stderr");
    for (size_t degree = 1; degree < 101; ++degree) {
        auto model = BuildAutoregressiveModel(time_series, degree);

        std::cout << "Degree: " << degree << std::endl << model.c_hat << std::endl;
        for (const auto &item : model.phi_hat) {
            std::cout << std::setprecision(10) << std::setw(10) << item << ' ';
        }
        std::cout << std::endl;

        std::cout << "Root square error: " << std::setprecision(10)
                  << GetRootSquareError(model, time_series, degree) << std::endl;

        std::cout << std::endl << std::endl;

        out << GetRootSquareError(model, time_series, degree) << '\n';
    }

    std::cout << GetMean(time_series);

    return 0;
}

/*
0.384810	0.319719	0.227814	0.130160

0.408910	0.385517	0.320082	0.227765
0.385517	0.409336	0.385608	0.319835
0.320082	0.385608	0.409172	0.385242
0.227765	0.319835	0.385242	0.408779
3.736794
-5.474295
3.731127
-0.996783
 */
