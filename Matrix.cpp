#include "Matrix.h"

void Matrix::operator+=(const Matrix& rhs) {
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t column = 0; column < columns_; ++column) {
            matrix_[row][column] += rhs.matrix_[row][column];
        }
    }
}

void Matrix::operator-=(const Matrix& rhs) {
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t column = 0; column < columns_; ++column) {
            matrix_[row][column] -= rhs.matrix_[row][column];
        }
    }
}

void Matrix::operator*=(const double& rhs) {
    for (size_t row = 0; row < rows_; ++row) {
        for (size_t column = 0; column < columns_; ++column) {
            matrix_[row][column] *= rhs;
        }
    }
}

Matrix Matrix::operator*(const Matrix& rhs) {
    Matrix product(rows_, rhs.columns_);

    for (size_t row = 0; row < rows_; ++row) {
        for (size_t column = 0; column < rhs.columns_; ++column) {
            for (size_t ind = 0; ind < columns_; ++ind) {
                product.matrix_[row][column] += matrix_[row][ind] * rhs.matrix_[ind][column];
            }
        }
    }

    return product;
}
Matrix Matrix::operator*(const double& rhs) {
    Matrix matrix(*this);
    matrix *= rhs;
    return matrix;
}

Matrix Matrix::operator-(const Matrix& rhs) const {
    Matrix matrix(*this);
    matrix -= rhs;
    return matrix;
}

Matrix Matrix::operator+(const Matrix& rhs) const {
    Matrix matrix(*this);
    matrix += rhs;
    return matrix;
}
std::istream& operator>>(std::istream& in, Matrix& rhs) {
    size_t rows, columns;
    in >> rows >> columns;

    Matrix matrix(rows, columns);
    for (auto& row : matrix.matrix_) {
        for (auto item : row) {
            in >> item;
        }
    }

    rhs = matrix;
    return in;
}

std::ostream& operator<<(std::ostream& out, const Matrix& rhs) {
    out << std::setprecision(Matrix::kOutputPrecision);

    for (const auto& row : rhs.matrix_) {
        for (const auto& item : row) {
            out << std::setprecision(Matrix::kOutputPrecision) << std::setw(Matrix::kOutputWidth)
                << item << ' ';
        }
        out << '\n';
    }
    out.flush();

    return out;
}

void Matrix::SwapRows(const size_t first_row, const size_t second_row) {
    if (first_row == second_row) {
        return;
    }

    matrix_[first_row].swap(matrix_[second_row]);
}

void Matrix::SwapColumns(const size_t first_col, const size_t second_col) {
    if (first_col == second_col) {
        return;
    }

    for (size_t ind = 0; ind < rows_; ++ind) {
        std::swap(matrix_[ind][first_col], matrix_[ind][second_col]);
    }
}

void Matrix::DivideRow(const size_t row, const double coefficient) {
    for (size_t ind = 0; ind < columns_; ++ind) {
        matrix_[row][ind] /= coefficient;
    }
}

void Matrix::ElementaryTransformation(const size_t transforming_row, const size_t main_row,
                                      const double coefficient) {
    for (size_t ind = 0; ind < columns_; ++ind) {
        matrix_[transforming_row][ind] -= coefficient * matrix_[main_row][ind];
    }
}

Matrix Matrix::InverseMatrix() {
    size_t max_row = 0;
    size_t max_col = 0;
    double max_value = matrix_[0][0];

    auto saved_matrix = matrix_;

    Matrix reverse(rows_, columns_);
    for (size_t ind = 0; ind < rows_; ++ind) {
        reverse.matrix_[ind][ind] = 1;
    }

    vector<size_t> rows_permutation(rows_);
    vector<size_t> columns_permutation(columns_);
    for (size_t ind = 0; ind < rows_; ++ind) {
        rows_permutation[ind] = ind;
        columns_permutation[ind] = ind;
    }

    for (size_t row = 0; row < rows_; ++row) {
        for (size_t column = 0; column < rows_; ++column) {
            if (std::abs(max_value) < std::abs(matrix_[row][column])) {
                max_row = row;
                max_col = column;
                max_value = matrix_[row][column];
            }
        }
    }

    for (size_t row = 0; row < rows_; ++row) {
        SwapColumns(row, max_col);
        SwapRows(row, max_row);

        reverse.SwapRows(row, max_row);

        std::swap(columns_permutation[row], columns_permutation[max_col]);
        std::swap(rows_permutation[row], rows_permutation[max_row]);

        for (size_t divided_row = 0; divided_row < columns_; ++divided_row) {
            if (divided_row != row) {
                matrix_[row][divided_row] /= matrix_[row][row];
            }
            reverse.matrix_[row][divided_row] /= matrix_[row][row];
        }
        matrix_[row][row] = 1;

        max_row = max_col = row + 1;
        max_value = 0;

        for (size_t cur_row = 0; cur_row < rows_; ++cur_row) {
            if (cur_row == row) {
                continue;
            }

            double mul = matrix_[cur_row][row];

            for (size_t cur_col = 0; cur_col < columns_; ++cur_col) {
                matrix_[cur_row][cur_col] -= mul * matrix_[row][cur_col];
                if (cur_row == row && cur_col == row) {
                    matrix_[cur_row][cur_col] = 0;
                }

                reverse.matrix_[cur_row][cur_col] -= mul * reverse.matrix_[row][cur_col];

                if (std::abs(max_value) < std::abs(matrix_[cur_row][cur_col])) {
                    max_row = cur_row;
                    max_col = cur_col;
                    max_value = matrix_[cur_row][cur_col];
                }
            }
        }
    }

    for (size_t cur_row = 0; cur_row < columns_permutation.size(); ++cur_row) {
        size_t right_pos = 0;
        for (size_t ind = cur_row; ind < columns_permutation.size(); ++ind) {
            if (columns_permutation[ind] == cur_row) {
                std::swap(columns_permutation[ind], columns_permutation[cur_row]);
                right_pos = ind;
                break;
            }
        }

        reverse.SwapRows(right_pos, cur_row);
    }

    matrix_ = saved_matrix;
    return reverse;
}

void Matrix::RandomGenerate() {
    std::mt19937 gen(clock());
    std::uniform_real_distribution<double> dist(1e3, 1e5);

    for (auto& row : matrix_) {
        for (auto& item : row) {
            item = dist(gen);
        }
    }
}

double Matrix::GetElement(size_t row, size_t column) {
    return matrix_[row][column];
}

void Matrix::SetElement(size_t row, size_t column, double value) {
    matrix_[row][column] = value;
}
Matrix Matrix::TransposeMatrix() {
    Matrix transpose(columns_, rows_);

    for (size_t row = 0; row < rows_; ++row) {
        for (size_t column = 0; column < columns_; ++column) {
            transpose.matrix_[column][row] = matrix_[row][column];
        }
    }

    return transpose;
}
Matrix Matrix::GaussianElimination(Matrix terms) {
    int max_row = 0, max_col = 0;
    auto saved_matrix = matrix_;
    double max_value = matrix_[0][0];
    double det = 1;

    vector<int> var_permutation(columns_);
    for (int i = 0; i < columns_; ++i) {
        var_permutation[i] = i;
    }

    // searching for the maximum
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < rows_; ++j) {
            if (std::abs(max_value) < std::abs(matrix_[i][j])) {
                max_row = i;
                max_col = j;
                max_value = matrix_[i][j];
            }
        }
    }

    for (int i = 0; i < rows_; ++i) {
        // relocating maximum to current row and column
        SwapColumns(i, max_col);
        SwapRows(i, max_row);
        if ((max_col + max_row - 2 * i) % 2 == 1) {
            det *= -1;
        }

        std::swap(terms.matrix_[i][0], terms.matrix_[max_row][0]);
        std::swap(var_permutation[i], var_permutation[max_col]);

        // dividing by leading element
        terms.matrix_[i][0] /= matrix_[i][i];
        for (int j = i + 1; j < columns_; ++j) {
            matrix_[i][j] /= matrix_[i][i];
        }
        det *= matrix_[i][i];
        matrix_[i][i] = 1;

        max_value = max_row = max_col = i + 1;

        // straightforward motion of the Gaussian algorithm
        for (int cur_row = i + 1; cur_row < rows_; ++cur_row) {
            double mul = matrix_[cur_row][i];

            for (int cur_col = i; cur_col < columns_; ++cur_col) {
                matrix_[cur_row][cur_col] -= mul * matrix_[i][cur_col];

                // searching for the maximum
                if (std::abs(max_value) < std::abs(matrix_[cur_row][cur_col])) {
                    max_row = cur_row;
                    max_col = cur_col;
                    max_value = matrix_[cur_row][cur_col];
                }
            }
            terms.matrix_[cur_row][0] -= mul * terms.matrix_[i][0];
        }
    }

    // backward motion
    Matrix answer(columns_, 1);
    for (int cur_var = columns_ - 1; cur_var >= 0; --cur_var) {
        answer.matrix_[cur_var][0] = terms.matrix_[cur_var][0];

        for (int ind = cur_var + 1; ind < columns_; ++ind) {
            answer.matrix_[cur_var][0] -= answer.matrix_[ind][0] * matrix_[cur_var][ind];
        }
    }

    // sorting answer array
    auto copy = answer;
    for (int i = 0; i < columns_; ++i) {
        copy.matrix_[var_permutation[i]][0] = answer.matrix_[i][0];
    }

    matrix_.swap(saved_matrix);
    return copy;
}
