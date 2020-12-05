#ifndef AUTOREGRESSION_MATRIX_H
#define AUTOREGRESSION_MATRIX_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <random>

using std::vector;

class Matrix {
public:
    Matrix() = default;
    Matrix(size_t rows, size_t columns)
        : rows_(rows), columns_(columns), matrix_(rows, vector<double>(columns, 0)) {
    }

    virtual ~Matrix() = default;

    Matrix operator-(const Matrix &rhs) const;
    Matrix operator+(const Matrix &rhs) const;
    void operator-=(const Matrix &rhs);
    void operator+=(const Matrix &rhs);
    void operator*=(const double &rhs);
    Matrix operator*(const Matrix &rhs);
    Matrix operator*(const double &rhs);

    friend std::istream &operator>>(std::istream &in, Matrix &rhs);
    friend std::ostream &operator<<(std::ostream &out, const Matrix &rhs);

    Matrix GaussianElimination(Matrix terms);
    Matrix GaussianEliminationImproved(Matrix terms);
    Matrix InverseMatrix();
    Matrix TransposeMatrix();

    void RandomGenerate();

    double GetElement(size_t row, size_t column);
    void SetElement(size_t row, size_t column, double value);
    size_t GetRowsNumber() {
        return rows_;
    }
    size_t GetColumnsNumber() {
        return columns_;
    }

private:
    size_t rows_;
    size_t columns_;
    vector<vector<double>> matrix_;

    static const size_t kOutputPrecision = 10;
    static const size_t kOutputWidth = 13;

    void SwapRows(const size_t first_row, const size_t second_row);
    void SwapColumns(const size_t first_col, const size_t second_col);
    void DivideRow(const size_t row, const double coefficient);
    void ElementaryTransformation(const size_t transforming_row, const size_t main_row,
                                  const double coefficient);
};

#endif  // AUTOREGRESSION_MATRIX_H
