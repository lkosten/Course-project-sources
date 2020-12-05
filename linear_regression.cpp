#include <iostream>
#include <fstream>
#include <random>
#include "Matrix.h"

struct DayStatistics {
    double day_number;
    double year;
    double season;
    double month;
    double is_holiday;
    double week_day;
    double is_working_day;
    double weather;
    double real_temperature;
    double feeling_temperature;
    double humidity;
    double wind_speed;

    double casual_users;
    double registered_users;
    double all_users;

    friend std::istream &operator>>(std::istream &in, DayStatistics &data) {
        std::string buffer;

        std::getline(in, buffer, ',');
        data.day_number = std::stod(buffer);

        std::getline(in, buffer, ',');

        std::getline(in, buffer, ',');
        data.season = std::stod(buffer);

        std::getline(in, buffer, ',');
        data.year = std::stod(buffer);

        std::getline(in, buffer, ',');
        data.month = std::stod(buffer);

        std::getline(in, buffer, ',');
        data.is_holiday = std::stod(buffer);

        std::getline(in, buffer, ',');
        data.week_day = std::stod(buffer);
        if (data.week_day == 0) {
            data.week_day = 7;
        }

        std::getline(in, buffer, ',');
        data.is_working_day = std::stod(buffer);

        std::getline(in, buffer, ',');
        data.weather = std::stod(buffer);

        std::getline(in, buffer, ',');
        data.real_temperature = std::stod(buffer);

        std::getline(in, buffer, ',');
        data.feeling_temperature = std::stod(buffer);

        std::getline(in, buffer, ',');
        data.humidity = std::stod(buffer);

        std::getline(in, buffer, ',');
        data.wind_speed = std::stod(buffer);

        std::getline(in, buffer, ',');
        data.casual_users = std::stod(buffer);

        std::getline(in, buffer, ',');
        data.registered_users = std::stod(buffer);

        std::getline(in, buffer, '\n');
        data.all_users = std::stod(buffer);

        return in;
    }
};

std::vector<DayStatistics> ParseData(const std::string &file_name) {
    std::ifstream input(file_name);
    if (!input) {
        throw std::runtime_error("File not found!");
    }

    std::vector<DayStatistics> data;
    std::string buffer;
    std::getline(input, buffer);

    while (!input.eof()) {
        DayStatistics current_day;
        input >> current_day;

        data.push_back(current_day);
    }
    std::cout << "Dataset contains " << data.size() << " instances" << std::endl;
    return data;
}

void TestReverseMatrix(const size_t dimension = 900) {
    Matrix rnd(dimension, dimension);
    rnd.RandomGenerate();

    rnd = rnd.TransposeMatrix() * rnd;
    auto start_time = clock();
    auto inv = rnd.InverseMatrix();
    auto e = inv * rnd * rnd * inv;
    auto end_time = clock();

    double min_diag = e.GetElement(0, 0);
    double max_diag = e.GetElement(0, 0);

    for (size_t row = 0; row < dimension; ++row) {
        min_diag = std::min(min_diag, e.GetElement(row, row));
        max_diag = std::max(max_diag, e.GetElement(row, row));
    }

    double max_not_diag = e.GetElement(1, 0);
    double min_not_diag = e.GetElement(1, 0);

    for (size_t row = 0; row < dimension; ++row) {
        for (size_t column = 0; column < dimension; ++column) {
            if (row != column) {
                max_not_diag = std::max(max_not_diag, e.GetElement(row, column));
                min_not_diag = std::min(min_not_diag, e.GetElement(row, column));
            }
        }
    }

    std::cout << "Dimension: " << dimension << std::endl;
    std::cout << "Diagonal max min: " << max_diag << '\t' << min_diag << std::endl;
    std::cout << "Non diagonal max min: " << max_not_diag << '\t' << min_not_diag << std::endl;

    std::cout << "Time: " << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC
              << std::endl;
}

Matrix ParseResponseVector(const std::vector<DayStatistics> &data) {
    Matrix y_values(data.size(), 1);

    for (size_t ind = 0; ind < data.size(); ++ind) {
        y_values.SetElement(ind, 0, data[ind].all_users);
    }

    return y_values;
}

Matrix ParseExplanatoryVector(const std::vector<DayStatistics> &data) {
    Matrix x_values(data.size(), 12);

    for (size_t ind = 0; ind < data.size(); ++ind) {
        x_values.SetElement(ind, 0, 1);
        x_values.SetElement(ind, 1, data[ind].year);
        x_values.SetElement(ind, 2, data[ind].season);
        x_values.SetElement(ind, 3, data[ind].month);
        x_values.SetElement(ind, 4, data[ind].is_holiday);
        x_values.SetElement(ind, 5, data[ind].week_day);
        x_values.SetElement(ind, 6, data[ind].is_working_day);
        x_values.SetElement(ind, 7, data[ind].weather);
        x_values.SetElement(ind, 8, data[ind].real_temperature);
        x_values.SetElement(ind, 9, data[ind].feeling_temperature);
        x_values.SetElement(ind, 10, data[ind].humidity);
        x_values.SetElement(ind, 11, data[ind].wind_speed);
    }

    return x_values;
}

void BikeSharingTask() {
    auto data = ParseData("day.csv");

    auto y_values = ParseResponseVector(data);
    std::cout << y_values.GetRowsNumber() << ' ' << y_values.GetColumnsNumber() << std::endl;
    auto x_values = ParseExplanatoryVector(data);
    std::cout << x_values.GetRowsNumber() << ' ' << x_values.GetColumnsNumber() << std::endl;

    auto mtr = x_values.TransposeMatrix() * x_values;

    auto inverse = mtr.InverseMatrix();

    auto beta = inverse * x_values.TransposeMatrix() * y_values;
    std::cout << beta;

    auto approx_y = x_values * beta;
    auto error = approx_y - y_values;

    std::ofstream out("linear_approx");
    for (size_t ind = 0; ind < error.GetRowsNumber(); ++ind) {
        out << approx_y.GetElement(ind, 0) << '\n';
    }
    long double stder = 0;
    for (size_t ind = 0; ind < error.GetRowsNumber(); ++ind) {
        stder += error.GetElement(ind, 0) * error.GetElement(ind, 0) / error.GetRowsNumber();
    }
    stder = std::sqrt(stder);
    std::cout << stder;
}

struct LineStatistics {
    double y_value;
    double x_value;
};

std::vector<LineStatistics> ParseDataLine(const std::string &file_name) {
    std::ifstream input(file_name);
    if (!input) {
        throw std::runtime_error("File not found!");
    }

    std::vector<LineStatistics> data;
    std::string buffer;

    while (!input.eof()) {
        LineStatistics current_day;
        input >> current_day.x_value >> current_day.y_value;

        data.push_back(current_day);
    }
    std::cout << "Dataset contains " << data.size() << " instances" << std::endl;
    return data;
}

Matrix ParseExplanatoryVector(const std::vector<LineStatistics> &data) {
    Matrix x_values(data.size(), 2);

    for (size_t ind = 0; ind < data.size(); ++ind) {
        x_values.SetElement(ind, 0, 1);
        x_values.SetElement(ind, 1, data[ind].x_value);
    }

    return x_values;
}

Matrix ParseResponseVector(const std::vector<LineStatistics> &data) {
    Matrix y_values(data.size(), 1);

    for (size_t ind = 0; ind < data.size(); ++ind) {
        y_values.SetElement(ind, 0, data[ind].y_value);
    }

    return y_values;
}

void SimpleLineTask() {
    auto data = ParseDataLine("line_generated");

    auto x_values = ParseExplanatoryVector(data);
    auto y_values = ParseResponseVector(data);

    auto mtr = x_values.TransposeMatrix() * x_values;

    auto inverse = mtr.InverseMatrix();

    auto beta = inverse * x_values.TransposeMatrix() * y_values;
    std::cout << beta;

    auto approx_y = x_values * beta;
    auto error = approx_y - y_values;

    std::ofstream out("line_approx");
    out << beta.GetElement(0, 0) - 20 * beta.GetElement(1, 0) << '\n';
    out << beta.GetElement(0, 0) + 51 * beta.GetElement(1, 0) << '\n';

    long double stder = 0;
    for (size_t ind = 0; ind < error.GetRowsNumber(); ++ind) {
        stder += error.GetElement(ind, 0) * error.GetElement(ind, 0) / error.GetRowsNumber();
    }
    stder = std::sqrt(stder);
    std::cout << stder;
}

struct MultiLineStatistics {
    std::vector<double> x_value;
    double y_value;
};

std::vector<MultiLineStatistics> ParseDataMultiLine(const std::string &file_name) {
    std::ifstream input(file_name);
    if (!input) {
        throw std::runtime_error("File not found!");
    }

    std::vector<MultiLineStatistics> data;

    while (!input.eof()) {
        MultiLineStatistics current_day;
        current_day.x_value.resize(5);
        for (auto &item : current_day.x_value) {
            input >> item;
        }
        input >> current_day.y_value;

        data.push_back(current_day);
    }
    std::cout << "Dataset contains " << data.size() << " instances" << std::endl;
    return data;
}

Matrix ParseExplanatoryVector(const std::vector<MultiLineStatistics> &data) {
    Matrix x_values(data.size(), 6);

    for (size_t ind = 0; ind < data.size(); ++ind) {
        x_values.SetElement(ind, 0, 1);
        x_values.SetElement(ind, 1, data[ind].x_value[0]);
        x_values.SetElement(ind, 2, data[ind].x_value[1]);
        x_values.SetElement(ind, 3, data[ind].x_value[2]);
        x_values.SetElement(ind, 4, data[ind].x_value[3]);
        x_values.SetElement(ind, 5, data[ind].x_value[4]);
    }

    return x_values;
}

Matrix ParseResponseVector(const std::vector<MultiLineStatistics> &data) {
    Matrix y_values(data.size(), 1);

    for (size_t ind = 0; ind < data.size(); ++ind) {
        y_values.SetElement(ind, 0, data[ind].y_value);
    }

    return y_values;
}

void MultiLineTask() {
    auto data = ParseDataMultiLine("multi_line_generated");

    auto x_values = ParseExplanatoryVector(data);
    auto y_values = ParseResponseVector(data);

    auto mtr = x_values.TransposeMatrix() * x_values;

    auto inverse = mtr.InverseMatrix();

    auto beta = inverse * x_values.TransposeMatrix() * y_values;
    std::cout << beta;

    auto approx_y = x_values * beta;
    auto error = approx_y - y_values;

    long double stder = 0;
    for (size_t ind = 0; ind < error.GetRowsNumber(); ++ind) {
        stder += error.GetElement(ind, 0) * error.GetElement(ind, 0) / error.GetRowsNumber();
    }
    stder = std::sqrt(stder);
    std::cout << stder;
}

void RunMultiLineTask() {
    std::ofstream out("multi_line_generated");
    std::ofstream out_to_show("show_multi_line_generated");
    std::mt19937 gen(clock());
    std::normal_distribution<double> x_distr(10, 20);
    std::uniform_real_distribution<double> uni_distr(0, 1);
    std::normal_distribution<double> eps_distr(0, 3);

    size_t len = 100;
    for (size_t cnt = 0; cnt < len; ++cnt) {
        std::vector<double> x(5, 0);
        x[0] = x_distr(gen);
        x[1] = x_distr(gen);
        x[2] = x_distr(gen);

        auto val = uni_distr(gen);
        if (val <= 1. / 3) {
            x[3] = 1;

        } else if (val <= 2. / 3) {
            x[4] = 1;
        }

        double y = -9 * x[0] + 8 * x[1] - 2 * x[2] - 4 * x[3] + 6 * x[4];
        for (const auto &item : x) {
            out << item << ' ';
        }
        out << y;
        out_to_show << y << '\n';
        if (cnt + 1 < len) {
            out << '\n';
        }
    }

    out.close();
    MultiLineTask();
}

int main() {
    BikeSharingTask();
    return 0;
}
