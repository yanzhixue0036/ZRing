#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>


class DatasetManager {
public:
    explicit DatasetManager(const std::string& file_path) {
        load(file_path);
    }

    inline size_t size() const {
        return data_.size();
    }

    inline const std::vector<double> insert_data(double DR = 1.0) const {
        const size_t n = data_.size();
        size_t insert_num = static_cast<size_t>(std::floor(DR * n));

        if (insert_num > data_.size()) insert_num = data_.size();  
        return std::vector<double>(data_.begin(), data_.begin() + insert_num);
    }

    inline std::vector<int> insert_indices(size_t start = 0) const {
        const size_t n = data_.size();
        std::vector<int> idx;
        idx.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            idx.push_back(static_cast<int>(i+start*n));
        }
        return idx;
    }

    inline std::vector<double> delete_data(double DR) const {
        if (DR <= 0.0) return {};
        if (DR > 1.0) DR = 1.0;

        const size_t n = data_.size();
        const size_t del_num = static_cast<size_t>(std::floor(DR * n));

        std::vector<double> del;
        del.reserve(del_num);

        for (size_t i = 0; i < del_num; ++i) {
            del.push_back(data_[n - 1 - i]);
        }
        return del;
    }

    inline std::vector<int> delete_indices(double DR) const {
        if (DR <= 0.0) return {};
        if (DR > 1.0) DR = 1.0;

        const size_t n = data_.size();
        const size_t del_num = static_cast<size_t>(std::floor(DR * n));

        std::vector<int> idx;
        idx.reserve(del_num);

        for (size_t i = 0; i < del_num; ++i) {
            idx.push_back(static_cast<int>(n - 1 - i));
        }
        return idx;
    }

private:
    std::vector<double> data_;

private:
    void load(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file) {
            throw std::runtime_error("Failed to open dataset file: " + file_path);
        }

        double v;
        while (file >> v) {
            data_.push_back(v);
        }
        file.close();
    }
};
