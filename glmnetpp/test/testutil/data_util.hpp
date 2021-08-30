#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

namespace glmnetpp {

Eigen::MatrixXd read_csv(const std::string& filename)
{
    std::vector<double> matrixEntries;

    std::ifstream matrixDataFile(filename);

    std::string matrixRowString;
    std::string matrixEntry;

    int matrixRowNumber = 0;

    while (std::getline(matrixDataFile, matrixRowString)) 
    {
        std::stringstream matrixRowStringStream(matrixRowString); 
        while (std::getline(matrixRowStringStream, matrixEntry, ',')) 
        {
            matrixEntries.push_back(stod(matrixEntry));
        }
        matrixRowNumber++;
    }

    // here we convet the vector variable into the matrix and return the resulting object,
    // note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);
}

// Center and standardize a matrix with 0 mean columns and scaled by var/n each column
template <class T, int R, int C>
Eigen::Matrix<T, R, C> 
center_scale(const Eigen::Matrix<T, R, C>& X)
{
    Eigen::Matrix<T, R, C> out(X.rows(), X.cols());
    auto n = X.rows();
    for (int i = 0; i < X.cols(); ++i) {
        out.col(i) = X.col(i).array() - X.col(i).mean();
        out.col(i) /= out.col(i).norm() / std::sqrt(n);
    }
    return out;
}

} // namespace glmnetpp
