#include "activations.hpp"
#include <cmath>

namespace ml
{
    double sigmoid(double x)
    {
        return std::exp(x) / (std::exp(-x) + 1);
    }

    Matrix sigmoid(Matrix matrix)
    {
        for (size_t i = 0; i < matrix.width() * matrix.height(); ++i)
            matrix[i] = sigmoid(matrix[i]);
        return matrix;
    }

    Matrix d_sigmoid(Matrix matrix)
    {
        for (size_t i = 0; i < matrix.width() * matrix.height(); ++i)
            matrix[i] = sigmoid(matrix[i]) * (1 - sigmoid(matrix[i]));
        return matrix;
    }
}