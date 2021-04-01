#pragma once
#include "matrix.hpp"

namespace ml
{
    /**
     * @brief 
     * 
     * @param x 
     * @return double 
     */
    double sigmoid(double x);

    /**
     * @brief 
     * 
     * @param matrix 
     * @return Matrix 
     */
    Matrix sigmoid(Matrix matrix);

    /**
     * @brief 
     * 
     * @param matrix 
     * @return Matrix 
     */
    Matrix d_sigmoid(Matrix matrix);
}