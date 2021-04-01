#pragma once

#include <cstddef>
#include <ostream>
#include <vector>
#include <initializer_list>

namespace ml
{

  /**
   * @brief 
   * 
   */
  class Matrix
  {
  public:
    bool is_transposed;

    /**
     * @brief Construct a new Matrix object
     * 
     * @param width 
     * @param height 
     */
    explicit Matrix(const size_t width, const size_t height);

    /**
     * @brief Construct a new Matrix object
     * 
     * @param data 
     */
    explicit Matrix(const std::initializer_list<double> &data);

    /**
     * @brief 
     * 
     * @return constexpr size_t 
     */
    size_t width() const;

    /**
     * @brief 
     * 
     * @return constexpr size_t 
     */
    size_t height() const;

    /**
     * @brief 
     * 
     * @param rhs 
     * @return Matrix& 
     */
    Matrix &outer_product(const Matrix &rhs);

    /**
     * @brief 
     * 
     * @param x 
     * @param y 
     * @return double 
     */
    double operator()(const size_t x, const size_t y) const;

    /**
     * @brief 
     * 
     * @param x 
     * @param y 
     * @return double& 
     */
    double &operator()(const size_t x, const size_t y);

    /**
     * @brief 
     * 
     * @param i 
     * @return double 
     */
    double operator[](const size_t i) const;

    /**
     * @brief 
     * 
     * @param i 
     * @return double& 
     */
    double &operator[](const size_t i);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return Matrix& 
     */
    Matrix &operator+=(const Matrix &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return Matrix& 
     */
    Matrix &operator*=(const Matrix &rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return Matrix& 
     */
    Matrix &operator*=(const double rhs);

    /**
     * @brief 
     * 
     * @param rhs 
     * @return Matrix& 
     */
    Matrix &operator-=(const Matrix &rhs);

    /**
     * @brief 
     * 
     * @param lhs 
     * @param rhs 
     * @return Matrix 
     */
    friend Matrix operator*(Matrix lhs, const Matrix &rhs);

    /**
     * @brief 
     * 
     * @param lhs 
     * @param rhs 
     * @return Matrix 
     */
    friend Matrix operator*(Matrix lhs, const double rhs);

    /**
     * @brief 
     * 
     * @param lhs 
     * @param rhs 
     * @return Matrix 
     */
    friend Matrix operator+(Matrix lhs, const Matrix &rhs);

    /**
     * @brief 
     * 
     * @param lhs 
     * @param rhs 
     * @return Matrix 
     */
    friend Matrix operator-(Matrix lhs, const Matrix &rhs);

    /**
     * @brief 
     * 
     * @param out 
     * @param rhs 
     * @return std::ostream& 
     */
    friend std::ostream &operator<<(std::ostream &out, const Matrix &rhs);

  private:
    size_t width_, height_;
    std::vector<double> storage_;
  };
} // namespace ml
