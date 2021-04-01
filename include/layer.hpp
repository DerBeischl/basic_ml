#pragma once

#include <Matrix.hpp>
#include <cstddef>
#include <ostream>

namespace ml
{

  /**
   * @brief 
   * 
   */
  class Layer
  {
  public:
    /**
     * @brief Construct a new Layer object
     * 
     * @param input_shape 
     * @param output_shape 
     */
    explicit Layer(const size_t input_shape, const size_t output_shape);

    /**
     * @brief Construct a new Layer object
     * 
     * @param previous 
     * @param output_shape 
     */
    explicit Layer(const Layer &previous, const size_t output_shape);

    /**
     * @brief 
     * 
     * @return size_t 
     */
    size_t input_shape() const;

    /**
     * @brief 
     * 
     * @return size_t 
     */
    size_t output_shape() const;

    /**
     * @brief Get the weights object
     * 
     * @return const Matrix& 
     */
    const Matrix &get_weights() const;

    /**
     * @brief Get the bias object
     * 
     * @return const Matrix& 
     */
    const Matrix &get_bias() const;

    /**
     * @brief Get the weighted output object
     * 
     * @return const Matrix& 
     */
    const Matrix &get_weighted_output() const;

    /**
     * @brief 
     * 
     * @param input 
     * @return Matrix 
     */
    Matrix operator()(const Matrix &input);

    /**
     * @brief 
     * 
     * @param weighted_input 
     * @param delta 
     * @return Matrix 
     */
    Matrix grad(const Matrix &weighted_input, const Matrix &delta);

    /**
     * @brief 
     * 
     * @param alpha 
     */
    void update_weights(const double alpha);

    /**
     * @brief 
     * 
     * @param out 
     * @param rhs 
     * @return std::ostream& 
     */
    friend std::ostream &operator<<(std::ostream &out, const Layer &rhs);

  private:
    Matrix theta_, bias_;

    Matrix forward_cache_, weighted_output_, grad_cache_;
  };

}; // namespace ml