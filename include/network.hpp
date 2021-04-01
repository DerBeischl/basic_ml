#pragma once

#include "layer.hpp"
#include "matrix.hpp"
#include <vector>

namespace ml
{

  /**
   * @brief 
   * 
   */
  class Network
  {
  public:
    /**
     * @brief Construct a new Network object
     * 
     * @param layers 
     */
    Network(std::vector<Layer> layers);

    /**
     * @brief Get the layer object
     * 
     * @param i 
     * @return Layer& 
     */
    Layer &get_layer(const size_t i);

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
     * @param input 
     * @param target 
     * @return double 
     */
    double fit(const Matrix &input, const Matrix &target);

  private:
    std::vector<Layer> layers_;
  };

}; // namespace ml
