#include "layer.hpp"
#include "random.hpp"
#include "activations.hpp"
#include <cassert>
#include <iostream>

namespace ml
{

    Layer::Layer(const size_t input_shape, const size_t output_shape)
        : theta_(input_shape, output_shape),
          bias_(1, output_shape),
          forward_cache_(1, output_shape),
          weighted_output_(1, output_shape),
          grad_cache_(1, input_shape)
    {
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);

        for (size_t i = 0; i < theta_.width() * theta_.height(); ++i)
            theta_[i] = distribution(generator);
    }

    Layer::Layer(const Layer &previous, const size_t output_shape)
        : theta_(previous.output_shape(), output_shape),
          bias_(1, output_shape),
          forward_cache_(1, output_shape),
          weighted_output_(1, output_shape),
          grad_cache_(1, previous.output_shape())
    {
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);

        for (size_t i = 0; i < theta_.width() * theta_.height(); ++i)
            theta_[i] = distribution(generator);
    }

    size_t Layer::input_shape() const
    {
        return theta_.width();
    }

    size_t Layer::output_shape() const
    {
        return theta_.height();
    }

    const Matrix &Layer::get_weights() const
    {
        return theta_;
    }

    const Matrix &Layer::get_bias() const
    {
        return bias_;
    }

    const Matrix &Layer::get_weighted_output() const
    {
        return weighted_output_;
    }

    Matrix Layer::operator()(const Matrix &input)
    {
        assert((input.width() == 1 && input.height() == input_shape()));

        weighted_output_ = theta_ * input + bias_;

        // TODO: Check that sigmoid code uses copy ellision.
        forward_cache_ = input;

        return sigmoid(weighted_output_);
    }

    Matrix Layer::grad(const Matrix &weighted_input, const Matrix &delta)
    {
        assert((delta.width() == 1 && delta.height() == output_shape()));
        assert((weighted_input.width() == 1 && weighted_input.height() == input_shape()));

        theta_.is_transposed = true;

        // TODO: Check that d_sigmoid code uses copy ellision.
        auto result = (theta_ * delta).outer_product(d_sigmoid(weighted_input));
        grad_cache_ = delta;

        theta_.is_transposed = false;

        return result;
    }

    void Layer::update_weights(const double alpha)
    {
        forward_cache_.is_transposed = true;

        theta_ -= (grad_cache_ * forward_cache_) * alpha;
        bias_ -= grad_cache_ * alpha;

        forward_cache_.is_transposed = true;
    }

    std::ostream &operator<<(std::ostream &out, const Layer &rhs)
    {
        out << "Input: " << rhs.input_shape() << "; output: " << rhs.output_shape() << "\n";
        return out;
    }

};