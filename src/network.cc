#include "network.hpp"
#include "activations.hpp"
#include <cassert>
#include <iostream>

namespace ml
{
    Matrix mse(Matrix matrix, Matrix target)
    {
        return target - matrix;
    }

    Network::Network(std::vector<Layer> layers)
        : layers_(layers)
    {
    }

    Layer &Network::get_layer(const size_t i)
    {
        assert((i < layers_.size()));

        return layers_[i];
    }

    Matrix Network::operator()(const Matrix &input)
    {
        assert((layers_.size() > 0));

        auto temp = layers_[0](input);
        for (size_t i = 1; i < layers_.size(); ++i)
            temp = layers_[i](temp);

        return temp;
    }

    double Network::fit(const Matrix &input, const Matrix &target)
    {
        assert((layers_.size() > 0));
        assert((layers_.back().output_shape() == target.height()));

        auto error = mse((*this)(input), target).outer_product(d_sigmoid(layers_.back().get_weighted_output()));

        double error_mag = 0;
        for (size_t i = 0; i < error.width() * error.height(); ++i)
            error_mag += std::abs(error[i]);

        std::vector<Matrix> weighted_inputs{input};

        for (size_t i = 0; i < layers_.size() - 1; ++i)
            weighted_inputs.push_back(layers_[i].get_weighted_output());

        for (int i = layers_.size() - 1; i >= 0; --i)
            error = layers_[i].grad(weighted_inputs[i], error);

        for (auto &l : layers_)
            l.update_weights(0.01);

        return error_mag;
    }
}
