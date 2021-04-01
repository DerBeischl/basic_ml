#include "network.hpp"
#include "layer.hpp"
#include <vector>
#include <iostream>

int main()
{
    ml::Layer layer(5, 3);
    ml::Layer layer2(3, 1);
    ml::Network network({layer, layer2});

    ml::Matrix input({1, 1, 1, 1, 1});
    ml::Matrix target({0});

    for (size_t epoch = 0; epoch < 100; ++epoch)
        std::cout << "Epoch: " << epoch << " error: " << network.fit(input, target) << "\n";
}