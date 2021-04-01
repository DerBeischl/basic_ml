#include "network.hpp"
#include "layer.hpp"
#include "random.hpp"
#include <vector>
#include <iostream>

int main()
{
    ml::Layer layer1(2, 4);
    ml::Layer layer2(4, 4);
    ml::Layer layer3(4, 2);

    ml::Network network({layer1, layer2, layer3});

    std::uniform_int_distribution<int> dist(0, 1);

    for (size_t epoch = 0; epoch < 2000; ++epoch)
    {
        double x1 = dist(ml::generator);
        double x2 = dist(ml::generator);

        ml::Matrix input({x1, x2});
        ml::Matrix target({(double)(x1 != x2), (double)(x1 == x2)});
        std::cout << "Epoch: " << epoch << " error: " << network.fit(input, target) << "\n";
    }
}
