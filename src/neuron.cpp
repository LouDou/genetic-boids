#include <cmath>

#include "neuron.h"

Numeric sigmoid(const Numeric &x)
{
    return (x / std::sqrt(1 + (x * x)));
}
