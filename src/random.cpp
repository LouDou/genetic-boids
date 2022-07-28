#include "random.h"

#include <random>

std::default_random_engine randengine;

void random_seed(const int64_t seed)
{
    randengine.seed(seed);
}

std::uniform_real_distribution<double> randdist(0.0, 1.0);
const double randf()
{
    return randdist(randengine);
}

std::uniform_real_distribution<double> bipolarranddist(-1.0, 1.0);
const double bipolarrandf()
{
    return bipolarranddist(randengine);
}
