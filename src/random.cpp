#include "random.h"

#include <random>

std::default_random_engine randengine;

void random_seed(const int64_t seed)
{
    randengine.seed(seed);
}

std::uniform_real_distribution<Numeric> randdist(0.0, 1.0);
const Numeric randf()
{
    return randdist(randengine);
}

std::uniform_real_distribution<Numeric> bipolarranddist(-1.0, 1.0);
const Numeric bipolarrandf()
{
    return bipolarranddist(randengine);
}
