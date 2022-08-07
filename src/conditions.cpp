#include <cmath>

#include "config.h"
#include "conditions.h"

const Numeric Error_DistanceToTL(Agent::SP a)
{
    const auto &config = getConfig();
    const auto &p = a->position();
    const auto dx = p.x / config.SCREEN_WIDTH;
    const auto dy = p.y / config.SCREEN_HEIGHT;
    return std::sqrt(dx*dx + dy*dy) * 3.0;
}

const Numeric Error_DistanceToTR(Agent::SP a)
{
    const auto &config = getConfig();
    const auto &p = a->position();
    const auto dx = 1 - (p.x / config.SCREEN_WIDTH);
    const auto dy = p.y / config.SCREEN_HEIGHT;
    return std::sqrt(dx*dx + dy*dy) * 3.0;
}

const Numeric ErrorFunction(Agent::SP a)
{
    const auto posErrorTL = Error_DistanceToTL(a);
    const auto posErrorTR = Error_DistanceToTR(a);
    return posErrorTL * posErrorTR;
}
