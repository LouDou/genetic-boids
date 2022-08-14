#include <cmath>

#include "config.h"
#include "conditions.h"

const Numeric Error_DistanceTo(const Position &p, const Position &q, const Numeric &sx, const Numeric &sy)
{
    const auto dx = (p.x - q.x) / sx;
    const auto dy = (p.y - q.y) / sy;
    return std::sqrt(dx * dx + dy * dy);
}

const Numeric Error_DistanceToTL(Agent::SP a)
{
    const auto &config = getConfig();
    return Error_DistanceTo(
        a->position(),
        {0, 0},
        config.SCREEN_WIDTH,
        config.SCREEN_HEIGHT);
}

const Numeric Error_DistanceToTR(Agent::SP a)
{
    const auto &config = getConfig();
    return Error_DistanceTo(
        a->position(),
        {static_cast<Numeric>(config.SCREEN_WIDTH), 0},
        config.SCREEN_WIDTH,
        config.SCREEN_HEIGHT);
}

const Numeric Error_DistanceToBL(Agent::SP a)
{
    const auto &config = getConfig();
    return Error_DistanceTo(
        a->position(),
        {0, static_cast<Numeric>(config.SCREEN_HEIGHT)},
        config.SCREEN_WIDTH,
        config.SCREEN_HEIGHT);
}

const Numeric Error_DistanceToBR(Agent::SP a)
{
    const auto &config = getConfig();
    return Error_DistanceTo(
        a->position(),
        {static_cast<Numeric>(config.SCREEN_WIDTH), static_cast<Numeric>(config.SCREEN_HEIGHT)},
        config.SCREEN_WIDTH,
        config.SCREEN_HEIGHT);
}

const Numeric Error_DistanceToCentre(Agent::SP a)
{
    const auto &config = getConfig();
    const auto &p = a->position();
    return Error_DistanceTo(
        p,
        {static_cast<Numeric>(config.SCREEN_WIDTH) / 2,
         static_cast<Numeric>(config.SCREEN_HEIGHT) / 2},
        config.SCREEN_WIDTH,
        config.SCREEN_HEIGHT);
}

const Numeric Error_Redness(Agent::SP a)
{
    const auto &r = a->colour().r;
    return 1.0 - (r / 255.0);
}

const Numeric Error_Greenness(Agent::SP a)
{
    const auto &g = a->colour().g;
    return 1.0 - (g / 255.0);
}

const Numeric Error_Blueness(Agent::SP a)
{
    const auto &b = a->colour().b;
    return 1.0 - (b / 255.0);
}

const Numeric ErrorFunction(Agent::SP a)
{
    // const auto &config = getConfig();
    // const auto &p = a->position();
    // const auto err = Error_DistanceTo(p, {config.TARGET_X, config.TARGET_Y}, config.SCREEN_WIDTH, config.SCREEN_HEIGHT);
    // return 5.0 * err;
    return 8.0 * Error_DistanceToTL(a) * Error_DistanceToTR(a);
}
