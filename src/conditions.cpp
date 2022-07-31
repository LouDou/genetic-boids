#include <cmath>

#include "config.h"
#include "conditions.h"

const bool LiveStrategy_LeftHalf(Agent::SP a)
{
    return a->position().x < config.SCREEN_WIDTH / 2.f;
}

const bool LiveStrategy_RightHalf(Agent::SP a)
{
    return a->position().x > config.SCREEN_WIDTH / 2.f;
}

const bool LiveStrategy_CentreThirdBox(Agent::SP a)
{
    const auto &p = a->position();
    const bool validX = (p.x > (config.SCREEN_WIDTH * 1.f / 3.)) && (p.x < (config.SCREEN_WIDTH * 2.f / 3.));
    const bool validY = (p.y > (config.SCREEN_HEIGHT * 1.f / 3.)) && (p.y < (config.SCREEN_HEIGHT * 2.f / 3.));
    return validX && validY;
}

const bool LiveStrategy_CentreFifthBox(Agent::SP a)
{
    const auto &p = a->position();
    const bool validX = (p.x > (config.SCREEN_WIDTH * 2.f / 5.)) && (p.x < (config.SCREEN_WIDTH * 3.f / 5.));
    const bool validY = (p.y > (config.SCREEN_HEIGHT * 2.f / 5.)) && (p.y < (config.SCREEN_HEIGHT * 3.f / 5.));
    return validX && validY;
}

const bool LiveStrategy_CentreTenthBox(Agent::SP a)
{
    const auto &p = a->position();
    const bool validX = (p.x > (config.SCREEN_WIDTH * 4.5f / 10.)) && (p.x < (config.SCREEN_WIDTH * 5.5f / 10.));
    const bool validY = (p.y > (config.SCREEN_HEIGHT * 4.5f / 10.)) && (p.y < (config.SCREEN_HEIGHT * 5.5 / 10.));
    return validX && validY;
}

const bool LiveStrategy_OffCentreTenthBox1(Agent::SP a)
{
    const auto &p = a->position();
    const bool validX = (p.x > (config.SCREEN_WIDTH * 3.5f / 10.)) && (p.x < (config.SCREEN_WIDTH * 4.5f / 10.));
    const bool validY = (p.y > (config.SCREEN_HEIGHT * 6.5f / 10.)) && (p.y < (config.SCREEN_HEIGHT * 7.5 / 10.));
    return validX && validY;
}

const bool LiveStrategy_OffCentreTenthBox2(Agent::SP a)
{
    const auto &p = a->position();
    const bool validX = (p.x > (config.SCREEN_WIDTH * 6.5f / 10.)) && (p.x < (config.SCREEN_WIDTH * 7.5f / 10.));
    const bool validY = (p.y > (config.SCREEN_HEIGHT * 3.5f / 10.)) && (p.y < (config.SCREEN_HEIGHT * 4.5 / 10.));
    return validX && validY;
}

const bool LiveStrategy_CentreTwentiethBox(Agent::SP a)
{
    const auto &p = a->position();
    const bool validX = (p.x > (config.SCREEN_WIDTH * 9.5f / 20.)) && (p.x < (config.SCREEN_WIDTH * 10.5f / 20.));
    const bool validY = (p.y > (config.SCREEN_HEIGHT * 9.5f / 20.)) && (p.y < (config.SCREEN_HEIGHT * 10.5 / 20.));
    return validX && validY;
}

const bool LiveStrategy_OffCentreTwentiethBox(Agent::SP a)
{
    const auto &p = a->position();
    const bool validX = (p.x > (config.SCREEN_WIDTH * 3.5f / 20.)) && (p.x < (config.SCREEN_WIDTH * 4.5f / 20.));
    const bool validY = (p.y > (config.SCREEN_HEIGHT * 16.5f / 20.)) && (p.y < (config.SCREEN_HEIGHT * 17.5 / 20.));
    return validX && validY;
}

const bool LiveStrategy_LeftRightTenth(Agent::SP a)
{
    auto &p = a->position();
    bool valid = (p.x < (config.SCREEN_WIDTH * 0.1f / 10.)) || (p.x > (config.SCREEN_WIDTH * 0.9f / 10.));
    return valid;
}

const bool LiveStrategy_TopBottomTenth(Agent::SP a)
{
    const auto &p = a->position();
    const bool valid = (p.y < (config.SCREEN_HEIGHT * 0.1f / 10.)) || (p.y > (config.SCREEN_HEIGHT * 0.9f / 10.));
    return valid;
}

const bool LiveStrategy_TLTenth(Agent::SP a)
{
    const auto &p = a->position();
    const bool validX = p.x < (config.SCREEN_WIDTH * 0.1f);
    const bool validY = p.y < (config.SCREEN_HEIGHT * 0.1f);
    return validX && validY;
}

const bool LiveStrategy_LowVelocity(Agent::SP a)
{
    return a->velocity() < (config.MAX_VELOCITY / 10.0);
}

const bool LiveStrategy_TLCircle(Agent::SP a)
{
    const auto &p = a->position();
    return std::sqrt(p.x * p.x + p.y * p.y) < (config.SCREEN_WIDTH / 8.f);
}

const bool LiveStrategy_TRCircle(Agent::SP a)
{
    const auto &p = a->position();
    const auto dx = p.x - config.SCREEN_WIDTH;
    return std::sqrt(dx * dx + p.y * p.y) < (config.SCREEN_WIDTH / 8.f);
}

const bool LiveStrategy_TopCorners(Agent::SP a)
{
    return LiveStrategy_TLCircle(a) || LiveStrategy_TRCircle(a);
}

const bool LiveStrategy_BLCircle(Agent::SP a)
{
    const auto &p = a->position();
    const auto dy = p.y - config.SCREEN_HEIGHT;
    return std::sqrt(p.x * p.x + dy * dy) < (config.SCREEN_WIDTH / 8.f);
}

const bool LiveStrategy_BRCircle(Agent::SP a)
{
    const auto &p = a->position();
    const auto dx = p.x - config.SCREEN_WIDTH;
    const auto dy = p.y - config.SCREEN_HEIGHT;
    return std::sqrt(dx * dx + dy * dy) < (config.SCREEN_WIDTH / 8.f);
}

const bool LiveStrategy_BottomCorners(Agent::SP a)
{
    return LiveStrategy_BLCircle(a) || LiveStrategy_BRCircle(a);
}

const bool LiveStrategy_Corners(Agent::SP a)
{
    return LiveStrategy_TopCorners(a) || LiveStrategy_BottomCorners(a);
}

const bool LiveStrategy_HasVelocity(Agent::SP a)
{
    return a->velocity() > 0.001f;
}

const bool LiveStrategy_HorizTenths(Agent::SP a)
{
    const auto &p = a->position();
    return (static_cast<int>(std::round(p.x / 10.f)) % 2) == 0;
}

const bool LiveStrategy_VertTenths(Agent::SP a)
{
    const auto &p = a->position();
    return (static_cast<int>(std::round(p.y / 10.f)) % 2) == 0;
}

const bool LiveStrategy_InBounds(Agent::SP a)
{
    const auto &p = a->position();
    return p.x > 0 && p.x < config.SCREEN_WIDTH && p.y > 0 && p.y < config.SCREEN_HEIGHT;
}

const bool LiveStrategy_IsRed(Agent::SP a)
{
    const auto &c = a->colour();
    return (c.r / 2.0) > c.g & (c.r / 2.0) > c.b;
}

const bool LiveStrategy_IsGreen(Agent::SP a)
{
    const auto &c = a->colour();
    return (c.g / 2.0) > c.r & (c.g / 2.0) > c.b;
}

const bool LiveStrategy_IsBlue(Agent::SP a)
{
    const auto &c = a->colour();
    return (c.b / 2.0) > c.r & (c.b / 2.0) > c.g;
}

const bool LiveStrategy_IsLarge(Agent::SP a)
{
    return a->size() > config.MIN_SIZE + ((config.MAX_SIZE - config.MIN_SIZE) * 0.8);
}

const bool LiveStrategy_IsSmall(Agent::SP a)
{
    return a->size() < config.MIN_SIZE + ((config.MAX_SIZE - config.MIN_SIZE) * 0.2);
}

const bool LiveStrategy_StuckOnBorder(Agent::SP a)
{
    const auto &p = a->position();
    const auto mX = config.SCREEN_WIDTH / 25.0;
    const auto mY = config.SCREEN_HEIGHT / 25.0;
    const bool stuckX = std::abs(p.x) < mX || std::abs(config.SCREEN_WIDTH - p.x) < mX;
    const bool stuckY = std::abs(p.y) < mY || std::abs(config.SCREEN_HEIGHT - p.y) < mY;
    return stuckX || stuckY;
}

const bool LiveStrategy(Agent::SP a)
{
    return LiveStrategy_IsSmall(a) &&(
        (LiveStrategy_TopCorners(a) && LiveStrategy_IsRed(a))
        ||
        (LiveStrategy_BottomCorners(a) && LiveStrategy_IsGreen(a))
    );
}
