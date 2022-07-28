#pragma once

#include "neuron.h"
#include "conditions.h"

class Source_West : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return (config.SCREEN_WIDTH - a->position().x) / config.SCREEN_WIDTH;
    };
};

class Source_East : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return 1 - ((config.SCREEN_WIDTH - a->position().x) / config.SCREEN_WIDTH);
    };
};

class Source_North : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return (config.SCREEN_HEIGHT - a->position().y) / config.SCREEN_HEIGHT;
    };
};

class Source_South : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return 1 - ((config.SCREEN_HEIGHT - a->position().y) / config.SCREEN_HEIGHT);
    };
};

class Source_Velocity_X : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return a->velocity_x() / config.MAX_VELOCITY;
    };
};

class Source_Velocity_Y : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return a->velocity_y() / config.MAX_VELOCITY;
    };
};

class Source_Goal_Reached : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return LiveStrategy(a) ? 1 : 0;
    };
};

class Source_Out_of_Bounds : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return LiveStrategy_InBounds(a) ? 1 : 0;
    };
};

class Source_Red : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return a->colour().r / 255.0;
    };
};

class Source_Green : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return a->colour().g / 255.0;
    };
};

class Source_Blue : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return a->colour().b / 255.0;
    };
};

class Source_Size : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return a->size() / config.MAX_SIZE;
    };
};

#if USE_KDTREE
class Source_NumNeighbours : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        const auto &p = a->position();
        const double pt[2] = {p.x, p.y};
        std::vector<std::pair<unsigned int, double>> results;
        const auto count = 1 + kdtree.radiusSearch(pt, 8.0, results, searchParams);
        return results.size() / static_cast<Numeric>(config.NUMBOIDS);
    };
};
#endif

static const std::vector<Neuron::SP> Sources{
    std::make_shared<Source_West>(),
    std::make_shared<Source_East>(),
    std::make_shared<Source_North>(),
    std::make_shared<Source_South>(),
    std::make_shared<Source_Velocity_X>(),
    std::make_shared<Source_Velocity_Y>(),
    // std::make_shared<Source_Goal_Reached>(),
    // std::make_shared<Source_Out_of_Bounds>(),
    std::make_shared<Source_Red>(),
    std::make_shared<Source_Green>(),
    std::make_shared<Source_Blue>(),
    std::make_shared<Source_Size>(),
#if USE_KDTREE
    std::make_shared<Source_NumNeighbours>(),
#endif
};
