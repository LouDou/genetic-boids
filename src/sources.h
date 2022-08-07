#pragma once

#include "neuron.h"
#include "conditions.h"

class Source_Age : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        const auto &config = getConfig();
        return static_cast<Numeric>(a->age()) / config.GEN_ITERS;
    };
};

class Source_Velocity : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        const auto &config = getConfig();
        return a->velocity() / config.MAX_VELOCITY;
    };
};

class Source_West : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        const auto &config = getConfig();
        return (config.SCREEN_WIDTH - a->position().x) / config.SCREEN_WIDTH;
    };
};

class Source_East : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        const auto &config = getConfig();
        return 1 - ((config.SCREEN_WIDTH - a->position().x) / config.SCREEN_WIDTH);
    };
};

class Source_North : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        const auto &config = getConfig();
        return (config.SCREEN_HEIGHT - a->position().y) / config.SCREEN_HEIGHT;
    };
};

class Source_South : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        const auto &config = getConfig();
        return 1 - ((config.SCREEN_HEIGHT - a->position().y) / config.SCREEN_HEIGHT);
    };
};

class Source_Angular_Velocity : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        const auto &config = getConfig();
        return a->angular_vel() / config.MAX_ANGULAR_VELOCITY;
    };
};

class Source_Direction : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return a->direction() / TWOPI;
    };
};

class Source_Error : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return ErrorFunction(a);
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
        const auto &config = getConfig();
        return a->size() / config.MAX_SIZE;
    };
};

const NeuronRegistry &getSources();
