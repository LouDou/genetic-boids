#pragma once

#include "neuron.h"

class Sink_Position_X : public Neuron
{
public:
    virtual void write(Agent::SP a, const Numeric &weight)
    {
        a->moveX(weight * a->velocity_x());
    };
};

class Sink_Position_Y : public Neuron
{
public:
    virtual void write(Agent::SP a, const Numeric &weight)
    {
        a->moveY(weight * a->velocity_y());
    };
};

class Sink_Velocity_X : public Neuron
{
public:
    virtual void write(Agent::SP a, const Numeric &weight)
    {
        a->velocity_x(a->velocity_x() + weight);
    };
};

class Sink_Velocity_Y : public Neuron
{
public:
    virtual void write(Agent::SP a, const Numeric &weight)
    {
        a->velocity_y(a->velocity_y() + weight);
    };
};

class Sink_Red : public Neuron
{
public:
    virtual void write(Agent::SP a, const Numeric &weight)
    {
        auto &c = a->colour();
        c.r = std::abs(255 * weight);
    };
};

class Sink_Green : public Neuron
{
public:
    virtual void write(Agent::SP a, const Numeric &weight)
    {
        auto &c = a->colour();
        c.g = std::abs(255 * weight);
    };
};

class Sink_Blue : public Neuron
{
public:
    virtual void write(Agent::SP a, const Numeric &weight)
    {
        auto &c = a->colour();
        c.b = std::abs(255 * weight);
    };
};

class Sink_Size : public Neuron
{
public:
    virtual void write(Agent::SP a, const Numeric &weight)
    {
        a->size(std::abs((config.MAX_SIZE * weight)));
    };
};

static const std::vector<Neuron::SP> Sinks{
    std::make_shared<Sink_Position_X>(),
    std::make_shared<Sink_Position_Y>(),
    std::make_shared<Sink_Velocity_X>(),
    std::make_shared<Sink_Velocity_Y>(),
    std::make_shared<Sink_Red>(),
    std::make_shared<Sink_Green>(),
    std::make_shared<Sink_Blue>(),
    std::make_shared<Sink_Size>(),
};
