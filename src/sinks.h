#pragma once

#include "neuron.h"

class SummingSink : public Neuron
{
public:
    virtual void write(const Numeric &weight)
    {
        m_weight += weight;
    }

    virtual void reset()
    {
        m_weight = 0;
        m_applied = false;
    }

    virtual void apply(Agent::SP a)
    {
        if (!m_applied)
        {
            m_weight = sigmoid(m_weight);
            _apply(a);
            m_applied = true;
        }
    }
    virtual void _apply(Agent::SP a) = 0;

protected:
    Numeric m_weight;
    bool m_applied;
};

class Sink_Move : public SummingSink
{
public:
    virtual void _apply(Agent::SP a)
    {
        a->move(m_weight * a->velocity());
    };
};

class Sink_Direction : public SummingSink
{
public:
    virtual void _apply(Agent::SP a)
    {
        a->direction(a->direction() + m_weight);
    };
};

class Sink_Velocity : public SummingSink
{
public:
    virtual void _apply(Agent::SP a)
    {
        a->velocity(a->velocity() + m_weight);
    };
};

class Sink_Red : public SummingSink
{
public:
    virtual void _apply(Agent::SP a)
    {
        auto &c = a->colour();
        c.r = std::abs(255 * m_weight);
    };
};

class Sink_Green : public SummingSink
{
public:
    virtual void _apply(Agent::SP a)
    {
        auto &c = a->colour();
        c.g = std::abs(255 * m_weight);
    };
};

class Sink_Blue : public SummingSink
{
public:
    virtual void _apply(Agent::SP a)
    {
        auto &c = a->colour();
        c.b = std::abs(255 * m_weight);
    };
};

class Sink_Size : public SummingSink
{
public:
    virtual void _apply(Agent::SP a)
    {
        const auto &config = getConfig();
        a->size(std::abs((config.MAX_SIZE * m_weight)));
    };
};
