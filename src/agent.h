#pragma once

#include <inttypes.h>
#include <memory>

#include "config.h"

struct Position
{
    Numeric x;
    Numeric y;
};

struct Colour
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

class Agent : public std::enable_shared_from_this<Agent>
{
public:
    using SP = std::shared_ptr<Agent>;

    Agent() {}

    Agent(Agent::SP other);

    size_t &age()
    {
        return m_age;
    }

    void age(const size_t &next)
    {
        m_age = next;
    }

    Numeric &size()
    {
        return m_size;
    }

    void size(Numeric next);

    Numeric &velocity()
    {
        return m_velocity;
    }

    void velocity(const Numeric &next);

    Position &position()
    {
        return m_pos;
    }

    void position(const Position &next);
    void move(int delta);

    Colour &colour()
    {
        return m_col;
    }

    void colour(const Colour &next);

    Numeric &angular_vel()
    {
        return m_angular_vel;
    }

    void angular_vel(const Numeric &next);

    Numeric &direction()
    {
        return m_direction;
    }

    void direction(const Numeric &next);

private:
    size_t m_age;
    Numeric m_size;

    Numeric m_velocity;
    Position m_pos;
    
    Colour m_col;

    Numeric m_angular_vel;
    Numeric m_direction;
};
