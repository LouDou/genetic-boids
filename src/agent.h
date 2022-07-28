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

    Numeric &size()
    {
        return m_size;
    }

    void size(Numeric next);

    Position &position()
    {
        return m_pos;
    }

    void position(const Position &next);
    void moveX(int delta);
    void moveY(int delta);

    Colour &colour()
    {
        return m_col;
    }

    void colour(const Colour &next);

    Numeric &velocity_x()
    {
        return m_velocity_x;
    }

    void velocity_x(const Numeric &next);

    Numeric &velocity_y()
    {
        return m_velocity_y;
    }

    void velocity_y(const Numeric &next);

private:
    Numeric m_size;
    Position m_pos;
    Colour m_col;

    Numeric m_velocity_x;
    Numeric m_velocity_y;
};
