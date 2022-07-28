#include "agent.h"

Agent::Agent(Agent::SP other)
{
    size(other->size());
    position(other->position());
    colour(other->colour());
    velocity_x(other->velocity_x());
    velocity_y(other->velocity_y());
}

void Agent::size(Numeric next)
{
    m_size = std::max(config.MIN_SIZE, std::min(config.MAX_SIZE, next));
}

void Agent::position(const Position &next)
{
    m_pos.x = next.x;
    m_pos.y = next.y;
}

void Agent::moveX(int delta)
{
    m_pos.x += delta;
}

void Agent::moveY(int delta)
{
    m_pos.y += delta;
}

void Agent::colour(const Colour &next)
{
    m_col.r = next.r;
    m_col.g = next.g;
    m_col.b = next.b;
}

void Agent::velocity_x(const Numeric &next)
{
    m_velocity_x = next;
    if (m_velocity_x < -config.MAX_VELOCITY)
    {
        m_velocity_x = -config.MAX_VELOCITY;
    }
    if (m_velocity_x > config.MAX_VELOCITY)
    {
        m_velocity_x = config.MAX_VELOCITY;
    }
}

void Agent::velocity_y(const Numeric &next)
{
    m_velocity_y = next;
    if (m_velocity_y < -config.MAX_VELOCITY)
    {
        m_velocity_y = -config.MAX_VELOCITY;
    }
    if (m_velocity_y > config.MAX_VELOCITY)
    {
        m_velocity_y = config.MAX_VELOCITY;
    }
}
