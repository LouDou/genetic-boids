#include <cmath>

#include "agent.h"

Agent::Agent(Agent::SP other)
{
    size(other->size());
    position(other->position());
    colour(other->colour());
    velocity(other->velocity());
    angular_vel(other->angular_vel());
}

void Agent::size(Numeric next)
{
    const auto &config = getConfig();
    m_size = std::max(config.MIN_SIZE, std::min(config.MAX_SIZE, next));
}

void Agent::velocity(const Numeric &next)
{
    const auto &config = getConfig();
    m_velocity = next;
    if (m_velocity < -config.MAX_VELOCITY)
    {
        m_velocity = -config.MAX_VELOCITY;
    }
    if (m_velocity > config.MAX_VELOCITY)
    {
        m_velocity = config.MAX_VELOCITY;
    }
}

void Agent::position(const Position &next)
{
    m_pos.x = next.x;
    m_pos.y = next.y;
}

void Agent::move(int delta)
{
    m_pos.x += delta * std::sin(m_direction);
    m_pos.y += delta * std::cos(m_direction);
}

void Agent::colour(const Colour &next)
{
    m_col.r = next.r;
    m_col.g = next.g;
    m_col.b = next.b;
}

void Agent::angular_vel(const Numeric &next)
{
    const auto &config = getConfig();
    m_angular_vel = next;
    if (m_angular_vel < -config.MAX_ANGULAR_VELOCITY)
    {
        m_angular_vel = -config.MAX_ANGULAR_VELOCITY;
    }
    if (m_angular_vel > config.MAX_ANGULAR_VELOCITY)
    {
        m_angular_vel = config.MAX_ANGULAR_VELOCITY;
    }
}

void Agent::direction(const Numeric &next)
{
    m_direction = std::fmod(next, TWOPI);
}