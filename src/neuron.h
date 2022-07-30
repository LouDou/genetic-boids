#pragma once

#include <memory>

#include "config.h"
#include "agent.h"

class Neuron
{
public:
    using SP = std::shared_ptr<Neuron>;

    virtual const Numeric read(Agent::SP a, const Numeric &weight) { return 0.0; };
    virtual void write(const Numeric &weight){};
    virtual void reset(){};
    virtual void apply(Agent::SP a){};
};

Numeric sigmoid(const Numeric &x);
