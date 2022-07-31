#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

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

using NeuronFactory = std::function<Neuron::SP()>;
using NeuronRegistry = std::unordered_map<std::string, NeuronFactory>;

Numeric sigmoid(const Numeric &x);
