#pragma once

#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include "agent.h"
#include "neuron.h"
#include "sources.h"
#include "sinks.h"

// Special Neurons

class SummingMemoryNeuron : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return m_val;
    };
    virtual void write(const Numeric &weight)
    {
        m_val += weight;
    };
    virtual void reset()
    {
        m_val = 0;
    };

private:
    Numeric m_val;
};

class SummingSigmoidMemoryNeuron : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return sigmoid(m_val);
    };
    virtual void write(const Numeric &weight)
    {
        m_val += weight;
    };
    virtual void reset()
    {
        m_val = 0;
    };

private:
    Numeric m_val;
};

class MaxMemoryNeuron : public Neuron
{
public:
    virtual const Numeric read(Agent::SP a, const Numeric &weight)
    {
        return m_val;
    };
    virtual void write(const Numeric &weight)
    {
        m_val = std::abs(weight) > std::abs(m_val) ? weight : m_val;
    };
    virtual void reset()
    {
        m_val = 0;
    };

private:
    Numeric m_val;
};

// Brain

using BrainConnection = std::tuple<Neuron::SP, Numeric, Neuron::SP>;
using Brain = std::vector<BrainConnection>;

// Agent

class NeuralAgent : public Agent
{
public:
    using SP = std::shared_ptr<NeuralAgent>;

    NeuralAgent();

    NeuralAgent(const NeuralAgent::SP other);

    Brain &brain()
    {
        return m_brain;
    }

    const Brain &brain() const
    {
        return m_brain;
    }

    std::vector<Numeric> &weight_delta()
    {
        return m_weight_delta;
    }

    void update(const size_t &iter);

    void updateType(const NeuralUpdateType &next)
    {
        m_updateType = next;
    }

    const NeuralUpdateType &updateType() const
    {
        return m_updateType;
    }

    void brainType(const NeuralBrainType &next)
    {
        m_brainType = next;
    }

    const NeuralBrainType &brainType() const
    {
        return m_brainType;
    }

private:
    // Update strategies

    void update_Max();
    void update_Threshold();
    void update_Every();

    // Neuron management

    void resetNeurons();
    void applySinkValues();

    // Brain strategies

    void setupBrain_no_memory();
    void setupBrain_layered_memory();
    void setupBrain_fully_connected_memory();
    void setupBrain();

private:
    Brain m_brain;
    std::vector<Numeric> m_weight_delta;
    NeuralUpdateType m_updateType = NeuralUpdateType::EVERY;
    NeuralBrainType m_brainType = NeuralBrainType::LAYERED;
    std::vector<Neuron::SP> m_sources;
    std::vector<Neuron::SP> m_sinks;
    std::vector<Neuron::SP> m_memory;
};
