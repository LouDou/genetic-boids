#include "neuralagent.h"

NeuralAgent::NeuralAgent() : Agent()
{
    setupBrain();
}

NeuralAgent::NeuralAgent(const NeuralAgent::SP other) : Agent(other)
{
    updateType(other->updateType());
    brainType(other->brainType());

    setupBrain();

    // copy brain weights
    const auto &b = other->brain();
    // std::cout << "NA copy my brain = " << m_brain.size() << " other brain = " << b.size() << std::endl;
    for (size_t i = 0; i < m_brain.size(); ++i)
    {
        std::get<1>(m_brain[i]) = std::get<1>(b[i]);
    }
}

// Update strategies

void NeuralAgent::update(const size_t &iter)
{
    age(iter);
    resetNeurons();
    switch (m_updateType)
    {
    case NeuralUpdateType::MAX:
        update_Max();
        break;
    case NeuralUpdateType::THRESHOLD:
        update_Threshold();
        break;
    case NeuralUpdateType::EVERY:
        update_Every();
        break;
    }
    applySinkValues();
}

void NeuralAgent::update_Max()
{
    Numeric maxabsval = -1;
    int maxidx = -1;

    // calculate neuron activation values
    for (size_t i = 0; i < m_brain.size(); ++i)
    {
        const auto &[src, w, snk] = m_brain[i];
        const auto val = src->read(shared_from_this(), w) * w;
        // std::cout << "w=" << w << " val=" << val << " maxval=" << maxval << std::endl;
        // find maximally activated sink
        const auto absval = std::abs(val);
        if (absval > maxabsval)
        {
            maxabsval = absval;
            maxidx = i;
        }
    }

    if (maxidx > -1 && maxidx < m_brain.size())
    {
        const auto [src, w, snk] = m_brain[maxidx];
        snk->write(w);
    }
}

void NeuralAgent::update_Threshold()
{
    const auto &config = getConfig();
    // calculate neuron activation values
    for (size_t i = 0; i < m_brain.size(); ++i)
    {
        const auto &[src, w, snk] = m_brain[i];
        const auto val = src->read(shared_from_this(), w) * w;
        // activate above threshold
        if (std::abs(val) > config.NEURAL_THRESHOLD)
        {
            snk->write(val);
        }
    }
}

void NeuralAgent::update_Every()
{
    // calculate neuron activation values
    for (size_t i = 0; i < m_brain.size(); ++i)
    {
        const auto &[src, w, snk] = m_brain[i];
        const auto val = src->read(shared_from_this(), w) * w;
        snk->write(val);
    }
}

// Memory management

void NeuralAgent::resetNeurons()
{
    for (size_t i = 0; i < m_brain.size(); ++i)
    {
        const auto &[src, w, snk] = m_brain[i];
        src->reset();
        snk->reset();
    }
}

void NeuralAgent::applySinkValues()
{
    for (size_t i = 0; i < m_brain.size(); ++i)
    {
        const auto &[src, w, snk] = m_brain[i];
        snk->apply(shared_from_this());
    }
}

// Brain strategies

void NeuralAgent::setupBrain_no_memory()
{
    for (size_t i = 0; i < m_sources.size(); ++i)
    {
        auto src = m_sources[i];
        for (size_t j = 0; j < m_sinks.size(); ++j)
        {
            auto snk = m_sinks[j];
            BrainConnection c(src, 0.f, snk);
            m_brain.push_back(c);
        }
    }
}

void NeuralAgent::setupBrain_layered_memory()
{
    const auto &config = getConfig();
    for (size_t i = 0; i < config.NUM_MEMORY_LAYERS * config.NUM_MEMORY_PER_LAYER; ++i)
    {
        m_memory.push_back(std::make_shared<SummingSigmoidMemoryNeuron>());
    }
    // std::cout << " total mem neurons " << m_memory.size() << std::endl;

    // connect every source to every memory neuron in the first layer
    for (size_t i = 0; i < m_sources.size(); ++i)
    {
        auto src = m_sources[i];
        for (size_t j = 0; j < config.NUM_MEMORY_PER_LAYER; ++j)
        {
            auto m = m_memory[j];
            // std::cout << " connect src " << i << " to mem " << j << std::endl;
            BrainConnection c(src, 0.f, m);
            m_brain.push_back(c);
        }
    }

    // there are no direct Source - Sink connections

    // connect each layer to the next
    for (size_t w = 0; w < config.NUM_MEMORY_LAYERS - 1; ++w)
    {
        for (size_t i = 0; i < config.NUM_MEMORY_PER_LAYER; ++i)
        {
            const auto im1 = i + (w * config.NUM_MEMORY_PER_LAYER);
            auto m1 = m_memory[im1];
            for (size_t j = 0; j < config.NUM_MEMORY_PER_LAYER; ++j)
            {
                const auto im2 = j + ((w + 1) * config.NUM_MEMORY_PER_LAYER);
                auto m2 = m_memory[im2];
                // std::cout << " connect mem " << im1 << " to mem " << im2 << std::endl;
                BrainConnection c(m1, 0.f, m2);
                m_brain.push_back(c);
            }
        }
    }

    // connect every memory neuron in the last layer to every sink
    for (size_t i = 0; i < config.NUM_MEMORY_PER_LAYER; ++i)
    {
        const auto im = i + ((config.NUM_MEMORY_LAYERS - 1) * config.NUM_MEMORY_PER_LAYER);
        auto m = m_memory[im];
        for (size_t j = 0; j < m_sinks.size(); ++j)
        {
            auto snk = m_sinks[j];
            // std::cout << " connect mem " << im << " to sink " << j << std::endl;
            BrainConnection c(m, 0.f, snk);
            m_brain.push_back(c);
        }
    }
}

void NeuralAgent::setupBrain_fully_connected_memory()
{
    const auto &config = getConfig();
    for (size_t i = 0; i < config.NUM_MEMORY_LAYERS * config.NUM_MEMORY_PER_LAYER; ++i)
    {
        m_memory.push_back(std::make_shared<SummingSigmoidMemoryNeuron>());
    }

    // the order of connection is important;
    // we want to perform all memory writes
    // before any memory reads

    for (size_t i = 0; i < m_sources.size(); ++i)
    {
        auto src = m_sources[i];
        // connect all sources and sinks
        for (size_t j = 0; j < m_sinks.size(); ++j)
        {
            auto snk = m_sinks[j];
            BrainConnection c(src, 0.f, snk);
            m_brain.push_back(c);
        }
        // connect every source to every memory neuron
        for (size_t k = 0; k < m_memory.size(); ++k)
        {
            auto m = m_memory[k];
            BrainConnection c(src, 0.f, m);
            m_brain.push_back(c);
        }
    }

    // connect all memory neurons together;
    // this is both read and write on memory;
    // is this consistent?
    for (size_t i = 0; i < m_memory.size(); ++i)
    {
        for (size_t j = 0; j < m_memory.size(); ++j)
        {
            auto m1 = m_memory[i];
            auto m2 = m_memory[j];
            BrainConnection c1(m1, 0.f, m2);
            m_brain.push_back(c1);
        }
    }

    // connect all memory neurons to all sinks
    for (size_t i = 0; i < m_memory.size(); ++i)
    {
        auto m = m_memory[i];
        for (size_t j = 0; j < m_sinks.size(); ++j)
        {
            auto snk = m_sinks[j];
            BrainConnection c(m, 0.f, snk);
            m_brain.push_back(c);
        }
    }
}

void NeuralAgent::setupBrain()
{
    const auto &config = getConfig();

    m_brain.clear();
    m_sources.clear();
    m_sinks.clear();
    m_memory.clear();

    const auto &sourceRegistry = getSources();
    const auto &sinkRegistry = getSinks();

    // Create sources and sinks
    std::vector<Neuron::SP> sources;
    for (const auto &sourceName : config.NEURON_SOURCES)
    {
        auto sf = sourceRegistry.at(sourceName);
        sources.push_back(sf());
    }
    m_sources.swap(sources);

    std::vector<Neuron::SP> sinks;
    for (const auto &sourceName : config.NEURON_SINKS)
    {
        auto sf = sinkRegistry.at(sourceName);
        sinks.push_back(sf());
    }
    m_sinks.swap(sinks);

    switch (m_brainType)
    {
    case NeuralBrainType::NO_MEMORY:
        setupBrain_no_memory();
        break;
    case NeuralBrainType::LAYERED:
        setupBrain_layered_memory();
        break;
    case NeuralBrainType::FULLY_CONNECTED:
        setupBrain_fully_connected_memory();
        break;
    }
}
