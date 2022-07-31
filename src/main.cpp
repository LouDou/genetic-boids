#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <nanoflann.hpp>

#include "config.h"
#include "agent.h"
#include "conditions.h"
#include "neuralagent.h"
#include "random.h"
#include "sources.h"
#include "sinks.h"
#include "ui.h"
#include "video.h"

Position RandomPosition(const size_t maxx, const size_t maxy)
{
    Position p;
    p.x = std::abs(maxx * bipolarrandf());
    p.y = std::abs(maxy * bipolarrandf());
    return p;
}

Colour RandomColour()
{
    Colour c;
    c.r = 255 * randf();
    c.g = 255 * randf();
    c.b = 255 * randf();
    return c;
}

// Agent kdtree

struct Population
{
    std::vector<Agent::SP> agents;

#if USE_KDTREE
    inline size_t kdtree_get_point_count() const
    {
        return agents.size();
    }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        auto ent = agents.at(idx);
        auto &pos = ent->position();
        if (dim == 0)
        {
            return pos.x;
        }
        if (dim == 1)
        {
            return pos.y;
        }
        return 0.0;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX &) const { return false; }
#endif
} population;

#if USE_KDTREE
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<Numeric, Population>,
    Population,
    2>
    kdtree_t;

kdtree_t kdtree(2, population, nanoflann::KDTreeSingleIndexAdaptorParams(5));

nanoflann::SearchParams searchParams(32, 0, false);
#endif

void InitialCondition(Agent::SP a)
{
    a->size(config.MIN_SIZE + (randf() * (config.MAX_SIZE - config.MIN_SIZE)));
    a->position(RandomPosition(config.SCREEN_WIDTH, config.SCREEN_HEIGHT));
    a->colour(RandomColour());
    a->direction(randf() * TWOPI);
    a->velocity(bipolarrandf() * config.MAX_VELOCITY);
}

int InitPopulation()
{
    population.agents.clear();

    for (size_t i = 0; i < config.NUMBOIDS; ++i)
    {
        auto a = std::make_shared<NeuralAgent>();
        population.agents.push_back(a);
        InitialCondition(a);

        // randomize brain weights
        auto &b = a->brain();
        for (size_t j = 0; j < b.size(); ++j)
        {
            std::get<1>(b[j]) = bipolarrandf();
        }
    }

    return 0;
}

size_t NUM_SURVIVORS = 0;

int NextGeneration(size_t generation)
{
    std::vector<Agent::SP> survivors;
    // remove dead
    for (auto e : population.agents)
    {
        if (LiveStrategy(e))
        {
            survivors.push_back(e);
        }
    }

    std::cout << "generation " << generation << " survivors = " << survivors.size() << std::endl;

    NUM_SURVIVORS = survivors.size();
    if (NUM_SURVIVORS == 0)
    {
        // re-popluate
        std::cout << "Everyone's dead, Dave. Re-populating in generation " << (generation + 1) << std::endl;
        InitPopulation();
        survivors.swap(population.agents);
    }

    // reproduce;
    // create another full population based on clones of the survivors' brains
    std::vector<Agent::SP> nextpop;
    for (size_t i = 0; i < config.NUMBOIDS; ++i)
    {
        auto cloneFrom = std::static_pointer_cast<NeuralAgent>(survivors[i % survivors.size()]);
        auto a = std::make_shared<NeuralAgent>(cloneFrom);
        auto e = std::static_pointer_cast<Agent>(a);
        nextpop.push_back(e);

        // New initial conditions
        InitialCondition(nextpop[i]);
    }

    // mutate
    for (auto e : nextpop)
    {
        auto a = std::static_pointer_cast<NeuralAgent>(e);
        auto &b = a->brain();
        for (size_t j = 0; j < b.size(); ++j)
        {
            if (config.BOUNDED_WEIGHTS)
            {
                std::get<1>(b[j]) = std::max(
                    -config.MAX_WEIGHT,
                    std::min(
                        config.MAX_WEIGHT,
                        std::get<1>(b[j]) + (bipolarrandf() * config.MUTATION)));
            }
            else
            {

                std::get<1>(b[j]) += (bipolarrandf() * config.MUTATION);
            }
        }
    }

    population.agents.swap(nextpop);
    return 0;
}

// UI

#if USE_KDTREE
int UpdateKdTree()
{
    kdtree.buildIndex();
    return 0;
}
#endif

int UpdateAgents(const size_t &iter)
{
#pragma omp parallel for
    for (auto &entity : population.agents)
    {
        auto a = std::static_pointer_cast<NeuralAgent>(entity);
        a->update(iter);
    }

    return 0;
}

typedef std::chrono::steady_clock::time_point tp;
const auto now = std::chrono::steady_clock::now;
const auto dt = [](tp begin, tp end)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
};

int cleanup(int returnCode)
{
    CleanupSDL();
    CleanupAV();
    return returnCode;
}

int main()
{
    random_seed(config.SEED);

    if (InitSDL() != 0)
    {
        return cleanup(1);
    }

    const auto &uiconfig = GetUIConfig();

    if (InitAV(uiconfig.winWidth, uiconfig.winHeight) != 0)
    {
        return cleanup(1);
    }

    if (InitPopulation() != 0)
    {
        return cleanup(1);
    }

    tp t_start = now();
    tp t_iter = t_start;

    long f = 0;
    double t = 0;
    for (size_t g = 0; g < config.MAX_GENS; g++)
    {
        for (size_t i = 0; i < config.GEN_ITERS; ++i, ++f, t_iter = now(), t = dt(t_start, t_iter) / 1000.0)
        {
#if USE_KDTREE
            UpdateKdTree();
#endif
            if (ProcessEvents() != 0)
            {
                return cleanup(1);
            }

            if (UpdateAgents(i) != 0)
            {
                std::cerr << "error updating entt" << std::endl;
                return cleanup(1);
            }

            if (Render(population.agents, g, i, f, t, NUM_SURVIVORS) != 0)
            {
                std::cerr << "error rendering: " << SDL_GetError() << std::endl;
                return cleanup(1);
            }

            // slow down for real-time animation 1/REALTIME_EVERY_NGENS generations,
            // but not the first
            if (g != 0 && g % config.REALTIME_EVERY_NGENS == 0)
            {
                const auto t_render = now();
                const auto dt_render = dt(t_iter, t_render);
                const auto delay = (1000 / 24.) - dt_render;
                // std::cout << " rt delay = " << delay << std::endl;
                if (delay > 0)
                {
                    SDL_Delay(delay);
                }
            }
        }

        if (NextGeneration(g))
        {
            return cleanup(1);
        }
    }

    return cleanup(0);
}
