#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <fontconfig/fontconfig.h>

#include <nanoflann.hpp>

#define USE_KDTREE 0

using Numeric = double;

constexpr uint SCREEN_WIDTH = 1800;
constexpr uint SCREEN_HEIGHT = 1800;

constexpr uint NUMBOIDS = 5000;
constexpr Numeric MUTATION = 0.008f;
constexpr Numeric NEURAL_THRESHOLD = 0.12f;
constexpr size_t MAX_GENS = 12000;
constexpr size_t GEN_ITERS = 500;
constexpr Numeric MAX_VELOCITY = 4.5f;
constexpr size_t NUM_MEMORY = 4;

size_t NUM_SURVIVORS = 0;

SDL_Window *window = nullptr;
SDL_Renderer *render = nullptr;
TTF_Font *sans = nullptr;
SDL_Event event;

std::default_random_engine randengine;
std::uniform_real_distribution<double> randdist(0.0, 1.0);
double randf()
{
    return randdist(randengine);
}

std::uniform_real_distribution<double> bipolarranddist(-1.0, 1.0);
double bipolarrandf()
{
    return bipolarranddist(randengine);
}

using Position = SDL_FPoint;

Position RandomPosition(const size_t maxx, const size_t maxy)
{
    Position p;
    p.x = std::abs(maxx * bipolarrandf());
    p.y = std::abs(maxy * bipolarrandf());
    return p;
}

struct Colour
{
    uint r;
    uint g;
    uint b;
};

Colour RandomColour()
{
    Colour c;
    c.r = 255 * randf();
    c.g = 255 * randf();
    c.b = 255 * randf();
    return c;
}

class Agent {
public:

    using SP = std::shared_ptr<Agent>;

    Agent() { }

    Agent(Agent::SP other) {
        position(other->position());
        colour(other->colour());
        velocity_x(other->velocity_x());
        velocity_y(other->velocity_y());
    }

    Position& position() {
        return m_pos;
    }

    void position(const Position &next) {
        m_pos.x = next.x;
        m_pos.y = next.y;
    }

    void moveX(int delta) {
        m_pos.x += delta;
    }

    void moveY(int delta) {
        m_pos.y += delta;
    }

    Colour& colour() {
        return m_col;
    }

    void colour(const Colour &next) {
        m_col.r = next.r;
        m_col.g = next.g;
        m_col.b = next.b;
    }

    Numeric& velocity_x() {
        return m_velocity_x;
    }

    void velocity_x(const Numeric &next) {
        m_velocity_x = next;
        if (m_velocity_x < -MAX_VELOCITY) {
            m_velocity_x = -MAX_VELOCITY;
        }
        if (m_velocity_x > MAX_VELOCITY) {
            m_velocity_x = MAX_VELOCITY;
        }
    }

    Numeric& velocity_y() {
        return m_velocity_y;
    }

    void velocity_y(const Numeric &next) {
        m_velocity_y = next;
        if (m_velocity_y < -MAX_VELOCITY) {
            m_velocity_y = -MAX_VELOCITY;
        }
        if (m_velocity_y > MAX_VELOCITY) {
            m_velocity_y = MAX_VELOCITY;
        }
    }

private:
    Position m_pos;
    Colour m_col;

    Numeric m_velocity_x;
    Numeric m_velocity_y;
};

// Agent kdtree

struct Population
{
    std::vector<Agent::SP> agents;

    inline size_t kdtree_get_point_count() const { return agents.size(); }

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

// Neurons

class Neuron {
public:
    using SP = std::shared_ptr<Neuron>;

    virtual Numeric calculate(Agent *a, Numeric weight, bool read) = 0;

};

// Source neurons

class Source_West : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        return 1 - (a->position().x / SCREEN_WIDTH);
    }
};

class Source_East : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        return (a->position().x / SCREEN_WIDTH);
    }
};

class Source_North : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        return 1 - (a->position().y / SCREEN_HEIGHT);
    }
};

class Source_South : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        return (a->position().y / SCREEN_HEIGHT);
    }
};

class Source_Velocity_X : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        return a->velocity_x() / MAX_VELOCITY;
    }
};

class Source_Velocity_Y : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        return a->velocity_y() / MAX_VELOCITY;
    }
};

#if USE_KDTREE
class Source_NumNeighbours : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        const auto &p = a->position();
        const double pt[2] = {p.x, p.y};
        std::vector<std::pair<unsigned int, double>> results;
        const auto count = 1 + kdtree.radiusSearch(pt, 8.0, results, searchParams);
        return results.size() / static_cast<Numeric>(NUMBOIDS);
    }
};
#endif

std::vector<Neuron::SP> Sources;

int InitSources() {
    Sources.push_back(std::make_shared<Source_West>());
    Sources.push_back(std::make_shared<Source_East>());
    Sources.push_back(std::make_shared<Source_North>());
    Sources.push_back(std::make_shared<Source_South>());
    Sources.push_back(std::make_shared<Source_Velocity_X>());
    Sources.push_back(std::make_shared<Source_Velocity_Y>());

#if USE_KDTREE
    Sources.push_back(std::make_shared<Source_NumNeighbours>());
#endif

    return 0;
}

// Sink neurons

class Sink_Move_Horizontal : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        a->moveX(weight * a->velocity_x());
        return 0.f;
    }
};

class Sink_Move_Vertical : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        a->moveY(weight * a->velocity_y());
        return 0.f;
    }
};

class Sink_Velocity_X : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        a->velocity_x(a->velocity_x() + weight);
        return 0.f;
    }
};

class Sink_Velocity_Y : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        a->velocity_y(a->velocity_y() + weight);
        return 0.f;
    }
};

static std::vector<Neuron::SP> Sinks;

int InitSinks() {
    Sinks.push_back(std::make_shared<Sink_Velocity_X>());
    Sinks.push_back(std::make_shared<Sink_Velocity_Y>());
    Sinks.push_back(std::make_shared<Sink_Move_Horizontal>());
    Sinks.push_back(std::make_shared<Sink_Move_Vertical>());

    return 0;
}

// Special Neurons

class MemoryNeuron : public Neuron {
public:
    virtual Numeric calculate(Agent *a, Numeric weight, bool read) {
        if (!read) // write
        {
            m_val = weight;
        }
        return m_val;
    }

private:
    Numeric m_val;
};

// Agents

using BrainConnection = std::tuple<Neuron::SP, Numeric, Neuron::SP>;
using Brain = std::vector<BrainConnection>;

class NeuralAgent : public Agent {
public:
    using SP = std::shared_ptr<NeuralAgent>;

    NeuralAgent() : Agent() {
        setupBrain();
    }

    NeuralAgent(const NeuralAgent::SP other) : Agent(other) {
        setupBrain();
        // copy brain weights
        const auto &b = other->brain();
        // std::cout << "NA copy my brain = " << m_brain.size() << " other brain = " << b.size() << std::endl;
        for (size_t i = 0; i < m_brain.size(); ++i) {
            std::get<1>(m_brain[i]) = std::get<1>(b[i]);
        }
    }

    Brain& brain() {
        return m_brain;
    }

    const Brain& brain() const {
        return m_brain;
    }

    void update_Max() {
        Numeric maxval = -1;
        int maxidx = -1;

        // calculate neuron activation values
        for (size_t i = 0; i < m_brain.size(); ++i) {
            auto &[src, w, snk] = m_brain[i];
            auto val = src->calculate(this, w, true) * w;
            // std::cout << "w=" << w << " val=" << val << " maxval=" << maxval << std::endl;
            // find maximally activated sink
            if (val > 0.f && val > maxval) {
                maxval = val;
                maxidx = i;
            }
        }

        if (maxidx > -1 && maxidx < m_brain.size()) {
            auto [src, w, snk] = m_brain[maxidx];
            // activate sink
            snk->calculate(this, w, false);
        }
    }

    void update_Threshold() {
        // calculate neuron activation values
        for (size_t i = 0; i < m_brain.size(); ++i) {
            auto &[src, w, snk] = m_brain[i];
            auto val = src->calculate(this, w, true) * w;
            // activate above threshold
            if (std::abs(val) > NEURAL_THRESHOLD) {
                snk->calculate(this, val, false);
            }
        }
    }

    void update_Every() {
        // calculate neuron activation values
        for (size_t i = 0; i < m_brain.size(); ++i) {
            auto &[src, w, snk] = m_brain[i];
            auto val = src->calculate(this, w, true) * w;
            snk->calculate(this, val, false);
        }
    }
    
    void update() {
        update_Every();
    }
    
private:
    void setupBrain_no_memory() {
        m_brain.clear();
        m_memory.clear();

        for (size_t i = 0; i < Sources.size(); ++i) {
            auto src = Sources[i];
            for (size_t j = 0; j < Sinks.size(); ++j) {
                auto snk = Sinks[j];
                BrainConnection c(src, 0.f, snk);
                m_brain.push_back(c);
            }
        }
    }

    void setupBrain_layered_memory() {
        m_brain.clear();
        m_memory.clear();

        for (size_t i = 0; i < NUM_MEMORY; ++i) {
            m_memory.push_back(std::make_shared<MemoryNeuron>());
        }

        // connect every source to every memory neuron
        for (size_t i = 0; i < Sources.size(); ++i) {
            auto src = Sources[i];
            for (size_t k = 0; k < m_memory.size(); ++k) {
                auto m = m_memory[k];
                BrainConnection c(src, 0.f, m);
                m_brain.push_back(c);
            }
        }

        // there are no direct Source - Sink connections

        // connect every memory neuron to every sink
        for (size_t k = 0; k < m_memory.size(); ++k) {
            auto m = m_memory[k];
            for (size_t i = 0; i < Sinks.size(); ++i) {
                auto snk = Sinks[i];
                BrainConnection c(m, 0.f, snk);
                m_brain.push_back(c);
            }
        }
    }

    void setupBrain_fully_connected_memory() {
        m_brain.clear();
        m_memory.clear();

        for (size_t i = 0; i < NUM_MEMORY; ++i) {
            m_memory.push_back(std::make_shared<MemoryNeuron>());
        }

        // the order of connection is important;
        // we want to perform all memory writes
        // before any memory reads

        for (size_t i = 0; i < Sources.size(); ++i) {
            auto src = Sources[i];
            // connect all sources and sinks
            for (size_t j = 0; j < Sinks.size(); ++j) {
                auto snk = Sinks[j];
                BrainConnection c(src, 0.f, snk);
                m_brain.push_back(c);

            }
            // connect every source to every memory neuron
            for (size_t k = 0; k < m_memory.size(); ++k) {
                auto m = m_memory[k];
                BrainConnection c(src, 0.f, m);
                m_brain.push_back(c);
            }
        }

        // connect all memory neurons together;
        // this is both read and write on memory;
        // is this consistent?
        for (size_t i = 0; i < NUM_MEMORY; ++i) {
            for (size_t j = 0; j < NUM_MEMORY; ++j) {
                auto m1 = m_memory[i];
                auto m2 = m_memory[j];
                BrainConnection c1(m1, 0.f, m2);
                m_brain.push_back(c1);
            }
        }

        // connect all memory neurons to all sinks
        for (size_t i = 0; i < NUM_MEMORY; ++i) {
            auto m = m_memory[i];
            for (size_t j = 0; j < Sinks.size(); ++j) {
                auto snk = Sinks[j];
                BrainConnection c(m, 0.f, snk);
                m_brain.push_back(c);
            }
        }
    }

    void setupBrain() {
        setupBrain_fully_connected_memory();
    }

    Brain m_brain;
    std::vector<MemoryNeuron::SP> m_memory;
};

bool LiveStrategy_LeftHalf(Agent::SP a) {
    return a->position().x < SCREEN_WIDTH / 2.f;
}

bool LiveStrategy_RightHalf(Agent::SP a) {
    return a->position().x > SCREEN_WIDTH / 2.f;
}

bool LiveStrategy_CentreThirdBox(Agent::SP a) {
    auto &p = a->position();
    bool validX = (p.x > (SCREEN_WIDTH * 1.f/3.)) && (p.x < (SCREEN_WIDTH * 2.f/3.));
    bool validY = (p.y > (SCREEN_HEIGHT * 1.f/3.)) && (p.y < (SCREEN_HEIGHT * 2.f/3.));
    return validX && validY;
}

bool LiveStrategy_CentreFifthBox(Agent::SP a) {
    auto &p = a->position();
    bool validX = (p.x > (SCREEN_WIDTH * 2.f/5.)) && (p.x < (SCREEN_WIDTH * 3.f/5.));
    bool validY = (p.y > (SCREEN_HEIGHT * 2.f/5.)) && (p.y < (SCREEN_HEIGHT * 3.f/5.));
    return validX && validY;
}

bool LiveStrategy_CentreTenthBox(Agent::SP a) {
    auto &p = a->position();
    bool validX = (p.x > (SCREEN_WIDTH * 4.5f/10.)) && (p.x < (SCREEN_WIDTH * 5.5f/10.));
    bool validY = (p.y > (SCREEN_HEIGHT * 4.5f/10.)) && (p.y < (SCREEN_HEIGHT * 5.5/10.));
    return validX && validY;
}

bool LiveStrategy_OffCentreTenthBox(Agent::SP a) {
    auto &p = a->position();
    bool validX = (p.x > (SCREEN_WIDTH * 3.5f/10.)) && (p.x < (SCREEN_WIDTH * 4.5f/10.));
    bool validY = (p.y > (SCREEN_HEIGHT * 6.5f/10.)) && (p.y < (SCREEN_HEIGHT * 7.5/10.));
    return validX && validY;
}

bool LiveStrategy_CentreTwentiethBox(Agent::SP a) {
    auto &p = a->position();
    bool validX = (p.x > (SCREEN_WIDTH * 9.5f/20.)) && (p.x < (SCREEN_WIDTH * 10.5f/20.));
    bool validY = (p.y > (SCREEN_HEIGHT * 9.5f/20.)) && (p.y < (SCREEN_HEIGHT * 10.5/20.));
    return validX && validY;
}

bool LiveStrategy_OffCentreTwentiethBox(Agent::SP a) {
    auto &p = a->position();
    bool validX = (p.x > (SCREEN_WIDTH * 3.5f/20.)) && (p.x < (SCREEN_WIDTH * 4.5f/20.));
    bool validY = (p.y > (SCREEN_HEIGHT * 16.5f/20.)) && (p.y < (SCREEN_HEIGHT * 17.5/20.));
    return validX && validY;
}

bool LiveStrategy_LeftRightTenth(Agent::SP a) {
    auto &p = a->position();
    bool valid = (p.x < (SCREEN_WIDTH * 0.1f/10.)) || (p.x > (SCREEN_WIDTH * 0.9f/10.));
    return valid;
}

bool LiveStrategy_TopBottomTenth(Agent::SP a) {
    auto &p = a->position();
    bool valid = (p.y < (SCREEN_HEIGHT * 0.1f/10.)) || (p.y > (SCREEN_HEIGHT * 0.9f/10.));
    return valid;
}

bool LiveStrategy_TLTenth(Agent::SP a) {
    auto &p = a->position();
    bool validX = p.x < (SCREEN_WIDTH * 0.1f);
    bool validY = p.y < (SCREEN_HEIGHT * 0.1f);
    return validX && validY;
}

bool LiveStrategy_LowVelocity(Agent::SP a) {
    auto vx = a->velocity_x();
    auto vy = a->velocity_y();
    return std::sqrt(vx*vx + vy*vy) < 2.5f;
}

bool LiveStrategy_TLCircle(Agent::SP a) {
    auto &p = a->position();
    return std::sqrt(p.x*p.x + p.y*p.y) < (SCREEN_WIDTH / 15.f);
}

bool LiveStrategy_TRCircle(Agent::SP a) {
    auto &p = a->position();
    auto dx = p.x - SCREEN_WIDTH;
    return std::sqrt(dx*dx + p.y*p.y) < (SCREEN_WIDTH / 15.f);
}

bool LiveStrategy_TopCorners(Agent::SP a) {
    return LiveStrategy_TLCircle(a) || LiveStrategy_TRCircle(a);
}

bool LiveStrategy_BLCircle(Agent::SP a) {
    auto &p = a->position();
    auto dy = p.y - SCREEN_HEIGHT;
    return std::sqrt(p.x*p.x + dy*dy) < (SCREEN_WIDTH / 15.f);
}

bool LiveStrategy_BRCircle(Agent::SP a) {
    auto &p = a->position();
    auto dx = p.x - SCREEN_WIDTH;
    auto dy = p.y - SCREEN_HEIGHT;
    return std::sqrt(dx*dx + dy*dy) < (SCREEN_WIDTH / 15.f);
}

bool LiveStrategy_BottomCorners(Agent::SP a) {
    return LiveStrategy_BLCircle(a) || LiveStrategy_BRCircle(a);
}

bool LiveStrategy_Corners(Agent::SP a) {
    return LiveStrategy_TopCorners(a) || LiveStrategy_BottomCorners(a);
}

bool LiveStrategy_HasVelocity(Agent::SP a) {
    auto vx = a->velocity_x();
    auto vy = a->velocity_y();
    return std::sqrt(vx*vx + vy*vy) > 0.01f;
}

bool LiveStrategy_HorizTenths(Agent::SP a) {
    auto &p = a->position();
    return (static_cast<int>(std::round(p.x / 10.f)) % 2) == 0;
}

bool LiveStrategy_VertTenths(Agent::SP a) {
    auto &p = a->position();
    return (static_cast<int>(std::round(p.y / 10.f)) % 2) == 0;
}

bool LiveStrategy(Agent::SP a) {
    return LiveStrategy_OffCentreTenthBox(a);
}

std::vector<Position> InitialPositions;
std::vector<Numeric> InitialVelsX;
std::vector<Numeric> InitialVelsY;

int InitPopulation() {
    population.agents.clear();

    for (size_t i = 0; i < NUMBOIDS; ++i) {
        auto a = std::make_shared<NeuralAgent>();
        population.agents.push_back(a);
     
        a->position(RandomPosition(SCREEN_WIDTH, SCREEN_HEIGHT));
        a->colour(RandomColour());
        a->velocity_x(bipolarrandf() * MAX_VELOCITY);
        a->velocity_y(bipolarrandf() * MAX_VELOCITY);

        InitialPositions.push_back(a->position());
        InitialVelsX.push_back(a->velocity_x());
        InitialVelsY.push_back(a->velocity_y());

        // randomize brain weights
        auto &b = a->brain();
        for (size_t j = 0; j < b.size(); ++j) {
            std::get<1>(b[j]) = bipolarrandf();
        }
    }

    return 0;
}

int NextGeneration(size_t generation) {
    std::vector<Agent::SP> survivors;
    // remove dead 
    for (auto e : population.agents) {
        if (LiveStrategy(e)) {
            survivors.push_back(e);
        }
    }
    
    std::cout << "generation " << generation << " survivors = " << survivors.size() << std::endl;

    NUM_SURVIVORS = survivors.size();
    if (NUM_SURVIVORS == 0) {
        return 1;
    }

    // reproduce;
    // create another full population based on clones of the survivors' brains
    std::vector<Agent::SP> nextpop;
    for (size_t i = 0; i < NUMBOIDS; ++i) {
        auto cloneFrom = std::static_pointer_cast<NeuralAgent>(survivors[i % survivors.size()]);
        auto a = std::make_shared<NeuralAgent>(cloneFrom);
        auto e = std::static_pointer_cast<Agent>(a);
        nextpop.push_back(e);
        nextpop[i]->position(InitialPositions[i]);
        nextpop[i]->velocity_x(InitialVelsX[i]);
        nextpop[i]->velocity_y(InitialVelsY[i]);
    }

    // mutate
    for (auto e : nextpop) {
        auto a = std::static_pointer_cast<NeuralAgent>(e);
        auto &b = a->brain();
        for (size_t j = 0; j < b.size(); ++j) {
            // // bounded weights
            // std::get<1>(b[j]) = std::max(
            //     -1.0,
            //     std::min(
            //         1.0,
            //         std::get<1>(b[j]) + (bipolarrandf() * MUTATION)
            //     )
            // );

            // unbounded weights
            std::get<1>(b[j]) += (bipolarrandf() * MUTATION);
        }
        // reset colours every 100 gens
        if (generation % 100 == 0)
        {
            a->colour(RandomColour());
        }
        // otherwise mutate
        else
        {
            // BUG: this has a tendency to fade to black over time?
            auto &c = a->colour();
            c.r = std::max(0.0, std::min(255., c.r + (256 * bipolarrandf() * MUTATION)));
            c.g = std::max(0.0, std::min(255., c.g + (256 * bipolarrandf() * MUTATION)));
            c.b = std::max(0.0, std::min(255., c.b + (256 * bipolarrandf() * MUTATION)));
        }
    }

    population.agents.swap(nextpop);
    return 0;
}

int cleanup(int returnCode)
{
    if (render != nullptr)
    {
        SDL_DestroyRenderer(render);
    }
    if (window != nullptr)
    {
        SDL_DestroyWindow(window);
    }
    SDL_Quit();
    return returnCode;
}

std::string FindFont()
{
    std::string out;

    auto *pat = FcNameParse((FcChar8 *)"Hack");
    if (!pat)
    {
        std::cerr << "Could not create font pattern" << std::endl;
        return out;
    }
    auto *os = FcObjectSetCreate();
    FcObjectSetAdd(os, "file");

    auto *fs = FcFontList(0, pat, os);

    FcObjectSetDestroy(os);
    FcPatternDestroy(pat);

    for (size_t i = 0; i < fs->nfont; ++i)
    {
        auto *font = fs->fonts[i];
        FcChar8 *ff = FcPatternFormat(font, (FcChar8 *)"%{file}");
        out = std::string((char *)ff);
        break;
    }

    FcFontSetDestroy(fs);
    FcFini();

    return out;
}

int InitSDL()
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        std::cerr << "could not init sdl2: " << SDL_GetError() << std::endl;
        return 1;
    }

    window = SDL_CreateWindow(
        "boids",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        SCREEN_WIDTH, SCREEN_HEIGHT,
        SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN_DESKTOP);
    if (window == nullptr)
    {
        std::cerr << "could not create window: " << SDL_GetError() << std::endl;
        return 1;
    }

    render = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED /* | SDL_RENDERER_PRESENTVSYNC */);
    if (render == nullptr)
    {
        std::cerr << "could not create renderer: " << SDL_GetError() << std::endl;
        return 1;
    }

    const auto ff = FindFont();
    if (ff.size() == 0)
    {
        std::cerr << "could not find font: " << std::endl;
        return 1;
    }

    if (TTF_Init() != 0)
    {
        std::cerr << "could not initialise ttf: " << SDL_GetError() << std::endl;
        return 1;
    }

    sans = TTF_OpenFont(ff.c_str(), 25);
    if (sans == nullptr)
    {
        std::cerr << "could not open font: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_SetRenderDrawColor(render, 0, 0, 0, 255);
    SDL_RenderClear(render);
    SDL_RenderPresent(render);
    SDL_SetRenderDrawBlendMode(render, SDL_BLENDMODE_BLEND);

    return 0;
}

int ProcessEvents()
{
    while (SDL_PollEvent(&event))
    {
        if (event.type == SDL_QUIT)
        {
            return 1;
        }
        if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)
        {
            return 1;
        }
    }
    return 0;
}

#if USE_KDTREE
int UpdateKdTree()
{
    kdtree.buildIndex();
    return 0;
}
#endif

int UpdateEntt(double time)
{
    #pragma omp parallel for
    for (auto &entity : population.agents) {
        auto a = std::static_pointer_cast<NeuralAgent>(entity);
        a->update();
    }

    return 0;
}

int Render(size_t generation, size_t iter, int frame, double time)
{
    // partial render 9/10 generations
    if (generation % 10 != 0)
    {
        if (frame % 75 != 0)
        {
            return 0;
        }
    }

    // reset background
    {
        SDL_SetRenderDrawColor(render, 0, 0, 0, 25);
        SDL_RenderFillRect(render, NULL);
    }

    // entt points
    {
        int szx;
        int szy;
        SDL_GetWindowSize(window, &szx, &szy);
        const auto offsx = (szx - SCREEN_WIDTH) / 2.0;
        const auto offsy = (szy - SCREEN_HEIGHT) / 2.0;
        for (const auto entity : population.agents)
        {
            const auto &col = entity->colour();
            auto alpha = LiveStrategy(entity) ? 255 : 48;
            SDL_SetRenderDrawColor(render, col.r, col.g, col.b, alpha);
            const auto &pos = entity->position();
            SDL_FRect r{offsx + pos.x - 2, offsy + pos.y - 2, 4, 4};
            SDL_RenderDrawRectF(render, &r);
        };
    }

    // stats
    {
        std::stringstream stats;
        stats.precision(3);
        stats.fill(' ');
        stats 
            << "   g= " << std::setw(5) << generation
            << "   i= " << std::setw(5) << iter
            << "   f= " << std::setw(5) << frame
            << "   t= " << std::setw(5) << time
            << "   p= " << std::setw(5) << population.agents.size()
            << "   s= " << std::setw(5) << NUM_SURVIVORS
            << "   fps= " << std::setw(3) << (frame / time);
        SDL_Color txtc{255, 255, 255};
        auto statsstr = stats.str();
        // std::cout << statsstr << std::endl;
        SDL_Surface *txts = TTF_RenderText_Solid(sans, statsstr.c_str(), txtc);
        SDL_Texture *txtt = SDL_CreateTextureFromSurface(render, txts);
        SDL_Rect txtbg{0, 0, SCREEN_WIDTH, txts->h + 10};
        SDL_SetRenderDrawColor(render, 0, 0, 0, 255);
        SDL_RenderFillRect(render, &txtbg);
        SDL_Rect txtp{25, 5, txts->w, txts->h};
        SDL_RenderCopy(render, txtt, NULL, &txtp);
        SDL_DestroyTexture(txtt);
        SDL_FreeSurface(txts);
    }

    // update
    SDL_RenderPresent(render);

    return 0;
}

typedef std::chrono::steady_clock::time_point tp;
const auto now = std::chrono::steady_clock::now;
const auto dt = [](tp begin, tp end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
};

int main()
{
    randengine.seed(std::chrono::system_clock::now().time_since_epoch().count());

    if (InitSDL() != 0)
    {
        return cleanup(1);
    }

    if (InitSources() != 0)
    {
        return cleanup(1);
    }
    std::cout << "Sources size = " << Sources.size() << std::endl;

    if (InitSinks() != 0)
    {
        return cleanup(1);
    }
    std::cout << "Sinks size = " << Sinks.size() << std::endl;

    if (InitPopulation() != 0)
    {
        return cleanup(1);
    }

    tp t_start = now();
    tp t_iter = t_start;

    long f = 0;
    double t = 0;
    for (size_t g = 0; g < MAX_GENS; g++) {
        for (size_t i = 0; i < GEN_ITERS; ++i, ++f, t_iter = now(), t = dt(t_start, t_iter) / 1000.0)
        {
#if USE_KDTREE
            UpdateKdTree();
#endif
            if (ProcessEvents() != 0)
            {
                return cleanup(1);
            }

            if (UpdateEntt(t) != 0)
            {
                std::cerr << "error updating entt" << std::endl;
                return cleanup(1);
            }

            if (Render(g, i, f, t) != 0)
            {
                std::cerr << "error rendering: " << SDL_GetError() << std::endl;
                return cleanup(1);
            }
        }

        if (NextGeneration(g)) {
            std::cout << "Everyone's dead dave. end." << std::endl;
            break;
        }
    }

    return cleanup(0);
}
