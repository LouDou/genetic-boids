#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef FEATURE_CLI_OPTIONS
#include <argparse/argparse.hpp>
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif // __EMSCRIPTEN__

#include "config.h"

Config config;

const Config &getConfig()
{
    return config;
}

#include "agent.h"
#include "conditions.h"
#include "neuralagent.h"
#include "random.h"
#include "sources.h"
#include "sinks.h"
#include "ui.h"
#include "video.h"

NeuronRegistry sourcesRegistry{
    {"age", []()
     { return std::make_shared<Source_Age>(); }},
    {"direction", []()
     { return std::make_shared<Source_Direction>(); }},
    {"west", []()
     { return std::make_shared<Source_West>(); }},
    {"east", []()
     { return std::make_shared<Source_East>(); }},
    {"north", []()
     { return std::make_shared<Source_North>(); }},
    {"south", []()
     { return std::make_shared<Source_South>(); }},
    {"angular-velocity", []()
     { return std::make_shared<Source_Angular_Velocity>(); }},
    {"velocity", []()
     { return std::make_shared<Source_Velocity>(); }},
    {"goal-reached", []()
     { return std::make_shared<Source_Goal_Reached>(); }},
    {"out-of-bounds", []()
     { return std::make_shared<Source_Out_of_Bounds>(); }},
    {"red", []()
     { return std::make_shared<Source_Red>(); }},
    {"green", []()
     { return std::make_shared<Source_Green>(); }},
    {"blue", []()
     { return std::make_shared<Source_Blue>(); }},
    {"size", []()
     { return std::make_shared<Source_Size>(); }},
};

const NeuronRegistry &getSources()
{
    return sourcesRegistry;
}

NeuronRegistry sinksRegistry{
    {"angular-velocity", []()
     { return std::make_shared<Sink_Angular_Velocity>(); }},
    {"direction", []()
     { return std::make_shared<Sink_Direction>(); }},
    {"velocity", []()
     { return std::make_shared<Sink_Velocity>(); }},
    {"move", []()
     { return std::make_shared<Sink_Move>(); }},
    {"red", []()
     { return std::make_shared<Sink_Red>(); }},
    {"green", []()
     { return std::make_shared<Sink_Green>(); }},
    {"blue", []()
     { return std::make_shared<Sink_Blue>(); }},
    {"size", []()
     { return std::make_shared<Sink_Size>(); }},
};

const NeuronRegistry &getSinks()
{
    return sinksRegistry;
}

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

// Agent Population

struct Population
{
    std::vector<Agent::SP> agents;
} population;

void InitialCondition(Agent::SP a)
{
    a->size(config.MIN_SIZE + (randf() * (config.MAX_SIZE - config.MIN_SIZE)));
    a->position(RandomPosition(config.SCREEN_WIDTH, config.SCREEN_HEIGHT));
    a->colour(RandomColour());
    a->direction(randf() * TWOPI);
    a->velocity(bipolarrandf() * config.MAX_VELOCITY);
    a->angular_vel(bipolarrandf() * config.MAX_ANGULAR_VELOCITY);
}

int InitPopulation()
{
    population.agents.clear();

    for (size_t i = 0; i < config.NUMBOIDS; ++i)
    {
        auto a = std::make_shared<NeuralAgent>();
        population.agents.push_back(a);
        InitialCondition(a);

        a->updateType(config.NEURAL_UPDATE_TYPE);
        a->brainType(config.NEURAL_BRAIN_TYPE);

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
        auto &d = a->weight_delta();
        for (size_t j = 0; j < b.size(); ++j)
        {
            if (config.BOUNDED_WEIGHTS)
            {
                std::get<1>(b[j]) = std::max(
                    -config.MAX_WEIGHT,
                    std::min(
                        config.MAX_WEIGHT,
                        std::get<1>(b[j]) + (d[j] * randf() * config.MUTATION)));
            }
            else
            {
                std::get<1>(b[j]) += (d[j] * randf() * config.MUTATION);
            }
            if (randf() < config.MUTATION)
            {
                d[j] *= -1; // swap mutation direction
            }
        }
    }

    population.agents.swap(nextpop);
    return 0;
}

// UI

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
#ifdef FEATURE_RENDER_VIDEO
    CleanupAV();
#endif // FEATURE_RENDER_VIDEO
    return returnCode;
}

#ifdef FEATURE_CLI_OPTIONS
int ParseArgs(int argc, char *argv[])
{
    argparse::ArgumentParser program("boids", "1.0", argparse::default_arguments::help);

    const auto AsLong = [](const std::string &v)
    { return std::stol(v); };

    const auto AsInt = [](const std::string &v)
    { return std::stoi(v); };

    const auto AsFloat = [](const std::string &v)
    { return std::stof(v); };

    program.add_argument("-s", "--seed")
        .default_value(std::chrono::system_clock::now().time_since_epoch().count())
        .action(AsLong)
        .help("Random seed");

    program.add_argument("-n", "--num-boids")
        .default_value(1000)
        .action(AsInt)
        .help("Number of boid Agents to simulate");
    program.add_argument("-m", "--mutation")
        .default_value(0.001f)
        .action(AsFloat)
        .help("Parameter mutation factor");
    program.add_argument("-t", "--neural-threshold")
        .default_value(0.12f)
        .action(AsFloat)
        .help("For Threshold update strategy; neuron activation threshold weight");
    program.add_argument("-k", "--memory-layer-size")
        .default_value(4L)
        .action(AsLong)
        .help("Layered memory type: Number of neurons per layer");
    program.add_argument("-l", "--memory-layers")
        .default_value(2L)
        .action(AsLong)
        .help("Layered memory type: Number of neuron layers");

    program.add_argument("-b", "--neuron-bounded-weights")
        .default_value(false)
        .implicit_value(true)
        .help("Neurons: Clamp neuron output weight values");
    program.add_argument("-x", "--neuron-max-weight")
        .default_value(2.0f)
        .action(AsFloat)
        .help("Neurons: With bounded weights: Maximum neuron output weight magnitude");
    program.add_argument("--neuron-sources")
        .nargs(1, 13)
        .help("Neurons: Neural sources. Choose from: age, west, east, north, south, direction, velocity, goal-reached, out-of-bounds, red, green, blue, size");
    program.add_argument("--neuron-sinks")
        .nargs(1, 7)
        .help("Neurons: Neural sinks. Choose from: move, direction, velocity, red, green, blue, size");
    program.add_argument("--neuron-update-type")
        .default_value(std::string("every"))
        .action(
            [](const std::string &value)
            {
                static const std::vector<std::string> choices = {"max", "threshold", "every"};
                if (std::find(choices.begin(), choices.end(), value) != choices.end())
                {
                    return value;
                }
                return std::string{"every"};
            })
        .help("Neurons: Update type. Choose from: max, threshold, every");
    program.add_argument("--neuron-connection-type")
        .default_value(std::string("layered"))
        .action(
            [](const std::string &value)
            {
                static const std::vector<std::string> choices = {"no-memory", "layered", "fully-connected"};
                if (std::find(choices.begin(), choices.end(), value) != choices.end())
                {
                    return value;
                }
                return std::string{"layered"};
            })
        .help("Neurons: Connection type. Choose from: no-memory, layered, fully-connected");

    program.add_argument("-p", "--agent-min-size")
        .default_value(5.0f)
        .action(AsFloat)
        .help("Agent properties: Minimum size");
    program.add_argument("-q", "--agent-max-size")
        .default_value(20.0f)
        .action(AsFloat)
        .help("Agent properties: Maximum size");
    program.add_argument("-r", "--agent-max-velocity")
        .default_value(24.0f)
        .action(AsFloat)
        .help("Agent properties: Maximum velocity magnitude");
    program.add_argument("-o", "--agent-max-angular-velocity")
        .default_value(static_cast<float>(TWOPI))
        .action(AsFloat)
        .help("Agent properties: Maximum angular velocity magnitude");

    program.add_argument("-w", "--simulation-bound-width")
        .default_value(1280)
        .action(AsInt)
        .help("Simulation: Width of \"in bounds\" area");
    program.add_argument("-h", "--simulation-bound-height")
        .default_value(720)
        .action(AsInt)
        .help("Simulation: Height of \"in bounds\" area");
    program.add_argument("-g", "--simulation-generations")
        .default_value(10000L)
        .action(AsLong)
        .help("Simulation: Maximum number of generations");
    program.add_argument("-i", "--simulation-iterations")
        .default_value(400L)
        .action(AsLong)
        .help("Simulation: Iterations per generation");

    program.add_argument("-z", "--render-zoom-factor")
        .default_value(1.0f)
        .action(AsFloat)
        .help("Rendering: Zoom factor");
    program.add_argument("-u", "--render-interval")
        .default_value(50)
        .action(AsInt)
        .help("Rendering: Render all iterations generation interval");

#ifdef FEATURE_RENDER_VIDEO
    program.add_argument("-v", "--render-save-video")
        .default_value(false)
        .implicit_value(true)
        .help("Rendering: Save video of simulation");
    program.add_argument("-d", "--render-video-scale")
        .default_value(1.0f)
        .action(AsFloat)
        .help("Rendering: Output video scale factor");
#endif // FEATURE_RENDER_VIDEO

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        exit(0);
    }

    config.SEED = program.get<long>("-s");
    config.NUMBOIDS = program.get<int>("-n");
    config.MUTATION = program.get<float>("-m");
    config.NUM_MEMORY_PER_LAYER = program.get<long>("-k");
    config.NUM_MEMORY_LAYERS = program.get<long>("-l");
    config.NEURAL_THRESHOLD = program.get<float>("-t");
    config.BOUNDED_WEIGHTS = program.get<bool>("-b");
    config.MAX_WEIGHT = program.get<float>("-x");
    config.MIN_SIZE = program.get<float>("-p");
    config.MAX_SIZE = program.get<float>("-q");
    config.MAX_VELOCITY = program.get<float>("-r");
    config.MAX_ANGULAR_VELOCITY = program.get<float>("-o");
    config.SCREEN_WIDTH = program.get<int>("-w");
    config.SCREEN_HEIGHT = program.get<int>("-h");
    config.MAX_GENS = program.get<long>("-g");
    config.GEN_ITERS = program.get<long>("-i");
    config.ZOOM = program.get<float>("-z");
    config.REALTIME_EVERY_NGENS = program.get<int>("-u");
#ifdef FEATURE_RENDER_VIDEO
    config.SAVE_FRAMES = program.get<bool>("-v");
    config.VIDEO_SCALE = program.get<float>("-d");
#endif // FEATURE_RENDER_VIDEO

    std::vector<std::string> sources = program.get<std::vector<std::string>>("--neuron-sources");
    std::vector<std::string> validSources;
    std::copy_if(
        sources.begin(), sources.end(),
        std::back_inserter(validSources),
        [](auto &v)
        { return sourcesRegistry.find(v) != sourcesRegistry.end(); });
    if (validSources.size() > 0)
    {
        config.NEURON_SOURCES.swap(validSources);
    }

    const char *const delim = ", ";
    const auto sourceslist = std::accumulate(
        std::next(config.NEURON_SOURCES.begin()),
        config.NEURON_SOURCES.end(),
        config.NEURON_SOURCES[0],
        [](std::string a, std::string b)
        {
            return a + "," + b;
        });

    std::vector<std::string> sinks = program.get<std::vector<std::string>>("--neuron-sinks");
    std::vector<std::string> validSinks;
    std::copy_if(
        sinks.begin(), sinks.end(),
        std::back_inserter(validSinks),
        [](auto &v)
        { return sinksRegistry.find(v) != sinksRegistry.end(); });
    if (validSinks.size() > 0)
    {
        config.NEURON_SINKS.swap(validSinks);
    }

    const auto sinkslist = std::accumulate(
        std::next(config.NEURON_SINKS.begin()),
        config.NEURON_SINKS.end(),
        config.NEURON_SINKS[0],
        [](std::string a, std::string b)
        {
            return a + "," + b;
        });

    auto updateType = program.get<std::string>("--neuron-update-type");
    if (updateType == "max")
    {
        config.NEURAL_UPDATE_TYPE = NeuralUpdateType::MAX;
    }
    if (updateType == "threshold")
    {
        config.NEURAL_UPDATE_TYPE = NeuralUpdateType::THRESHOLD;
    }
    if (updateType == "every")
    {
        config.NEURAL_UPDATE_TYPE = NeuralUpdateType::EVERY;
    }

    auto brainType = program.get<std::string>("--neuron-connection-type");
    if (brainType == "no-memory")
    {
        config.NEURAL_BRAIN_TYPE = NeuralBrainType::NO_MEMORY;
    }
    if (brainType == "layered")
    {
        config.NEURAL_BRAIN_TYPE = NeuralBrainType::LAYERED;
    }
    if (brainType == "fully-connected")
    {
        config.NEURAL_BRAIN_TYPE = NeuralBrainType::FULLY_CONNECTED;
    }

    std::cout
        << "Config:" << std::endl
        << " SEED=" << config.SEED << std::endl
        << " NUMBOIDS=" << config.NUMBOIDS << std::endl
        << " MUTATION=" << config.MUTATION << std::endl
        << " NUM_MEMORY_PER_LAYER=" << config.NUM_MEMORY_PER_LAYER << std::endl
        << " NUM_MEMORY_LAYERS=" << config.NUM_MEMORY_LAYERS << std::endl
        << " SOURCES=" << sourceslist << std::endl
        << " SINKS=" << sinkslist << std::endl
        << " NEURAL_UPDATE_TYPE=" << (int)config.NEURAL_UPDATE_TYPE << std::endl
        << " NEURAL_BRAIN_TYPE=" << (int)config.NEURAL_BRAIN_TYPE << std::endl
        << " NEURAL_THRESHOLD=" << config.NEURAL_THRESHOLD << std::endl
        << " BOUNDED_WEIGHTS=" << config.BOUNDED_WEIGHTS << std::endl
        << " MAX_WEIGHT=" << config.MAX_WEIGHT << std::endl
        << " MIN_SIZE=" << config.MIN_SIZE << std::endl
        << " MAX_SIZE=" << config.MAX_SIZE << std::endl
        << " MAX_VELOCITY=" << config.MAX_VELOCITY << std::endl
        << " MAX_ANGULAR_VELOCITY=" << config.MAX_ANGULAR_VELOCITY << std::endl
        << " SCREEN_WIDTH=" << config.SCREEN_WIDTH << std::endl
        << " SCREEN_HEIGHT=" << config.SCREEN_HEIGHT << std::endl
        << " MAX_GENS=" << config.MAX_GENS << std::endl
        << " GEN_ITERS=" << config.GEN_ITERS << std::endl
        << " ZOOM=" << config.ZOOM << std::endl
        << " REALTIME_EVERY_NGENS=" << config.REALTIME_EVERY_NGENS << std::endl
#ifdef FEATURE_RENDER_VIDEO
        << " SAVE_FRAMES=" << config.SAVE_FRAMES << std::endl
        << " VIDEO_SCALE=" << config.VIDEO_SCALE << std::endl;
#else
        ;
#endif // FEATURE_RENDER_VIDEO

    return 0;
}
#else
int ParseArgs(int argc, char *argv[])
{
    // just fill in some config defaults
    config.SEED = std::chrono::system_clock::now().time_since_epoch().count();
    config.SCREEN_WIDTH = 800;
    config.SCREEN_HEIGHT = 800;
    config.ZOOM = 0.75;
    config.NUMBOIDS = 50;
    config.MUTATION = 0.01;
    config.NUM_MEMORY_PER_LAYER = 4;
    config.NUM_MEMORY_LAYERS = 2;
    // SOURCES is already set
    // SINKS is already set
    // NEURAL_THRESHOLD is not required
    // NEURAL_UPDATE_TYPE is already set
    // NEURAL_BRAIN_TYPE is already set
    // BOUNDED_WEIGHTS is already set
    // MAX_WEIGHT is not required
    config.MIN_SIZE = 1.5;
    config.MAX_SIZE = 15;
    config.MAX_VELOCITY = 25;
    config.MAX_ANGULAR_VELOCITY = TWOPI;
    config.MAX_GENS = 12000;
    config.GEN_ITERS = 2000;
    config.REALTIME_EVERY_NGENS = 10;
#ifdef FEATURE_RENDER_VIDEO
    config.SAVE_FRAMES = true;
    config.VIDEO_SCALE = 1.0;
#endif // FEATURE_RENDER_VIDEO

    return 0;
}
#endif // FEATURE_CLI_OPTIONS

int main(int argc, char *argv[])
{
    if (ParseArgs(argc, argv) != 0)
    {
        return cleanup(1);
    }

    random_seed(config.SEED);

    if (InitSDL() != 0)
    {
        return cleanup(1);
    }

#ifdef FEATURE_RENDER_VIDEO
    const auto &uiconfig = GetUIConfig();
    if (InitAV(uiconfig.winWidth, uiconfig.winHeight) != 0)
    {
        return cleanup(1);
    }
#endif // FEATURE_RENDER_VIDEO

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

#ifdef __EMSCRIPTEN__
            emscripten_sleep(1);
#else
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
#endif // __EMSCRIPTEN__
        }

        if (NextGeneration(g))
        {
            return cleanup(1);
        }
    }

    return cleanup(0);
}
