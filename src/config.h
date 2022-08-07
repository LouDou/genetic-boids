#pragma once

#include <chrono>
#include <cstddef>
#include <inttypes.h>
#include <string>
#include <vector>

#include <SDL2/SDL.h>

#ifdef FEATURE_RENDER_VIDEO
extern "C"
{
#include <libavutil/opt.h>
}

// These are defined here because they have to match
constexpr auto AV_SRC_PF = AV_PIX_FMT_RGB24;
constexpr auto SDL_PF = SDL_PIXELFORMAT_RGB24;
#else
constexpr auto SDL_PF = SDL_PIXELFORMAT_RGB24;
#endif // FEATURE_RENDER_VIDEO

using Numeric = double;
constexpr Numeric TWOPI = 2 * 3.14159;

enum class NeuralUpdateType
{
    MAX,
    THRESHOLD,
    EVERY,
};

enum class NeuralBrainType
{
    NO_MEMORY,
    LAYERED,
    FULLY_CONNECTED,
};

struct Config
{
    int64_t SEED = 0;

    int SCREEN_WIDTH = 0;
    int SCREEN_HEIGHT = 0;
    Numeric ZOOM = 0.0;

    size_t NUMBOIDS = 0;

    Numeric MUTATION = 0.0;
    size_t NUM_MEMORY_PER_LAYER = 0;
    size_t NUM_MEMORY_LAYERS = 0;
    std::vector<std::string> NEURON_SOURCES = {
        "age",
        "west",
        "east",
        "north",
        "south",
        "direction",
        "velocity",
        "angular-velocity",
        "goal-reached",
        // "out-of-bounds",
        "red",
        "green",
        "blue",
        "size",
    };
    std::vector<std::string> NEURON_SINKS = {
        "angular-velocity",
        "velocity",
        "direction",
        "move",
        "red",
        "green",
        "blue",
        "size",
    };
    Numeric NEURAL_THRESHOLD = 0.0; // only for Threshold update strategy
    NeuralUpdateType NEURAL_UPDATE_TYPE = NeuralUpdateType::EVERY;
    NeuralBrainType NEURAL_BRAIN_TYPE = NeuralBrainType::LAYERED;

    bool BOUNDED_WEIGHTS = false;
    Numeric MAX_WEIGHT = 0.0;

    Numeric MIN_SIZE = 0.0;
    Numeric MAX_SIZE = 0.0;
    Numeric MAX_VELOCITY = 0.0;
    Numeric MAX_ANGULAR_VELOCITY = 0.0;

    size_t MAX_GENS = 0;
    size_t GEN_ITERS = 0;
    size_t REALTIME_EVERY_NGENS = 0;

#ifdef FEATURE_RENDER_VIDEO
    bool SAVE_FRAMES = false;
    Numeric VIDEO_SCALE = 0.0;
#endif // FEATURE_RENDER_VIDEO
};

const Config &getConfig();
