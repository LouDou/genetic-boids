#pragma once

#include <chrono>
#include <cstddef>
#include <inttypes.h>

#include <SDL2/SDL.h>

extern "C"
{
#include <libavutil/opt.h>
}

#define USE_KDTREE 0

using Numeric = double;

constexpr Numeric TWOPI = 2 * 3.14159;

// These are defined here because they have to match
constexpr auto SDL_PF = SDL_PIXELFORMAT_RGB24;
constexpr auto AV_SRC_PF = AV_PIX_FMT_RGB24;

struct Config
{
    int64_t SEED = 0;

    int SCREEN_WIDTH = 0;
    int SCREEN_HEIGHT = 0;
    Numeric ZOOM = 0.0;

    size_t NUMBOIDS = 0;

    Numeric MUTATION = 0.0;
    Numeric NEURAL_THRESHOLD = 0.0; // only for update_Threshold strategy
    size_t NUM_MEMORY_PER_LAYER = 0;
    size_t NUM_MEMORY_LAYERS = 0;

    bool BOUNDED_WEIGHTS = false;
    Numeric MAX_WEIGHT = 0.0;

    Numeric MIN_SIZE = 0.0;
    Numeric MAX_SIZE = 0.0;
    Numeric MAX_VELOCITY = 0.0;

    size_t MAX_GENS = 0;
    size_t GEN_ITERS = 0;
    size_t REALTIME_EVERY_NGENS = 0;

    bool SAVE_FRAMES = false;
    Numeric VIDEO_SCALE = 0.0;
};

const Config &getConfig();
