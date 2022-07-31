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

static const struct Config
{
    const int64_t SEED = std::chrono::system_clock::now().time_since_epoch().count();

    const int SCREEN_WIDTH = 1280;
    const int SCREEN_HEIGHT = 720;
    const Numeric ZOOM = 0.85;

    const size_t NUMBOIDS = 5000;

    const Numeric MUTATION = 0.005;
    const Numeric NEURAL_THRESHOLD = 0.12; // only for update_Threshold strategy
    const size_t NUM_MEMORY_PER_LAYER = 4;
    const size_t NUM_MEMORY_LAYERS = 2;

    const bool BOUNDED_WEIGHTS = true;
    const Numeric MAX_WEIGHT = 2.0;

    const Numeric MIN_SIZE = 5.0;
    const Numeric MAX_SIZE = 20.0;
    const Numeric MAX_VELOCITY = 24;

    const size_t MAX_GENS = 12000;
    const size_t GEN_ITERS = 400;
    const size_t REALTIME_EVERY_NGENS = 50;

    const bool SAVE_FRAMES = true;
    const Numeric VIDEO_SCALE = 0.25;
} config;
