#pragma once

#include <inttypes.h>

#define USE_KDTREE 0

using Numeric = double;

static const struct Config
{
    const size_t SCREEN_WIDTH = 750;
    const size_t SCREEN_HEIGHT = 750;
    const Numeric ZOOM = 0.85;

    const size_t NUMBOIDS = 5000;

    const Numeric MUTATION = 0.0012;
    const Numeric NEURAL_THRESHOLD = 0.12; // only for update_Threshold strategy
    const size_t NUM_MEMORY_PER_LAYER = 5;
    const size_t NUM_MEMORY_LAYERS = 3;

    const bool BOUNDED_WEIGHTS = true;
    const Numeric MAX_WEIGHT = 2.0;

    const Numeric MIN_SIZE = 5.0;
    const Numeric MAX_SIZE = 20.0;
    const Numeric MAX_VELOCITY = 18;

    const size_t MAX_GENS = 12000;
    const size_t GEN_ITERS = 350;
    const size_t REALTIME_EVERY_NGENS = 25;
} config;
