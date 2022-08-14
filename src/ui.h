#pragma once

#include <memory>
#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>

#ifdef FEATURE_RENDER_STATS
#include <SDL2/SDL_ttf.h>
#include <fontconfig/fontconfig.h>
#endif // FEATURE_RENDER_STATS

#include "agent.h"

#ifdef FEATURE_RENDER_CHARTS
class Chart;
#endif // FEATURE_RENDER_CHARTS

struct UIConfig
{
    SDL_Window *window = nullptr;
    SDL_Renderer *render = nullptr;
    SDL_Texture *texture = nullptr;

#ifdef FEATURE_RENDER_STATS
    TTF_Font *sans = nullptr;
#endif // FEATURE_RENDER_STATS

    SDL_Event event;

    size_t winWidth;
    size_t winHeight;

#ifdef FEATURE_RENDER_CHARTS
    std::shared_ptr<Chart> c_sc; // survivors
    std::shared_ptr<Chart> c_emn; // min error
    std::shared_ptr<Chart> c_ea; // avg error
    std::shared_ptr<Chart> c_emx; // max error
#endif // FEATURE_RENDER_CHARTS
};

const UIConfig &GetUIConfig();

std::string FindFont();
int InitSDL();
int ProcessEvents();
void CleanupSDL();
int Render(std::vector<Agent::SP> agents, const size_t &generation, const size_t &iter, const int &frame, const Numeric &time, const PopulationStats &stats);
