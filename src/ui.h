#pragma once

#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <fontconfig/fontconfig.h>

#include "agent.h"

struct UIConfig
{
    SDL_Window *window = nullptr;
    SDL_Renderer *render = nullptr;
    SDL_Texture *texture = nullptr;
    TTF_Font *sans = nullptr;
    SDL_Event event;

    size_t winWidth;
    size_t winHeight;
};

const UIConfig &GetUIConfig();

std::string FindFont();
int InitSDL();
int ProcessEvents();
void CleanupSDL();
int Render(std::vector<Agent::SP> agents, const size_t &generation, const size_t &iter, const int &frame, const Numeric &time, const size_t &sp);
