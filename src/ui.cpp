#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "conditions.h"
#include "ui.h"
#include "video.h"

UIConfig uiconfig;

const UIConfig &GetUIConfig()
{
    return uiconfig;
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

    uiconfig.window = SDL_CreateWindow(
        "boids",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        config.SCREEN_WIDTH, config.SCREEN_HEIGHT,
        SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN_DESKTOP);
    if (uiconfig.window == nullptr)
    {
        std::cerr << "could not create window: " << SDL_GetError() << std::endl;
        return 1;
    }

    uiconfig.render = SDL_CreateRenderer(uiconfig.window, -1, SDL_RENDERER_ACCELERATED /* | SDL_RENDERER_PRESENTVSYNC */);
    if (uiconfig.render == nullptr)
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

    uiconfig.sans = TTF_OpenFont(ff.c_str(), 25);
    if (uiconfig.sans == nullptr)
    {
        std::cerr << "could not open font: " << SDL_GetError() << std::endl;
        return 1;
    }

    int szx;
    int szy;
    SDL_GetWindowSize(uiconfig.window, &szx, &szy);
    uiconfig.winWidth = szx;
    uiconfig.winHeight = szy;
    uiconfig.texture = SDL_CreateTexture(uiconfig.render, SDL_PF, SDL_TEXTUREACCESS_TARGET, szx, szy);
    if (uiconfig.texture == nullptr)
    {
        std::cerr << "could not create render texture" << SDL_GetError() << std::endl;
        return 1;
    }
    SDL_SetTextureBlendMode(uiconfig.texture, SDL_BLENDMODE_BLEND);

    return 0;
}

int ProcessEvents()
{
    while (SDL_PollEvent(&uiconfig.event))
    {
        if (uiconfig.event.type == SDL_QUIT)
        {
            return 1;
        }
        if (uiconfig.event.type == SDL_KEYDOWN && uiconfig.event.key.keysym.sym == SDLK_ESCAPE)
        {
            return 1;
        }
    }
    return 0;
}

void CleanupSDL()
{
    if (uiconfig.texture != nullptr)
    {
        SDL_DestroyTexture(uiconfig.texture);
    }
    if (uiconfig.render != nullptr)
    {
        SDL_DestroyRenderer(uiconfig.render);
    }
    if (uiconfig.window != nullptr)
    {
        SDL_DestroyWindow(uiconfig.window);
    }
    SDL_Quit();
}

int Render(std::vector<Agent::SP> agents, const size_t &generation, const size_t &iter, const int &frame, const Numeric &time, const size_t &st)
{
    // render only end frame for most generations, but not the first
    if (generation != 0 && generation % config.REALTIME_EVERY_NGENS != 0)
    {
        if (iter != (config.GEN_ITERS - 1))
        {
            return 0;
        }
    }

    // reset background
    {
        // SDL_RenderClear(uiconfig.render);
        SDL_SetRenderTarget(uiconfig.render, uiconfig.texture);
        SDL_SetRenderDrawColor(uiconfig.render, 0, 0, 0, 25);
        SDL_RenderFillRect(uiconfig.render, NULL);
    }

    size_t living = 0;
    // Render population
    {
        const auto offsx = (uiconfig.winWidth - (config.SCREEN_WIDTH * config.ZOOM)) / 2.0;
        const auto offsy = (uiconfig.winHeight - (config.SCREEN_HEIGHT * config.ZOOM)) / 2.0;
        for (const auto entity : agents)
        {
            const auto &col = entity->colour();

            Uint8 alpha = 48;
            if (LiveStrategy(entity))
            {
                alpha = 255;
                living++;
            }

            const auto &pos = entity->position();
            const auto &sz = entity->size();
            filledCircleRGBA(uiconfig.render, offsx + (pos.x * config.ZOOM), offsy + (pos.y * config.ZOOM), sz * config.ZOOM, col.r, col.g, col.b, alpha);
        };
    }

    // Render stats
    {
        std::stringstream stats;
        stats.precision(3);
        stats.fill(' ');
        stats
            << "   g= " << std::setw(5) << (generation + 1)
            << "   i= " << std::setw(5) << (iter + 1)
            << "   f= " << std::setw(5) << (frame + 1)
            << "   t= " << std::setw(5) << time
            << "   p= " << std::setw(5) << agents.size()
            << "   sc= " << std::setw(5) << living
            << "   st= " << std::setw(5) << st
            << "   fps= " << std::setw(3) << (frame / time);
        SDL_Color txtc{255, 255, 255};
        auto statsstr = stats.str();
        // std::cout << statsstr << std::endl;
        SDL_Surface *txts = TTF_RenderText_Solid(uiconfig.sans, statsstr.c_str(), txtc);
        SDL_Texture *txtt = SDL_CreateTextureFromSurface(uiconfig.render, txts);
        SDL_Rect txtbg{0, 0, config.SCREEN_WIDTH, txts->h + 10};
        SDL_SetRenderDrawColor(uiconfig.render, 0, 0, 0, 255);
        SDL_RenderFillRect(uiconfig.render, &txtbg);
        SDL_Rect txtp{25, 5, txts->w, txts->h};
        SDL_RenderCopy(uiconfig.render, txtt, NULL, &txtp);
        SDL_DestroyTexture(txtt);
        SDL_FreeSurface(txts);
    }

    // update
    {
        SDL_SetRenderTarget(uiconfig.render, NULL);
        SDL_RenderCopy(uiconfig.render, uiconfig.texture, NULL, NULL);
        if (config.SAVE_FRAMES)
        {
            SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, uiconfig.winWidth, uiconfig.winHeight, 32, SDL_PF);
            SDL_RenderReadPixels(uiconfig.render, NULL, surface->format->format, surface->pixels, surface->pitch);
            uint8_t *pixels = static_cast<uint8_t *>(surface->pixels);
            SaveFrame(pixels, surface->pitch);
            SDL_FreeSurface(surface);
        }
        SDL_RenderPresent(uiconfig.render);
    }

    return 0;
}
