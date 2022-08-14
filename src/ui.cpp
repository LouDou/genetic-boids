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

#ifdef FEATURE_RENDER_CHARTS
class Chart
{

public:
    Chart(const int w, const int h, const int r, const int g, const int b)
        : m_w(w),
          m_h(h),
          m_r(r),
          m_g(g),
          m_b(b)
    {
        m_tex = SDL_CreateTexture(uiconfig.render, SDL_PF, SDL_TEXTUREACCESS_TARGET, m_w, m_h);
    }

    ~Chart()
    {
        if (m_tex != nullptr)
        {
            SDL_DestroyTexture(m_tex);
            m_tex = nullptr;
        }
    }

    void push(const Numeric &datum)
    {
        m_data.push_back(datum);
        if (datum > m_max)
        {
            m_max = datum;
        }
        if (datum < m_min)
        {
            m_min = datum;
        }
    }

    const SDL_Rect Render()
    {
        SDL_SetRenderTarget(uiconfig.render, m_tex);

        SDL_SetRenderDrawColor(uiconfig.render, 12, 12, 12, 255);
        const SDL_Rect dst1{0, 0, m_w, m_h};
        SDL_RenderFillRect(uiconfig.render, &dst1);

        SDL_SetRenderDrawColor(uiconfig.render, m_r, m_g, m_b, 200);
        const SDL_Rect dst2{1, 1, m_w - 2, m_h - 2};
        SDL_RenderDrawRect(uiconfig.render, &dst2);

        const size_t nd = m_data.size();
        for (int x = 1; x < m_w - 1; x++)
        {
            const size_t i = std::floor(nd * x / (m_w - 2));
            const auto d = m_data[i];
            const auto v = (d - m_min) / (m_max - m_min);
            const int y = 1 + (m_h - 2) - (v * (m_h - 2));
            filledCircleRGBA(uiconfig.render, x, y, 1.5, m_r, m_g, m_b, 200);
        }

        SDL_SetRenderTarget(uiconfig.render, NULL);

        return {0, 0, m_w, m_h};
    }

    SDL_Texture *texture()
    {
        return m_tex;
    }

private:
    int m_w;
    int m_h;

    int m_r;
    int m_g;
    int m_b;

    std::vector<Numeric> m_data;
    Numeric m_min = INFINITY;
    Numeric m_max = -INFINITY;

    SDL_Texture *m_tex = nullptr;
};
#endif // FEATURE_RENDER_CHARTS

#ifdef FEATURE_RENDER_STATS
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
#endif // FEATURE_RENDER_STATS

int InitSDL()
{
    const auto &config = getConfig();

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
    {
        std::cerr << "could not init sdl2: " << SDL_GetError() << std::endl;
        return 1;
    }

#ifdef __EMSCRIPTEN__
    const auto windowFlags = SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_OPENGL;
#else
    const auto windowFlags = SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN_DESKTOP;
#endif
    uiconfig.window = SDL_CreateWindow(
        "boids",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        config.SCREEN_WIDTH, config.SCREEN_HEIGHT,
        windowFlags);
    if (uiconfig.window == nullptr)
    {
        std::cerr << "could not create window: " << SDL_GetError() << std::endl;
        return 1;
    }

    uiconfig.render = SDL_CreateRenderer(uiconfig.window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE /* | SDL_RENDERER_PRESENTVSYNC */);
    if (uiconfig.render == nullptr)
    {
        std::cerr << "could not create renderer: " << SDL_GetError() << std::endl;
        return 1;
    }

#ifdef FEATURE_RENDER_STATS
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
#endif // FEATURE_RENDER_STATS

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

#ifdef FEATURE_RENDER_CHARTS
    uiconfig.c_sc = std::make_shared<Chart>(uiconfig.winWidth / 4, 400, 12, 200, 12);
    uiconfig.c_emn = std::make_shared<Chart>(uiconfig.winWidth / 4, 400, 120, 12, 200);
    uiconfig.c_ea = std::make_shared<Chart>(uiconfig.winWidth / 4, 400, 200, 12, 120);
    uiconfig.c_emx = std::make_shared<Chart>(uiconfig.winWidth / 4, 400, 200, 12, 12);
#endif // FEATURE_RENDER_CHARTS

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
        if (uiconfig.event.type = SDL_MOUSEBUTTONDOWN && uiconfig.event.button.clicks == 1)
        {
            auto &config = getConfig();
            const auto dx = uiconfig.event.button.x - (uiconfig.winWidth / 2.0) + (config.SCREEN_WIDTH / 2.0 * config.ZOOM);
            const auto dy = uiconfig.event.button.y - (uiconfig.winHeight / 2.0) + (config.SCREEN_HEIGHT / 2.0 * config.ZOOM);
            config.TARGET_X = dx / config.ZOOM;
            config.TARGET_Y = dy / config.ZOOM;
            std::cout << "Target now "
                      << "("
                      << config.TARGET_X << ", " << config.TARGET_Y
                      << ")"
                      << std::endl;
            return 0;
        }
    }
    return 0;
}

void CleanupSDL()
{
#ifdef FEATURE_RENDER_CHARTS
    uiconfig.c_sc.reset();
    uiconfig.c_emn.reset();
    uiconfig.c_ea.reset();
    uiconfig.c_emx.reset();
#endif // FEATURE_RENDER_CHARTS

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

int Render(std::vector<Agent::SP> agents, const size_t &generation, const size_t &iter, const int &frame, const Numeric &time, const PopulationStats &stats)
{
    const auto &config = getConfig();

    // render only end frame for most generations
    if (config.REALTIME_EVERY_NGENS == 0 || (generation % config.REALTIME_EVERY_NGENS != 0))
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

            const auto error = ErrorFunction(entity);
            Uint8 alpha = std::min(255.0, std::max(5.0, 5 + (250 * (1 - error))));
            if (error < stats.errThreshold)
            {
                living++;
            }

            const auto &pos = entity->position();
            const auto &sz = entity->size();
            filledCircleRGBA(uiconfig.render, offsx + (pos.x * config.ZOOM), offsy + (pos.y * config.ZOOM), sz * config.ZOOM, col.r, col.g, col.b, alpha);
        };
    }

#ifdef FEATURE_RENDER_STATS
    // Render stats
    {
        std::stringstream statsstream;
        statsstream.precision(3);
        statsstream.fill(' ');
        statsstream
            << "   g= " << (generation + 1)
            << "   i= " << (iter + 1)
            << "   f= " << (frame + 1)
            << "   t= " << std::setw(5) << time
            << "   p= " << agents.size()
            << "   sc= " << living
            << "   st= " << std::setw(5) << stats.survivors
            << "   Emn= " << std::setw(5) << stats.minError
            << "   Eav= " << std::setw(5) << stats.avgError
            << "   Emx= " << std::setw(5) << stats.maxError
            << "   ips= " << std::setw(3) << (frame / time);
        SDL_Color txtc{255, 255, 255};
        auto statsstr = statsstream.str();
        // std::cout << statsstr << std::endl;
        SDL_Surface *txts = TTF_RenderText_Solid(uiconfig.sans, statsstr.c_str(), txtc);
        SDL_Texture *txtt = SDL_CreateTextureFromSurface(uiconfig.render, txts);
        SDL_Rect txtbg{0, 0, uiconfig.winWidth, txts->h + 10};
        SDL_SetRenderDrawColor(uiconfig.render, 0, 0, 0, 255);
        SDL_RenderFillRect(uiconfig.render, &txtbg);
        SDL_Rect txtp{25, 5, txts->w, txts->h};
        SDL_RenderCopy(uiconfig.render, txtt, NULL, &txtp);
        SDL_DestroyTexture(txtt);
        SDL_FreeSurface(txts);
    }
#endif // FEATURE_RENDER_STATS

    // Render main simulation
    {
        SDL_SetRenderTarget(uiconfig.render, NULL);
        SDL_RenderClear(uiconfig.render);
        SDL_RenderCopy(uiconfig.render, uiconfig.texture, NULL, NULL);
    }

#ifdef FEATURE_RENDER_CHARTS
    // Render Charts
    if (config.RENDER_CHARTS)
    {
        uiconfig.c_sc->push(living);
        const auto src1 = uiconfig.c_sc->Render();
        const SDL_Rect dst1 = {
            0, uiconfig.winHeight - src1.h,
            src1.w, src1.h};
        SDL_RenderCopy(uiconfig.render, uiconfig.c_sc->texture(), NULL, &dst1);

        uiconfig.c_emn->push(stats.minError);
        const auto src2 = uiconfig.c_emn->Render();
        const SDL_Rect dst2 = {
            uiconfig.winWidth / 4, uiconfig.winHeight - src2.h,
            src2.w, src2.h};
        SDL_RenderCopy(uiconfig.render, uiconfig.c_emn->texture(), NULL, &dst2);

        uiconfig.c_ea->push(stats.avgError);
        const auto src3 = uiconfig.c_ea->Render();
        const SDL_Rect dst3 = {
            2 * uiconfig.winWidth / 4, uiconfig.winHeight - src3.h,
            src3.w, src3.h};
        SDL_RenderCopy(uiconfig.render, uiconfig.c_ea->texture(), NULL, &dst3);

        uiconfig.c_emx->push(stats.maxError);
        const auto src4 = uiconfig.c_emx->Render();
        const SDL_Rect dst4 = {
            3 * uiconfig.winWidth / 4, uiconfig.winHeight - src4.h,
            src4.w, src4.h};
        SDL_RenderCopy(uiconfig.render, uiconfig.c_emx->texture(), NULL, &dst4);
    }
#endif // FEATURE_RENDER_CHARTS

    // Save ?
    {
#ifdef FEATURE_RENDER_VIDEO
        if (config.SAVE_FRAMES)
        {
            SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, uiconfig.winWidth, uiconfig.winHeight, 32, SDL_PF);
            SDL_RenderReadPixels(uiconfig.render, NULL, surface->format->format, surface->pixels, surface->pitch);
            uint8_t *pixels = static_cast<uint8_t *>(surface->pixels);
            SaveFrame(pixels, surface->pitch);
            SDL_FreeSurface(surface);
        }
#endif // FEATURE_RENDER_VIDEO
    }

    // Present
    {
        SDL_RenderPresent(uiconfig.render);
    }

    return 0;
}
