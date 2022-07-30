#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <fontconfig/fontconfig.h>

#include <nanoflann.hpp>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include "config.h"
#include "agent.h"
#include "conditions.h"
#include "neuralagent.h"
#include "random.h"
#include "sources.h"
#include "sinks.h"

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

// Agent kdtree

struct Population
{
    std::vector<Agent::SP> agents;

#if USE_KDTREE
    inline size_t kdtree_get_point_count() const
    {
        return agents.size();
    }
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
#endif
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

void InitialCondition(Agent::SP a)
{
    a->size(config.MIN_SIZE + (randf() * (config.MAX_SIZE - config.MIN_SIZE)));
    a->position(RandomPosition(config.SCREEN_WIDTH, config.SCREEN_HEIGHT));
    a->colour(RandomColour());
    a->direction(randf() * TWOPI);
    a->velocity(bipolarrandf() * config.MAX_VELOCITY);
}

int InitPopulation()
{
    population.agents.clear();

    for (size_t i = 0; i < config.NUMBOIDS; ++i)
    {
        auto a = std::make_shared<NeuralAgent>();
        population.agents.push_back(a);
        InitialCondition(a);

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
        for (size_t j = 0; j < b.size(); ++j)
        {
            if (config.BOUNDED_WEIGHTS)
            {
                std::get<1>(b[j]) = std::max(
                    -config.MAX_WEIGHT,
                    std::min(
                        config.MAX_WEIGHT,
                        std::get<1>(b[j]) + (bipolarrandf() * config.MUTATION)));
            }
            else
            {

                std::get<1>(b[j]) += (bipolarrandf() * config.MUTATION);
            }
        }
    }

    population.agents.swap(nextpop);
    return 0;
}

// UI

SDL_Window *window = nullptr;
SDL_Renderer *render = nullptr;
SDL_Texture *texture = nullptr;
TTF_Font *sans = nullptr;
SDL_Event event;

constexpr auto SDL_PF = SDL_PIXELFORMAT_RGB24;
constexpr auto AV_SRC_PF = AV_PIX_FMT_RGB24;

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
        config.SCREEN_WIDTH, config.SCREEN_HEIGHT,
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

    int szx;
    int szy;
    SDL_GetWindowSize(window, &szx, &szy);
    texture = SDL_CreateTexture(render, SDL_PF, SDL_TEXTUREACCESS_TARGET, szx, szy);
    if (texture == nullptr)
    {
        std::cerr << "could not create render texture" << SDL_GetError() << std::endl;
        return 1;
    }
    SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_BLEND);

    return 0;
}

void CleanupSDL()
{
    if (texture != nullptr)
    {
        SDL_DestroyTexture(texture);
    }
    if (render != nullptr)
    {
        SDL_DestroyRenderer(render);
    }
    if (window != nullptr)
    {
        SDL_DestroyWindow(window);
    }
    SDL_Quit();
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

struct AVConfig
{
    size_t srcWidth;
    size_t srcHeight;

    std::string filename;
    std::string codec_name = "libx264";
    AVCodec *codec = nullptr;
    AVCodecContext *c = nullptr;
    FILE *f = nullptr;
    AVFrame *frame = nullptr;
    AVPacket *pkt = nullptr;
    struct SwsContext *swc = nullptr;

    int i = 0;
} avconfig;

int InitAV(const size_t &width, const size_t &height)
{
    if (config.SAVE_FRAMES)
    {
        avconfig.srcWidth = width;
        avconfig.srcHeight = height;

        std::stringstream ss;
        ss << "gb-"
           << std::hex << config.SEED
           << ".mp4";
        avconfig.filename = ss.str();

        avconfig.codec = avcodec_find_encoder_by_name(avconfig.codec_name.c_str());
        if (!avconfig.codec)
        {
            std::cerr << "cannot init codec " << avconfig.codec_name << std::endl;
            return 1;
        }

        avconfig.c = avcodec_alloc_context3(avconfig.codec);
        if (!avconfig.c)
        {
            std::cerr << "cannot init context " << std::endl;
            return 1;
        }

        avconfig.pkt = av_packet_alloc();
        if (!avconfig.pkt)
        {
            std::cerr << "cannot init av packet " << std::endl;
            return 1;
        }

        avconfig.c->bit_rate = 10000000;
        avconfig.c->width = width * config.VIDEO_SCALE;
        avconfig.c->height = height * config.VIDEO_SCALE;
        avconfig.c->time_base = AVRational{1, 25};
        avconfig.c->framerate = AVRational{25, 1};
        avconfig.c->gop_size = 10;
        avconfig.c->max_b_frames = 1;
        avconfig.c->pix_fmt = AV_PIX_FMT_YUV420P;

        if (avconfig.codec->id == AV_CODEC_ID_H264)
        {
            av_opt_set(avconfig.c->priv_data, "preset", "slow", 0);
        }

        int ret;

        ret = avcodec_open2(avconfig.c, avconfig.codec, NULL);
        if (ret < 0)
        {
            char errstr[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errstr, AV_ERROR_MAX_STRING_SIZE);
            std::cerr << "Could not open codec: " << errstr << std::endl;
            return 1;
        }

        avconfig.f = fopen(avconfig.filename.c_str(), "wb");
        if (!avconfig.f)
        {
            std::cerr << "could not open " << avconfig.filename << std::endl;
            return 1;
        }

        avconfig.frame = av_frame_alloc();
        if (!avconfig.frame)
        {
            std::cerr << "could not allocate video frame" << std::endl;
            return 1;
        }
        avconfig.frame->format = avconfig.c->pix_fmt;
        avconfig.frame->width = avconfig.c->width;
        avconfig.frame->height = avconfig.c->height;

        ret = av_frame_get_buffer(avconfig.frame, 0);
        if (ret < 0)
        {
            char errstr[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errstr, AV_ERROR_MAX_STRING_SIZE);
            std::cerr << "could not allocate video frame data: " << errstr << std::endl;
            return 1;
        }

        avconfig.swc = sws_getContext(
            width, height, AV_SRC_PF,
            width * config.VIDEO_SCALE, height * config.VIDEO_SCALE, avconfig.c->pix_fmt,
            0, 0, 0, 0);
        if (!avconfig.swc)
        {
            std::cerr << "could not allocate sws context" << std::endl;
            return 1;
        }
    }

    return 0;
}

void SaveFrame(bool flush = false)
{
    avconfig.frame->pts = avconfig.i++;

    auto ret = av_frame_make_writable(avconfig.frame);
    if (ret < 0)
    {
        char errstr[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errstr, AV_ERROR_MAX_STRING_SIZE);
        std::cerr << "could not make frame writable: " << errstr << std::endl;
        return;
    }

    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, avconfig.srcWidth, avconfig.srcHeight, 32, SDL_PF);
    SDL_RenderReadPixels(render, NULL, surface->format->format, surface->pixels, surface->pitch);

    uint8_t *inData[1] = {static_cast<uint8_t*>(surface->pixels)}; // ARGB24 have one plane
    int inLinesize[1] = {surface->pitch}; // ARGB stride for the single plane
    ret = sws_scale(
        avconfig.swc,
        inData, inLinesize,
        0, surface->h,
        avconfig.frame->data, avconfig.frame->linesize);
    if (ret < 0)
    {
        std::cerr << "could not convert frame" << std::endl;
        return;
    }
    SDL_FreeSurface(surface);

    ret = avcodec_send_frame(avconfig.c, flush ? NULL : avconfig.frame);
    if (ret < 0)
    {
        char errstr[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errstr, AV_ERROR_MAX_STRING_SIZE);
        std::cerr << "could not send frame to codec: " << errstr << std::endl;
        return;
    }

    while (ret >= 0)
    {
        ret = avcodec_receive_packet(avconfig.c, avconfig.pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        {
            return;
        }
        else if (ret < 0)
        {
            char errstr[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errstr, AV_ERROR_MAX_STRING_SIZE);
            std::cerr << "error during encoding: " << errstr << std::endl;
            return;
        }

        fwrite(avconfig.pkt->data, 1, avconfig.pkt->size, avconfig.f);
        av_packet_unref(avconfig.pkt);
    }
}

void CleanupAV()
{
    SaveFrame(true);

    if (avconfig.swc)
    {
        sws_freeContext(avconfig.swc);
    }

    if (avconfig.frame)
    {
        if (avconfig.codec->id == AV_CODEC_ID_MPEG1VIDEO || avconfig.codec->id == AV_CODEC_ID_MPEG2VIDEO)
        {
            uint8_t endcode[] = {0, 0, 1, 0xb7};
            fwrite(endcode, 1, sizeof(endcode), avconfig.f);
        }
    }

    if (avconfig.f)
    {
        fclose(avconfig.f);
    }

    if (avconfig.c)
    {
        avcodec_free_context(&avconfig.c);
    }

    if (avconfig.frame)
    {
        av_frame_free(&avconfig.frame);
    }

    if (avconfig.pkt)
    {
        av_packet_free(&avconfig.pkt);
    }
}

int Render(const size_t &generation, const size_t &iter, const int &frame, const Numeric &time)
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
        // SDL_RenderClear(render);
        SDL_SetRenderTarget(render, texture);
        SDL_SetRenderDrawColor(render, 0, 0, 0, 25);
        SDL_RenderFillRect(render, NULL);
    }

    size_t living = 0;
    int szx;
    int szy;
    SDL_GetWindowSize(window, &szx, &szy);
    // Render population
    {
        const auto offsx = (szx - (config.SCREEN_WIDTH * config.ZOOM)) / 2.0;
        const auto offsy = (szy - (config.SCREEN_HEIGHT * config.ZOOM)) / 2.0;
        for (const auto entity : population.agents)
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
            filledCircleRGBA(render, offsx + (pos.x * config.ZOOM), offsy + (pos.y * config.ZOOM), sz * config.ZOOM, col.r, col.g, col.b, alpha);
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
            << "   p= " << std::setw(5) << population.agents.size()
            << "   sc= " << std::setw(5) << living
            << "   st= " << std::setw(5) << NUM_SURVIVORS
            << "   fps= " << std::setw(3) << (frame / time);
        SDL_Color txtc{255, 255, 255};
        auto statsstr = stats.str();
        // std::cout << statsstr << std::endl;
        SDL_Surface *txts = TTF_RenderText_Solid(sans, statsstr.c_str(), txtc);
        SDL_Texture *txtt = SDL_CreateTextureFromSurface(render, txts);
        SDL_Rect txtbg{0, 0, config.SCREEN_WIDTH, txts->h + 10};
        SDL_SetRenderDrawColor(render, 0, 0, 0, 255);
        SDL_RenderFillRect(render, &txtbg);
        SDL_Rect txtp{25, 5, txts->w, txts->h};
        SDL_RenderCopy(render, txtt, NULL, &txtp);
        SDL_DestroyTexture(txtt);
        SDL_FreeSurface(txts);
    }

    // update
    {
        SDL_SetRenderTarget(render, NULL);
        SDL_RenderCopy(render, texture, NULL, NULL);
        if (config.SAVE_FRAMES)
        {
            SaveFrame();
        }
        SDL_RenderPresent(render);
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
    CleanupAV();
    return returnCode;
}

int main()
{
    random_seed(config.SEED);

    if (InitSDL() != 0)
    {
        return cleanup(1);
    }

    int szx;
    int szy;
    SDL_GetWindowSize(window, &szx, &szy);
    if (InitAV(szx, szy) != 0)
    {
        return cleanup(1);
    }

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
#if USE_KDTREE
            UpdateKdTree();
#endif
            if (ProcessEvents() != 0)
            {
                return cleanup(1);
            }

            if (UpdateAgents(i) != 0)
            {
                std::cerr << "error updating entt" << std::endl;
                return cleanup(1);
            }

            if (Render(g, i, f, t) != 0)
            {
                std::cerr << "error rendering: " << SDL_GetError() << std::endl;
                return cleanup(1);
            }

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
        }

        if (NextGeneration(g))
        {
            return cleanup(1);
        }
    }

    return cleanup(0);
}
