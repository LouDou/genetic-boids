#pragma once

#include <string>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
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
};

int InitAV(const size_t &width, const size_t &height);
void SaveFrame(uint8_t *pixels, const int &pitch);
void CleanupAV();
