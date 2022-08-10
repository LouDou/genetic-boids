#ifdef FEATURE_RENDER_VIDEO

#include <iostream>
#include <iomanip>

#include "config.h"
#include "video.h"

AVConfig avconfig;

int InitAV(const size_t &width, const size_t &height)
{
    const auto &config = getConfig();

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

        avconfig.c->bit_rate = 2000000;
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

void SaveFrame(uint8_t *pixels, const int &pitch)
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

    if (pixels != nullptr)
    {
        uint8_t *inData[1] = {pixels}; // ARGB24 have one plane
        int inLinesize[1] = {pitch};   // ARGB stride for the single plane
        ret = sws_scale(
            avconfig.swc,
            inData, inLinesize,
            0, avconfig.srcHeight,
            avconfig.frame->data, avconfig.frame->linesize);
        if (ret < 0)
        {
            std::cerr << "could not convert frame" << std::endl;
            return;
        }
    }

    ret = avcodec_send_frame(avconfig.c, pixels == nullptr ? NULL : avconfig.frame);
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
    if (avconfig.frame)
    {
        SaveFrame(nullptr, 0);
    }

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

#endif // FEATURE_RENDER_VIDEO
