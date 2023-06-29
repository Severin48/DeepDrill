// -----------------------------------------------------------------------------
// This file is part of DeepDrill
//
// A Mandelbrot generator based on perturbation and series approximation
//
// Copyright (C) Dirk W. Hoffmann. www.dirkwhoffmann.de
// Licensed under the GNU General Public License v3
//
// See https://www.gnu.org for license information
// -----------------------------------------------------------------------------

#include "Recorder.h"
#include "Exception.h"
#include "Options.h"
#include "Logger.h"

namespace dd {

Recorder::Recorder(const Options &opt) : opt(opt)
{

}

void
Recorder::startRecording()
{
    assert(!videoPipe.isOpen());

    // Create pipe
    if (!videoPipe.create()) {
        throw Exception("Failed to create the video encoder pipe.");
    }

    // Create temporary buffers
    videoData.resize(opt.image.width * opt.image.height);

    //
    // Assemble the command line arguments for the video encoder
    //

    // Console interactions
    string cmd = " -nostdin";

    // Verbosity
    cmd += " -loglevel " + loglevel();

    // Input stream format
    cmd += " -f:v rawvideo -pixel_format rgba";

    // Frame rate
    cmd += " -r " + std::to_string(60);

    // Frame size (width x height)
    cmd += " -s:v " + std::to_string(opt.image.width) + "x" + std::to_string(opt.image.height);

    // Input source (named pipe)
    cmd += " -i " + videoPipe.name;

    // Output stream format
    cmd += " -f mp4 -pix_fmt yuv420p";

    // Bit rate
    cmd += " -b:v " + std::to_string(opt.video.bitrate) + "k";

    // Output file
    cmd += " -y " + opt.files.outputs.front().string();

    //
    // Launch FFmpeg instance
    //

    assert(!videoFFmpeg.isRunning());

    // Launch the video encoder
    if (!videoFFmpeg.launch(cmd)) {
        throw Exception("Unable to launch the FFmpeg encoder.");
    }

    // Open the video pipe
    if (!videoPipe.open()) {
        throw Exception("Unable to open the video pipe.");
    }

    if (opt.flags.verbose) {

        log::cout << log::vspace;
        log::cout << log::ralign("Resolution: ");
        log::cout << opt.image.width << " x " << opt.image.height << log::endl;
        log::cout << log::ralign("Frame rate: ");
        log::cout << opt.video.frameRate << " Hz" << log::endl;
        log::cout << log::ralign("Keyframes: ");
        log::cout << opt.video.keyframes << log::endl;
        log::cout << log::ralign("Inbetweens: ");
        log::cout << opt.video.inbetweens << log::endl;
        log::cout << log::ralign("Bitrate: ");
        log::cout << opt.video.bitrate << log::endl;
        log::cout << log::vspace;
    }
}

void
Recorder::stopRecording()
{
    if (videoPipe.isOpen()) {

        // Close pipe
        videoPipe.close();

        // Wait for the decoder to terminate
        videoFFmpeg.join();
    }
}

void
Recorder::record(const sf::Image &img)
{
    assert(img.getSize().x == opt.image.width);
    assert(img.getSize().y == opt.image.height);
    assert(videoFFmpeg.isRunning());
    assert(videoPipe.isOpen());

    isize length = sizeof(u32) * opt.image.width * opt.image.height;

    u8 *src = (u8 *)img.getPixelsPtr();
    u8 *dst = (u8 *)videoData.data();

    std::memcpy(dst, src, length);

    // Feed the video pipe
    isize written = videoPipe.write((u8 *)videoData.data(), length);

    if (written != length) {
        throw Exception("FFmpeg: Failed to write frame\n");
    }
}

}
