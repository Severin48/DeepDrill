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

#include "Zoomer.h"
#include "Coord.h"
#include "Exception.h"
#include "IO.h"
#include "Options.h"
#include "Parser.h"
#include "ProgressIndicator.h"

#include <SFML/Graphics.hpp>
#include <thread>
#include <future>

namespace fs = std::filesystem;

namespace dd {

Zoomer::Zoomer(Options &o) : opt(o)
{
    init();
}

void
Zoomer::init()
{
    recordMode = !opt.files.outputs.empty();

    // Create the render window
    auto mode = sf::VideoMode(unsigned(opt.image.width), unsigned(opt.image.height));
    window.create(mode, "");

    // Hide the window in batch mode
    if (opt.flags.batch) window.setVisible(false);

    // Preview in real-time if no video is recorded
    window.setFramerateLimit(recordMode ? 0 : unsigned(opt.video.frameRate));

    // Initialize the colorizer
    colorizer.init(opt.image.illuminator, opt.video.scaler);
}

/*
void
Zoomer::launch()
{
    sf::Event event;

    // Start FFmpeg
    if (recordMode) recorder.startRecording();

    // Load the textures of the first two keyframes
    log::cout << log::vspace;
    log::cout << "Preloading map file 0" << log::endl << log::endl;
    (void)loadMapFile(0);

    log::cout << log::vspace;
    log::cout << "Preloading map file 1" << log::endl << log::endl;
    (void)loadMapFile(1);

    // Process all keyframes
    for (keyframe = 0; keyframe < opt.video.keyframes; keyframe++) {

        log::cout << log::vspace;
        log::cout << "Zooming from keyframe " << std::to_string(keyframe);
        log::cout << " to keyframe " << std::to_string(keyframe + 1) << ": ";
        log::cout << std::to_string(opt.video.inbetweens) << " inbetweens";
        log::cout << log::endl << log::endl;

        updateClock.reset();
        renderClock.reset();
        recordClock.reset();

        // Process all inbetweens
        for (frame = 0; frame < opt.video.inbetweens; frame++) {

            // Process all events
            if (!window.isOpen()) throw UserInterruptException();
            while (window.pollEvent(event)) {

                if (event.type == sf::Event::Closed)
                    window.close();
            }

            //Perform main tasks
            update();
            draw();
            record();

            // progress.step(1);
        }

        // progress.done();

        if (opt.flags.verbose) {

            log::cout << log::vspace;
            log::cout << log::ralign("Update: ");
            log::cout << updateClock.getElapsedTime() << log::endl;
            log::cout << log::ralign("Render: ");
            log::cout << renderClock.getElapsedTime() << log::endl;

            if (recordMode) {

                log::cout << log::ralign("Record: ");
                log::cout << recordClock.getElapsedTime() << log::endl;
            }
            log::cout << log::vspace;
        }

        // Wait for the async map file loader to finish
        (void)loadResult.get();
    }

    // Stop FFmpeg
    if (recordMode) recorder.stopRecording();
}
*/

void
Zoomer::launch()
{
    sf::Event event;

    // Start FFmpeg
    if (recordMode) recorder.startRecording();

    // Set animation parameter
    zoom.set(1.0);
    zoom.set(4.0, 2 * opt.video.inbetweens);

    // Experimental (REMOVE ASAP)
    zoom.set(2.0);
    zoom.set(0.5, 2 * opt.video.inbetweens);
    keyframe = 12;

    // Process all keyframes
    while (1) {

        // Process all events
        if (!window.isOpen()) throw UserInterruptException();
        while (window.pollEvent(event)) {

            if (event.type == sf::Event::Closed)
                window.close();
        }

        //Perform main tasks
        update();
        draw();
        record();

    }

        // Wait for the async map file loader to finish
        // (void)loadResult.get();

    // Stop FFmpeg
    // if (recordMode) recorder.stopRecording();
}

/*
void
Zoomer::update()
{
    if (frame == 0) {

        // Preload the next texture in the background
        loadResult = std::async([this]() {

            if (keyframe + 2 > opt.video.keyframes) {
                return false;
            }

            updateClock.go();
            auto result = loadMapFile(keyframe + 2);
            updateClock.stop();

            return result;
        });

        // Set animation start point
        zoom.set(1.0);

        // Set animation end point and speed
        zoom.set(2.0, opt.video.inbetweens);

        // Update window title bar
        string title = "DeepZoom - ";
        title += recordMode ? "Recording " : "Preview ";
        title += "[Keyframe " + std::to_string(keyframe + 1);
        title += " / " + std::to_string(opt.video.keyframes) + "] ";
        window.setTitle(title);

    } else {

        zoom.move();
    }
}
*/

void
Zoomer::update()
{
    //
    // Perform animation
    //

    zoom.move();

    if (zoom.current >= 2.0) {

        // Switch to next frame
        keyframe++;
        zoom.current /= 2.0;
    }

    if (zoom.current < 1.0 && keyframe > 0) {

        // Switch to previous frame
        keyframe--;
        zoom.current *= 2.0;
    }

    //
    // Update slots
    //

    // Check for dirty slots
    for (isize i = -1; i <= 2; i++) {

        auto frameNr = keyframe + i;
        auto nr = slotNr(frameNr);

        if (slot[nr] != frameNr) {

            // Preload the next texture in the background
            slot[nr] = frameNr;
            mapState[nr] = MapState::Loading;
            loadResult[nr] = loadMapFileAsync(frameNr);
        }
    }

    // Wait for the loaders to finish the maps we need
    for (isize i = keyframe; i <= keyframe + 1; i++) {

        if (mapState[slotNr(i)] != MapState::UpToDate) {

            (void)loadResult[slotNr(i)].get();
            mapState[slotNr(i)] = MapState::UpToDate;
        }
    }

    // Update window title bar
    /*
    string title = "DeepZoom - ";
    title += recordMode ? "Recording " : "Preview ";
    title += "[Keyframe " + std::to_string(keyframe + 1);
    title += " / " + std::to_string(opt.video.keyframes) + "] ";
    window.setTitle(title);
    */
}

void
Zoomer::draw()
{
    renderClock.go();

    // Colorize
    colorizer.draw(drillMap[slotNr(keyframe + 0)].colorMap,
                   drillMap[slotNr(keyframe + 1)].colorMap,
                   (float)frame / (float)opt.video.inbetweens,
                   float(zoom.current));

    // Display the result
    window.clear();
    window.draw(colorizer.getRect());
    window.display();

    renderClock.stop();
}

void
Zoomer::record()
{
    if (recordMode) {

        recordClock.go();
        recorder.record(colorizer.getImage());
        recordClock.stop();
    }
}

std::future<bool>
Zoomer::loadMapFileAsync(isize nr)
{
    // Preload the next texture in the background
    return std::async([this, nr]() {

        updateClock.go();
        auto result = loadMapFile(nr);
        updateClock.stop();

        return result;
    });
}

bool
Zoomer::loadMapFile(isize nr)
{
    fs::path path = opt.files.inputs.front();
    fs::path file = path / AssetManager::mapFile(nr);

    drillMap[slotNr(nr)].load(file);
    drillMap[slotNr(nr)].colorize();

    return true;
}

}
