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

#pragma once

#include "commons.h"
#include "Coord.h"
#include "Recorder.h"
#include "Animated.h"

#include <SFML/Graphics.hpp>

namespace dd {

class Zoomer {

    // Configuration options
    struct Options &opt;

    // The application window
    sf::RenderWindow window;
    
    // The source texture (read from image file)
    sf::Texture source;
    sf::RectangleShape sourceRect;

    // The downscaled source textures (latest three)
    sf::RenderTexture scaled[3];
    sf::RectangleShape scaledRect[3];

    // Storage position of the most recent scaled texture (experimental)
    isize latest = 0;

    // The render target
    sf::RenderTexture target;
    sf::RectangleShape targetRect;

    // Compute kernels
    sf::Shader scaler;
    sf::Shader merger;

    // The video recorder
    Recorder recorder = Recorder(opt);

    // Animation parameters
    Animated x, y, w, h;
//    isize keyframe = 0;
//    isize frame = 0;

public:

    // Constructor
    Zoomer(Options &opt);

    // Initializer
    void init();

    // Main entry point
    void launch();

private:

    // Called inside the main loop
    void update(isize keyframe, isize frame);
    void draw();

    // Indicates if we run in record mode or preview mode
    bool recordMode();

    // Loads a new image file from disk
    void updateTexture(isize nr);

    // Loads a new location from disk
    void updateLocation(isize nr, isize &dx, isize &dy);
};

}
