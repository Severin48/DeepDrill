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

#include "config.h"
#include "Types.h"
#include "Chrono.h"
#include "Coord.h"
#include "Recorder.h"
#include "Animated.h"
#include "DrillMap.h"
#include "ColorMap.h"
#include "Colorizer.h"
#include "Filter.h"

#include <SFML/Graphics.hpp>
#include <future>

namespace dd {

class Zoomer {

    // Configuration options
    struct Options &opt;

    // The application window
    sf::RenderWindow window;

    // Drill maps (read from map files)
    DrillMap drillMap[3] = { DrillMap(opt), DrillMap(opt), DrillMap(opt) };

    // Colorizer for converting the drill maps into an image
    Colorizer colorizer = Colorizer(opt);

    // The video recorder
    Recorder recorder = Recorder(opt);

    // Indicates if the application runs in record mode or preview mode
    bool recordMode;

    // The currently processed keyframe and inbetween
    isize keyframe = 0;
    isize frame = 0;

    // Animation parameters
    Animated zoom;

    // Stop watches
    Clock updateClock;
    Clock renderClock;
    Clock recordClock;

    // Synchronizer for the async map file loader
    std::future<bool> loadResult;

public:

    // Constructors
    Zoomer(Options &opt);

    // Initializers
    void init();

    // Main entry point
    void launch();

private:

    // Called inside the main loop
    void update();
    void draw();
    void record();

    // Loads a new map file from disk
    bool loadMapFile(isize nr);
};

}
