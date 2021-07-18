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

namespace dd {

class Maker {

    // Key-value pairs
    map<string,string> &keys;
    
    // Configuration options
    const struct Options &opt;
    
    // Names
    string project;
    
    // Paths
    string execPath;
    string pathLoc;
    string pathPrf;
    string pathMakefile;
    
public:
    
    // Constructor
    Maker(map<string,string> &keys, const Options &opt);
    
    // Main entry point
    void generate(); 
    
private:
    
    void generateLocationFile();
    void generateProfile();
    void generateMakefile();
    
    void writeHeader(std::ofstream &os);
};

}
