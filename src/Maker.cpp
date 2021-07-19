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

#include "config.h"
#include "Maker.h"
#include "Logger.h"
#include "Options.h"

namespace dd {

Maker::Maker(Options &o) : opt(o)
{
    project = stripSuffix(stripPath(opt.input));
    projectDir = std::filesystem::path(opt.output);
}

void
Maker::generate()
{
    generateLocationFile();
    generateProfile();
    generateMakefile();
}

void
Maker::generateLocationFile()
{
    auto &keys = opt.keys;

    // Open output stream
    std::ofstream os(projectDir / (project + ".loc"));

    // Write header
    writeHeader(os);
    
    // Write location section
    os << "[location]" << std::endl;
    os << "real = " << keys["location.real"] << std::endl;
    os << "imag = " << keys["location.imag"] << std::endl;
    os << "zoom = " << keys["location.zoom"] << std::endl;
    os << "depth = " << keys["location.depth"] << std::endl;
    os << std::endl;
}

void
Maker::generateProfile()
{
    auto &keys = opt.keys;
    
    // Open output stream
    std::ofstream os(projectDir / (project + ".prf"));
    
    // Write header
    writeHeader(os);
    
    // Write image section
    os << "[image]" << std::endl;
    os << "width = " << keys["image.width"] << std::endl;
    os << "height = " << keys["image.height"] << std::endl;
    os << "badpixels = " << keys["image.badpixels"] << std::endl;
    os << std::endl;

    // Write perturbation section
    os << "[perturbation]" << std::endl;
    os << "tolerance = " << keys["perturbation.tolerance"] << std::endl;
    os << "rounds = " << keys["perturbation.rounds"] << std::endl;
    os << std::endl;

    // Write series approximation section
    os << "[approximation]" << std::endl;
    os << "coefficients = " << keys["approximation.coefficients"] << std::endl;
    os << "tolerance = " << keys["approximation.tolerance"] << std::endl;
    os << std::endl;

    // Write palette section
    os << "[palette]" << std::endl;
    os << "values = " << keys["palette.values"] << std::endl;
    os << std::endl;
}

void
Maker::generateMakefile()
{
    auto &keys = opt.keys;

    // Open output stream
    std::ofstream os(projectDir / "Makefile");

    // Write header
    writeHeader(os);
    os << std::endl;

    // Write definitions
    os << "DEEPDRILL = " << keys["exec"] << std::endl;
    os << std::endl;

    // Declare phony targets
    os << ".PHONY: all clean" << std::endl;
    os << std::endl;

    // Write targets
    os << "all: " << project << ".tiff" << std::endl;
    os << std::endl;

    os << project << ".tiff: " << project << ".map" << std::endl;
    os << "\t";
    os << "$(DEEPDRILL) -v ";
    os << " -p " << project << ".prf";
    os << " -o " << project << ".tiff";
    os << " " << project << ".map " << std::endl;
    os << std::endl;

    os << project << ".map: " << project << ".loc" << std::endl;
    os << "\t";
    os << "$(DEEPDRILL) -v ";
    os << " -p " << project << ".prf";
    os << " -o " << project << ".map";
    os << " " << project << ".loc " << std::endl;
    os << std::endl;

    os << "clean:" << std::endl;
    os << "\t";
    os << "rm *.map *.tiff" << std::endl;
    os << std::endl;
}

void
Maker::writeHeader(std::ofstream &os)
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    
    os << "# Generated by DeepDrill " << VER_MAJOR << "." << VER_MINOR;
    os << " on " << std::ctime(&time);
}

}
