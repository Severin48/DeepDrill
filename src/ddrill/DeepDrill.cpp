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

#include "DeepDrill.h"
#include "SlowDriller.h"
#include "Driller.h"
#include "ColorMap.h"
#include "ProgressIndicator.h"
#include "Compressor.h"

#include <getopt.h>

int main(int argc, char *argv[])
{
    return dd::DeepDrill().main(argc, argv);
}

namespace dd {

void
DeepDrill::syntax()
{
    log::cout << "Usage: ";
    log::cout << "deepdrill [-bv] [-a <path>] [-c <keyvalue>] [-p <profile>] -o <output> <input>" << log::endl;
    log::cout << log::endl;
    log::cout << "       -b or --batch     Run in batch mode" << log::endl;
    log::cout << "       -v or --verbose   Run in verbose mode" << log::endl;
    log::cout << "       -a or --assets    Optional path to asset files" << log::endl;
    log::cout << "       -p or --profile   Customize settings" << log::endl;
    log::cout << "       -o or --output    Output file" << log::endl;
}

void
DeepDrill::parseArguments(int argc, char *argv[])
{
    static struct option long_options[] = {
        
        { "verbose",  no_argument,       NULL, 'v' },
        { "batch",    no_argument,       NULL, 'b' },
        { "assets",   required_argument, NULL, 'a' },
        { "profile",  required_argument, NULL, 'p' },
        { "output",   required_argument, NULL, 'o' },
        { NULL,       0,                 NULL,  0  }
    };

    // Don't print the default error messages
    opterr = 0;
    
    // Remember the path to the DeppDrill executable
    opt.files.exec = makeAbsolutePath(argv[0]);

    // Parse all options
    while (1) {
        
        int arg = getopt_long(argc, argv, ":vba:p:o:", long_options, NULL);
        if (arg == -1) break;

        switch (arg) {
                
            case 'v':
                opt.flags.verbose = true;
                break;
                
            case 'b':
                opt.flags.batch = true;
                break;

            case 'a':
                assets.addSearchPath(optarg);
                break;

            case 'p':
                opt.files.profiles.push_back(optarg);
                break;
            
            case 'o':
                opt.files.outputs.push_back(optarg);
                break;

            case ':':
                throw SyntaxError("Missing argument for option '" +
                                  string(argv[optind - 1]) + "'");
                
            default:
                throw SyntaxError("Invalid option '" +
                                  string(argv[optind - 1]) + "'");
        }
    }
    
    // Parse remaining arguments
    while (optind < argc) {

        string arg = argv[optind++];

        if (arg.find('=') != std::string::npos) {
            opt.overrides.push_back(arg);
        } else {
            opt.files.inputs.push_back(arg);
        }
    }

    // Add default outputs if none are given
    if (opt.files.outputs.empty() && !opt.files.inputs.empty()) {

        auto name = stripSuffix(opt.files.inputs.front());
        opt.files.outputs.push_back(name + ".map");
        opt.files.outputs.push_back(name + ".jpg");
    }
}

void
DeepDrill::checkCustomArguments()
{
    // The user needs to specify at least one output file
    if (opt.files.outputs.size() < 1) throw SyntaxError("No output file is given");

    // Valid input files are map files and locations files
    for (auto &it: opt.files.inputs) {
        (void)assets.findAsset(it, { Format::LOC, Format::MAP });
    }

    // Valid output files are map files and image files
    for (auto &it: opt.files.outputs) {
        AssetManager::assureFormat(it, { Format::MAP, Format::BMP, Format::JPG, Format::PNG });
    }
}

void
DeepDrill::run()
{
    auto input = opt.files.inputs.front();
    auto inputFormat = AssetManager::getFormat(input);

    drillMap.resize();

    // Create or load the drill map
    if (inputFormat == Format::MAP) {

        // Load the drill map from disk
        drillMap.load(input);

    } else {

        BatchProgressIndicator progress(opt, "Drilling",  opt.files.outputs.front());

        // Run the driller
        if (opt.perturbation.enable) {

            Driller driller(opt, drillMap);
            driller.drill();

        } else {

            SlowDriller driller(opt, drillMap);
            driller.drill();
        }
    }

    // Generate outputs
    for (auto &it : opt.files.outputs) {

        auto outputFormat = AssetManager::getFormat(it);

        if (AssetManager::isImageFormat(outputFormat)) {

            // Create and save image file
            colorizer.init(opt.image.illuminator, opt.image.scaler);
            colorizer.draw(drillMap);
            colorizer.save(it, outputFormat);

        } else {

            // Save map file
            drillMap.save(it);
        }
    }

    // Analyze the drill map
    if (inputFormat != Format::MAP && opt.flags.verbose) {

        // log::cout << log::vspace << "Finalizing" << log::endl << log::endl;
        drillMap.analyze();
    }
}

}
