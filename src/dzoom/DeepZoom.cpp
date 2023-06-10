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

#include "DeepZoom.h"
#include "Zoomer.h"

#include <getopt.h>

int main(int argc, char *argv[])
{
    return dd::DeepZoom().main(argc, argv);
}

namespace dd {

void
DeepZoom::syntax()
{
    log::cout << "Usage: ";
    log::cout << "deepzoom [-bv] [-p <profile>] [-o <output>] <input>" << log::endl;
    log::cout << log::endl;
    log::cout << "       -b or --batch     Run in batch mode" << log::endl;
    log::cout << "       -v or --verbose   Run in verbose mode" << log::endl;
    log::cout << "       -a or --assets    Optional path to asset files" << log::endl;
    log::cout << "       -c or --config    Configures a single key-value pair" << log::endl;
    log::cout << "       -p or --profile   Customize settings" << log::endl;
}

void
DeepZoom::initialize()
{
    // Check for shader support
    if (!sf::Shader::isAvailable()) {
        throw Exception("No GPU shader support");
    }

    // Initialize FFmpeg
    FFmpeg::init();
}

void
DeepZoom::parseArguments(int argc, char *argv[])
{
    static struct option long_options[] = {

        { "verbose",  no_argument,       NULL, 'v' },
        { "batch",    no_argument,       NULL, 'b' },
        { "assets",   required_argument, NULL, 'a' },
        { "config",   required_argument, NULL, 'c' },
        { "profile",  required_argument, NULL, 'p' },
        { "output",   required_argument, NULL, 'o' },
        { NULL,       0,                 NULL,  0  }
    };

    // Don't print the default error messages
    opterr = 0;

    // Remember the path to the executable
    opt.files.exec = makeAbsolutePath(argv[0]);
    
    // Parse all options
    while (1) {

        int arg = getopt_long(argc, argv, ":vba:c:p:o:", long_options, NULL);
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

            case 'c':
                opt.overrides.push_back(optarg);
                break;

            case 'p':
                opt.files.profiles.push_back(makeAbsolutePath(optarg));
                break;

            case 'o':
                opt.files.outputs.push_back(makeAbsolutePath(optarg));
                break;

            case ':':
                throw SyntaxError("Missing argument for option '" +
                                  string(argv[optind - 1]) + "'");

            default:
                throw SyntaxError("Invalid option '" +
                                  string(argv[optind - 1]) + "'");
        }
    }

    // Parse all remaining arguments
    while (optind < argc) {
        opt.files.inputs.push_back(argv[optind++]);
    }
}

void
DeepZoom::checkCustomArguments()
{
    // The input file must be a prj file
    (void)assets.findAsset(opt.files.inputs.front(), Format::PRJ);

    // The user must not specify more than one output file
    if (opt.files.outputs.size() > 1) throw SyntaxError("More than one output file is given");

    if (!opt.files.outputs.empty()) {

        auto output = opt.files.outputs.front();

        // The output file must be a video file
        AssetManager::assureFormat(output, Format::MPG);

        // The output file must be writable
        std::ofstream file(output, std::ofstream::out);
        if (!file.is_open()) {
            throw SyntaxError("Can't write to file " + output.string());
        }

        // FFmpeg must be installed
        if (!FFmpeg::available()) {
            throw Exception("Unable to locate FFmpeg.");
        }
    }
}

void
DeepZoom::run()
{
    Zoomer(opt).launch();
}

}
