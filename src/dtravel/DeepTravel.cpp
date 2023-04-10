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
#include "DeepTravel.h"
#include "Options.h"
#include "Traveller.h"
#include "Logger.h"
#include <getopt.h>

int main(int argc, char *argv[])
{
    using namespace dd;

    try {

        DeepTravel().main(argc, argv);

    } catch (dd::SyntaxError &e) {

        log::cout << "Usage: ";
        log::cout << "deeptravel [-bv] [-p <profile>] -o <output> <input>" << log::endl;
        log::cout << log::endl;
        log::cout << "       -b or --batch    Run in batch mode" << log::endl;
        log::cout << "       -v or --verbose  Run in verbose mode" << log::endl;
        log::cout << "       -p or --profile  Customize settings" << log::endl;
        log::cout << log::endl;

        if (!e.description.empty()) {
            log::cout << e.what() << log::endl;
        }

        return 1;

    } catch (std::exception &e) {

        log::cout << e.what() << log::endl;
        return 1;
    }

    return 0;
}

namespace dd {

void
DeepTravel::main(int argc, char *argv[])
{
    map<string,string> keys;
    string option = "";

    // Parse all command line arguments
    parseArguments(argc, argv, keys);

    log::cout << "DeepTravel " << VER_MAJOR << "." << VER_MINOR;
    log::cout << " - (C)opyright Dirk W. Hoffmann";
    log::cout << log::endl << log::endl;

    // Setup the GMP library
    setupGmp(keys);

    // Parse all options
    Options opt(keys);

    // Start a stop watch
    Clock stopWatch;

    // Execute
    Traveller(opt).launch();

    log::cout << log::vspace << "Total time: " << stopWatch.stop() << log::endl;
}

void
DeepTravel::parseArguments(int argc, char *argv[], map<string,string> &keys)
{
    static struct option long_options[] = {

        { "verbose", no_argument,       NULL, 'v' },
        { "batch",   no_argument,       NULL, 'b' },
        { "profile", required_argument, NULL, 'p' },
        { "output",  required_argument, NULL, 'o' },
        { NULL,      0,                 NULL,  0  }
    };

    // Don't print the default error messages
    opterr = 0;

    // Remember the path to the executable
    keys["exec"] = makeAbsolutePath(argv[0]);

    // Parse all options
    while (1) {

        int arg = getopt_long(argc, argv, ":vbp:o:", long_options, NULL);
        if (arg == -1) break;

        switch (arg) {

            case 'v':
                keys["verbose"] = "1";
                break;

            case 'b':
                log::cout.setSilent(true);
                break;

            case 'p':
                profiles.push_back(makeAbsolutePath(optarg));
                break;

            case 'o':
                outputs.push_back(makeAbsolutePath(optarg));
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
        inputs.push_back(makeAbsolutePath(argv[optind++]));
    }

    checkArguments(keys);
    readInputs(keys);
    readOutputs(keys);
    readProfiles(keys);
}

void
DeepTravel::checkArguments(map<string,string> &keys)
{
    // The user needs to specify a single input
    if (inputs.size() < 1) throw SyntaxError("No input file is given");
    if (inputs.size() > 1) throw SyntaxError("More than one input file is given");

    // The user needs to specify at most one output
    if (outputs.size() > 1) throw SyntaxError("More than one output file is given");

    // All profiles must be files of type ".prf"
    for (auto &it : profiles) {
        if (extractSuffix(it) != "prf") {
            throw SyntaxError(it + " is not a profile (.prf)");
        }
    }

    if (!outputs.empty()) {

        auto out = outputs.front();
        auto outSuffix = extractSuffix(out);

        // The output file must be a mpg file
        if (outSuffix != "mpg") {
            throw SyntaxError(out + ": Invalid format. Expected .mpg");
        }

        // The output file must be writable
        std::ofstream file(out, std::ofstream::out);
        if (!file.is_open()) {
            throw SyntaxError("Can't write to file " + out);
        }
    }
}

void
DeepTravel::readInputs(map<string,string> &keys)
{
    string path = inputs.front();
    string name;

    keys["input"] = path;

    for (isize i = 0;; i++) {

        name = path + "_" + std::to_string(i) + ".png";
        if (!fileExists(name)) break;
        imageFiles.push_back(name);
    }

    if (imageFiles.empty()) {
        throw FileNotFoundError("File " + name + " does not exist");
    }
}

void
DeepTravel::readOutputs(map<string,string> &keys)
{
    if (!outputs.empty()) {

        auto path = outputs.front();
        auto suffix = extractSuffix(path);

        keys["output"] = path;

        if (suffix == "mpg") {
            return;
        }

        throw SyntaxError(path + ": Unknown output format");
    }
}

void
DeepTravel::readProfiles(map<string,string> &keys)
{
    for (auto &path: profiles) {
        Parser::parse(path, keys);
    }
}

void
DeepTravel::setupGmp(std::map <string,string> &keys)
{
    mpf_set_default_prec(64);
}

}
