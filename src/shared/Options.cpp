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

#include "Options.h"
#include "Coord.h"
#include "Exception.h"
#include "IO.h"
#include "Logger.h"
#include "Parser.h"

namespace dd {

Options::Options(const AssetManager &assets) : assets(assets)
{
    // Register default keys

    // Location keys
    defaults["location.real"] = "0.0";
    defaults["location.imag"] = "0.0";
    defaults["location.zoom"] = "1.0";
    defaults["location.depth"] = "800";

    // Map keys
    defaults["map.width"] = "1920";
    defaults["map.height"] = "1080";
    defaults["map.depth"] = "1";
    defaults["map.compress"] = "true";

    // Image keys
    defaults["image.width"] = "1920";
    defaults["image.height"] = "1080";
    defaults["image.depth"] = "0";
    defaults["image.illuminator"] = "lambert.glsl";
    defaults["image.scaler"] = "bicubic.glsl";

    // Video keys
    defaults["video.framerate"] = "60";
    defaults["video.keyframes"] = "0";
    defaults["video.inbetweens"] = "0";
    defaults["video.inbetweens2"] = "0:00/180, 0:05/90, 0:10/30";
    defaults["video.bitrate"] = "8000";
    defaults["video.scaler"] = "tricubic.glsl";

    // Color keys
    defaults["colors.mode"] = "default";
    defaults["colors.palette"] = "";
    defaults["colors.texture"] = "";
    defaults["colors.scale"] = "1.0";
    defaults["colors.opacity"] = "0.5";
    defaults["colors.alpha"] = "45";
    defaults["colors.beta"] = "45";

    // Perturbation keys
    defaults["perturbation.enable"] = "yes";
    defaults["perturbation.tolerance"] = "1e-6";
    defaults["perturbation.badpixels"] = "0.001";
    defaults["perturbation.rounds"] = "50";
    defaults["perturbation.color"] = "black";

    // Approximation keys
    defaults["approximation.enable"] = "yes";
    defaults["approximation.coefficients"] = "5";
    defaults["approximation.tolerance"] = "1e-12";

    // Area checking keys
    defaults["areacheck.enable"] = "yes";
    defaults["areacheck.color"] = "black";

    // Attractor checking keys
    defaults["attractorcheck.enable"] = "yes";
    defaults["attractorcheck.tolerance"] = "1e-34";
    defaults["attractorcheck.color"] = "black";

    // Period checking keys
    defaults["periodcheck.enable"] = "yes";
    defaults["periodcheck.tolerance"] = "1e-74";
    defaults["periodcheck.color"] = "black";
}

std::vector <fs::path>
Options::getInputs(Format format)
{
    std::vector <fs::path> result;

    for (const auto &it : files.inputs) {
        if (AssetManager::getFormat(it) == format) result.push_back(it);
    }
    return result;
}

std::vector <fs::path>
Options::getOutputs(Format format)
{
    std::vector <fs::path> result;

    for (const auto &it : files.outputs) {
        if (AssetManager::getFormat(it) == format) result.push_back(it);
    }
    return result;
}

void
Options::parse(string keyvalue)
{
    // Seek the '=' symbol
    auto pos = keyvalue.find("=");
    if (pos == std::string::npos) throw SyntaxError("Parse error");

    // Split string
    auto key = keyvalue.substr(0, pos);
    auto value = keyvalue.substr(pos + 1, std::string::npos);

    parse(key, value);
}

void
Options::parse(string key, string value)
{
    // Convert the key to lower case
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char ch){ return std::tolower(ch); });

    // Strip quotation marks
    value.erase(remove(value.begin(), value.end(), '\"'), value.end());

    keys[key] = value;

    if (key == "location.real") {

        Parser::parse(key, value, location.real);

    } else if (key == "location.imag") {

        Parser::parse(key, value, location.imag);

    } else if (key == "location.zoom") {

        Parser::parse(key, value, location.zoom);

    } else if (key == "location.depth") {

        Parser::parse(key, value, location.depth);

    } else if (key == "map.width") {

        Parser::parse(key, value, drillmap.width, MIN_MAP_WIDTH, MAX_MAP_WIDTH);

    } else if (key == "map.height") {

        Parser::parse(key, value, drillmap.height, MIN_MAP_HEIGHT, MAX_MAP_HEIGHT);

    } else if (key == "map.depth") {

        Parser::parse(key, value, drillmap.depth, 0, 1);

    } else if (key == "map.compress") {

        Parser::parse(key, value, drillmap.compress);

    } else if (key == "image.width") {

        Parser::parse(key, value, image.width, MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH);

    } else if (key == "image.height") {

        Parser::parse(key, value, image.height, MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT);

        if (image.height % 2 == 1) {
            throw KeyValueError(key, "Height must be dividable by 2.");
        }

    } else if (key == "image.depth") {

        Parser::parse(key, value, image.depth, 0, 1);

    } else if (key == "image.illuminator") {

        image.illuminator = assets.findAsset(value, Format::GLSL);

    } else if (key == "image.scaler") {

        image.scaler = assets.findAsset(value, Format::GLSL);

    } else if (key == "video.framerate") {

        Parser::parse(key, value, video.frameRate, 25, 240);

    } else if (key == "video.keyframes") {

        Parser::parse(key, value, video.keyframes);

    } else if (key == "video.inbetweens") {

        Parser::parse(key, value, video.inbetweens);

    } else if (key == "video.inbetweens2") {

        Parser::parse(key, value, video.inbetweens2);

    } else if (key == "video.bitrate") {

        Parser::parse(key, value, video.bitrate);

    } else if (key == "video.scaler") {

        video.scaler = assets.findAsset(value, Format::GLSL);

    } else if (key == "colors.mode") {

        Parser::parse(key, value, colors.mode);

    } else if (key == "colors.palette") {

        if (value != "") {
            colors.palette = assets.findAsset(value, { Format::BMP, Format::JPG, Format:: PNG });
        }

    } else if (key == "colors.texture") {

        if (value != "") {
            colors.texture = assets.findAsset(value, { Format::BMP, Format::JPG, Format:: PNG });
        }

    } else if (key == "colors.scale") {

        Parser::parse(key, value, colors.scale);

    } else if (key == "colors.opacity") {

        Parser::parse(key, value, colors.opacity, 0.0, 1.0);

    } else if (key == "colors.alpha") {

        Parser::parse(key, value, colors.alpha);

        if (colors.alpha < 0.0 || colors.alpha >= 360.0) {
            throw KeyValueError(key, "Angle out of range");
        }

    } else if (key == "colors.beta") {

        Parser::parse(key, value, colors.beta);

        if (colors.beta < 0.0 || colors.beta >= 360.0) {
            throw KeyValueError(key, "Angle out of range");
        }

    } else if (key == "areacheck.enable") {

        Parser::parse(key, value, areacheck.enable);

    } else if (key == "areacheck.color") {

        Parser::parse(key, value, areacheck.color);

    } else if (key == "attractorcheck.enable") {

        Parser::parse(key, value, attractorcheck.enable);

    } else if (key == "attractorcheck.tolerance") {

        Parser::parse(key, value, attractorcheck.tolerance);

    } else if (key == "attractorcheck.color") {

        Parser::parse(key, value, attractorcheck.color);

    } else if (key == "periodcheck.enable") {

        Parser::parse(key, value, periodcheck.enable);

    } else if (key == "periodcheck.tolerance") {

        Parser::parse(key, value, periodcheck.tolerance);

    } else if (key == "periodcheck.color") {

        Parser::parse(key, value, periodcheck.color);

    } else if (key == "perturbation.enable") {

        Parser::parse(key, value, perturbation.enable);

    } else if (key == "perturbation.tolerance") {

        Parser::parse(key, value, perturbation.tolerance);

    } else if (key == "perturbation.badpixels") {

        Parser::parse(key, value, perturbation.badpixels);

    } else if (key == "perturbation.rounds") {

        Parser::parse(key, value, perturbation.rounds);

    } else if (key == "perturbation.color") {

        Parser::parse(key, value, perturbation.color);

    } else if (key == "approximation.enable") {

        Parser::parse(key, value, approximation.enable);

    } else if (key == "approximation.coefficients") {

        Parser::parse(key, value, approximation.coefficients);

    } else if (key == "approximation.tolerance") {

        Parser::parse(key, value, approximation.tolerance);

    } else {

        throw KeyValueError(key, "Unexpected key");
    }
}

/*
void
Options::parse(const string &key, const string &value, string &parsed)
{
    parsed = value;
}

void
Options::parse(const string &key, const string &value, bool &parsed)
{
    if (value == "true" || value == "yes" || value == "on") {

        parsed = true;
        return;
    }
    if (value == "false" || value == "no" || value == "off") {

        parsed = false;
        return;
    }

    throw Exception("Invalid argument for key " + key + ": " + value);
}

void
Options::parse(const string &key, const string &value, isize &parsed)
{
    try {
        parsed = stol(value);
    } catch (...) {
        throw Exception("Invalid argument for key " + key + ": " + value);
    }
}

void
Options::parse(const string &key, const string &value, isize &parsed, isize min, isize max)
{
    parse(key, value, parsed);

    if (parsed < min) {
        throw Exception("Invalid argument for key " + key +
                        ": Value must be >= " + std::to_string(min));
    }
    if (parsed > max) {
        throw Exception("Invalid argument for key " + key +
                        ": Value must be <= " + std::to_string(max));
    }
}

void
Options::parse(const string &key, const string &value, double &parsed)
{
    try {
        parsed = stod(value);
    } catch (...) {
        throw Exception("Invalid argument for key " + key + ": " + value);
    }
}

void
Options::parse(const string &key, const string &value, double &parsed, double min, double max)
{
    parse(key, value, parsed);

    if (parsed < min) {
        throw Exception("Invalid argument for key " + key +
                        ": Value must be >= " + std::to_string(min));
    }
    if (parsed > max) {
        throw Exception("Invalid argument for key " + key +
                        ": Value must be <= " + std::to_string(max));
    }
}

void
Options::parse(const string &key, const string &value, mpf_class &parsed)
{
    try {
        parsed = mpf_class(value);
    } catch (...) {
        throw Exception("Invalid argument for key " + key + ": " + value);
    }
}

void
Options::parse(const string &key, const string &value, GpuColor &parsed)
{
    std::map <string, GpuColor> modes = {

        { "black",      GpuColor::black     },
        { "white",      GpuColor::white     },
        { "red",        GpuColor::red       },
        { "green",      GpuColor::green     },
        { "blue",       GpuColor::blue      },
        { "yellow",     GpuColor::yellow    },
        { "magenta",    GpuColor::magenta   },
        { "cyan",       GpuColor::cyan      }
    };

    try {
        parsed = modes.at(value);
    } catch (...) {
        throw Exception("Invalid argument for key " + key + ": " + value);
    }
}

void
Options::parse(const string &key, const string &value, ColoringMode &parsed)
{
    std::map <string, ColoringMode> modes = {

        { "default", ColoringMode::Default },
        // { "textured", ColoringMode::Textured }
    };

    try {
        parsed = modes.at(value);
    } catch (...) {
        throw Exception("Invalid argument for key " + key + ": " + value);
    }
}
*/

void
Options::applyDefaults()
{
    // Adjust some default values
    if (keys.find("image.width") != keys.end()) {
        defaults["map.width"] = keys["image.width"];
    }
    if (keys.find("image.height") != keys.end()) {
        defaults["map.height"] = keys["image.height"];
    }

    // Use default values for all missing keys
    for (auto &it : defaults) {
        if (keys.find(it.first) == keys.end()) {
            parse(it.first, it.second);
        }
    }

    // Apply overrides
    for (auto &it : overrides) {
        parse(it);
    }
}

void
Options::derive()
{
    // Derive unspecified video parameters
    if (!video.inbetweens) {

        video.inbetweens = 2 * video.frameRate;
    }
    if (!video.keyframes) {

        auto zoom = ExtendedDouble(location.zoom);
        video.keyframes = isize(std::ceil(zoom.log2().asDouble()));
    }
}

}
