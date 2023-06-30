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
//    defaults["video.velocity"] = "0:00/1.0, 0:01/-1.0, 0:02/1.0, 0:03/-1.0, 0:04/1.0, 0:5/-1.0, 0:6/1.0, 0:70/1.0";
//    defaults["video.velocity"] = "0:00/1.0, 0:01/-1.0, 0:02/2.0, 0:03/-2.0, 0:04/3.0, 0:5/-3.0, 0:6/4.0, 0:7/1.0, 0:70/1.0";
    defaults["video.velocity"] = "0.5";
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

    // Remove white spaces
    erase(value, ' ');

    // Strip quotation marks
    value.erase(remove(value.begin(), value.end(), '\"'), value.end());

    keys[key] = value;

    try {

        if (key == "location.real") {

            Parser::parse(value, location.real);

        } else if (key == "location.imag") {

            Parser::parse(value, location.imag);

        } else if (key == "location.zoom") {

            Parser::parse(value, location.zoom);

        } else if (key == "location.depth") {

            Parser::parse(value, location.depth);

        } else if (key == "map.width") {

            Parser::parse(value, drillmap.width, MIN_MAP_WIDTH, MAX_MAP_WIDTH);

        } else if (key == "map.height") {

            Parser::parse(value, drillmap.height, MIN_MAP_HEIGHT, MAX_MAP_HEIGHT);

        } else if (key == "map.depth") {

            Parser::parse(value, drillmap.depth, 0, 1);

        } else if (key == "map.compress") {

            Parser::parse(value, drillmap.compress);

        } else if (key == "image.width") {

            Parser::parse(value, image.width, MIN_IMAGE_WIDTH, MAX_IMAGE_WIDTH);

        } else if (key == "image.height") {

            Parser::parse(value, image.height, MIN_IMAGE_HEIGHT, MAX_IMAGE_HEIGHT);

            if (image.height % 2 == 1) {
                throw Exception("Height must be dividable by 2");
            }

        } else if (key == "image.depth") {

            Parser::parse(value, image.depth, 0, 1);

        } else if (key == "image.illuminator") {

            image.illuminator = assets.findAsset(value, Format::GLSL);

        } else if (key == "image.scaler") {

            image.scaler = assets.findAsset(value, Format::GLSL);

        } else if (key == "video.framerate") {

            Parser::parse(value, video.frameRate, 25, 240);

        } else if (key == "video.keyframes") {

            Parser::parse(value, video.keyframes);

        } else if (key == "video.velocity") {

            Parser::parse(value, video.velocity);

        } else if (key == "video.bitrate") {

            Parser::parse(value, video.bitrate);

        } else if (key == "video.scaler") {

            video.scaler = assets.findAsset(value, Format::GLSL);

        } else if (key == "colors.mode") {

            Parser::parse(value, colors.mode);

        } else if (key == "colors.palette") {

            if (value != "") {
                colors.palette = assets.findAsset(value, { Format::BMP, Format::JPG, Format:: PNG });
            }

        } else if (key == "colors.texture") {

            if (value != "") {
                colors.texture = assets.findAsset(value, { Format::BMP, Format::JPG, Format:: PNG });
            }

        } else if (key == "colors.scale") {

            Parser::parse(value, colors.scale);

        } else if (key == "colors.opacity") {

            Parser::parse(value, colors.opacity, 0.0, 1.0);

        } else if (key == "colors.alpha") {

            Parser::parse(value, colors.alpha);

            if (colors.alpha < 0.0 || colors.alpha >= 360.0) {
                throw Exception("Angle out of range");
            }

        } else if (key == "colors.beta") {

            Parser::parse(value, colors.beta);

            if (colors.beta < 0.0 || colors.beta >= 360.0) {
                throw Exception("Angle out of range");
            }

        } else if (key == "areacheck.enable") {

            Parser::parse(value, areacheck.enable);

        } else if (key == "areacheck.color") {

            Parser::parse(value, areacheck.color);

        } else if (key == "attractorcheck.enable") {

            Parser::parse(value, attractorcheck.enable);

        } else if (key == "attractorcheck.tolerance") {

            Parser::parse(value, attractorcheck.tolerance);

        } else if (key == "attractorcheck.color") {

            Parser::parse(value, attractorcheck.color);

        } else if (key == "periodcheck.enable") {

            Parser::parse(value, periodcheck.enable);

        } else if (key == "periodcheck.tolerance") {

            Parser::parse(value, periodcheck.tolerance);

        } else if (key == "periodcheck.color") {

            Parser::parse(value, periodcheck.color);

        } else if (key == "perturbation.enable") {

            Parser::parse(value, perturbation.enable);

        } else if (key == "perturbation.tolerance") {

            Parser::parse(value, perturbation.tolerance);

        } else if (key == "perturbation.badpixels") {

            Parser::parse(value, perturbation.badpixels);

        } else if (key == "perturbation.rounds") {

            Parser::parse(value, perturbation.rounds);

        } else if (key == "perturbation.color") {

            Parser::parse(value, perturbation.color);

        } else if (key == "approximation.enable") {

            Parser::parse(value, approximation.enable);

        } else if (key == "approximation.coefficients") {

            Parser::parse(value, approximation.coefficients);

        } else if (key == "approximation.tolerance") {

            Parser::parse(value, approximation.tolerance);

        } else {

            throw Exception("Unknown key");
        }

    } catch (Exception &e) {
        throw KeyValueError(key, e.what());
    }
}

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
    if (!video.keyframes) {

        auto zoom = ExtendedDouble(location.zoom);
        video.keyframes = isize(std::ceil(zoom.log2().asDouble()));
    }
}

}
