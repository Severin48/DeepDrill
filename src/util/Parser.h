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
#include <functional>

namespace dd {

class Parser {

    typedef std::function<void(string,string)> Callback;

public:
    
    static void parse(const fs::path &path, Callback callback, isize nr = 0);
    static void parse(std::ifstream &stream, Callback callback, isize nr = 0);
    static void parse(std::stringstream &stream, Callback callback, isize nr = 0);

private:

    static std::pair<isize,isize> getRange(string &key, bool strip = false);
    static std::pair<isize,isize> stripRange(string &key) { return getRange(key, true); }

    static void ltrim(string &s);
    static void rtrim(string &s);
    static void trim(string &s);
    static void erase(string &s, char c);
    static void tolower(string &s);
};

}
