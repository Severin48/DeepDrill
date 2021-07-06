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

#include "Types.h"

namespace dd {

class UnitTester {

public:
    
    UnitTester();
    void run();

private:
    
    void assertAlmostEqual(const double x, const double y);
    void assertAlmostEqual(const StandardComplex &x, const StandardComplex &y);
};

}
