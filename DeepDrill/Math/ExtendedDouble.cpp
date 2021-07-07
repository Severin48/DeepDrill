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

#include "Types.h"

namespace dd {

ExtendedDouble::ExtendedDouble(double m)
{
    mantissa = m;
    exponent = 0;
}

ExtendedDouble::ExtendedDouble(const mpf_class &value)
{
    mantissa = mpf_get_d_2exp(&exponent, value.get_mpf_t());
}

ExtendedDouble &
ExtendedDouble::operator=(const mpf_class &other)
{
    mantissa = mpf_get_d_2exp(&exponent, other.get_mpf_t());
    return *this;
}

std::ostream& operator<<(std::ostream& os, const ExtendedDouble& value)
{
    return os << value.mantissa << "*2^" << value.exponent;
}

}
