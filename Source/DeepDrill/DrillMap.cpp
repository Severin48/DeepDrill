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
#include "DrillMap.h"

namespace dd {

DrillMap::DrillMap(isize w, isize h)
{
    resize(w, h);
}

DrillMap::DrillMap(const string &path)
{
    load(path);
}

void
DrillMap::load(const string &path)
{
    std::ifstream os(path.c_str(), std::ios::binary);
    if (!os.is_open()) throw Exception("Failed to open file " + path);

    // Read header
    os.read((char *)&width, sizeof(width));
    os.read((char *)&height, sizeof(height));
    os.read((char *)&logBailout, sizeof(logBailout));
    
    // Adjust the map size
    resize(width, height);
    
    // Write data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            
            u32 iteration;
            float lognorm;
            
            os.read((char *)&iteration, sizeof(iteration));
            os.read((char *)&lognorm, sizeof(lognorm));

            set(x, y, MapEntry { iteration, lognorm });
        }
    }
}

void
DrillMap::resize(isize w, isize h)
{
    assert(w <= 3840);
    assert(h <= 2160);

    if (data) delete [] data;
    
    width = w;
    height = h;
    data = new MapEntry[w * h];
}

const MapEntry &
DrillMap::get(isize w, isize h) const
{
    assert(data != nullptr && w < width && h < height);
    return data[h * width + w];
}

void
DrillMap::set(isize w, isize h, const MapEntry &entry)
{
    assert(data != nullptr && w < width && h < height);
    data[h * width + w] = entry;
}

void
DrillMap::save(const string &path)
{
    printf("DrillMap::save(%s)\n", path.c_str());
    
    std::ofstream os(path.c_str(), std::ios::binary);
    if (!os.is_open()) throw Exception("Failed to write file " + path);
    save(os);
}

void
DrillMap::save(std::ostream &os)
{
    // Write header
    os.write((char *)&width, sizeof(width));
    os.write((char *)&height, sizeof(height));
    os.write((char *)&logBailout, sizeof(logBailout));

    printf("save: width = %zd height = %zd\n", width, height);

    // Write data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            
            auto item = get(x,y);
            os.write((char *)&item.iteration, sizeof(item.iteration));
            os.write((char *)&item.lognorm, sizeof(item.lognorm));
        }
    }
}

}
