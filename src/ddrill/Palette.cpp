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
#include "Palette.h"
#include "DrillOptions.h"

namespace dd {

Palette::Palette(const DrillOptions &options) : opt(options)
{
    init(opt.palette.values);    
}

void
Palette::init(string values)
{
    // Revert to the default palette if none is given
    if (values == "") values = defaultPalette;
    
    std::stringstream stream(values);
    u32 value;

    while(stream >> value) {
        colors.push_back(value);
    }
}

u32
Palette::colorize(isize depth) const
{
    return colors[depth % colors.size()];
}

const string
Palette::defaultPalette = "2687244 2621711 2621714 2621717 2621977 2621980 2621983 2622243 2622246 2622249 2622252 2622512 2622515 2622518 2622778 2622781 2622784 2622787 2623047 2623050 2623053 2623313 2557780 2557783 2557786 2558046 2558049 2558052 2558312 2558315 2558318 2558321 2558581 2558584 2558587 2558847 2558850 2558853 2558856 2559116 2559119 2559122 2559382 2493849 2493852 2493855 2494115 2494118 2494121 2494381 2494384 2494387 2494390 2494650 2494653 2494656 2494916 2494919 2494922 2494925 2495185 2495188 2495191 2495451 2430427 2430940 2431452 2432221 2432733 2433246 2434014 2434527 2435039 2435808 2436320 2436833 2437601 2438114 2438626 2373859 2374371 2374884 2375652 2376165 2376677 2377190 2377958 2378471 2378983 2379752 2380264 2380777 2381545 2382058 2382570 2317803 2318315 2318828 2319596 2320109 2320621 2321390 2321902 2322415 2323183 2323696 2324208 2324721 2325489 2326002 2326514 2261747 2262259 2262772 2263540 2264053 2264565 2265334 2265846 2266359 2267127 2267640 2268152 2268921 2269433 2269946 2270715 2269946 2269433 2268921 2268152 2267640 2267127 2266359 2265846 2265334 2264565 2264053 2263540 2262772 2262259 2261747 2326514 2326002 2325489 2324721 2324208 2323696 2323183 2322415 2321902 2321390 2320621 2320109 2319596 2318828 2318315 2317803 2382570 2382058 2381545 2380777 2380264 2379752 2378983 2378471 2377958 2377190 2376677 2376165 2375652 2374884 2374371 2373859 2438626 2438114 2437601 2436833 2436320 2435808 2435039 2434527 2434014 2433246 2432733 2432221 2431452 2430940 2430427 2495451 2495191 2495188 2495185 2494925 2494922 2494919 2494916 2494656 2494653 2494650 2494390 2494387 2494384 2494381 2494121 2494118 2494115 2493855 2493852 2493849 2559382 2559122 2559119 2559116 2558856 2558853 2558850 2558847 2558587 2558584 2558581 2558321 2558318 2558315 2558312 2558052 2558049 2558046 2557786 2557783 2557780 2623313 2623053 2623050 2623047 2622787 2622784 2622781 2622778 2622518 2622515 2622512 2622252 2622249 2622246 2622243 2621983 2621980 2621977 2621717 2621714 2621711 2621711 2621711 2621711 2621711 2621711 2621711 2621711 2621714 2621717 2621977 2621980 2621983 2622243 2622246 2622249 2622252 2622512 2622515 2622518 2622778 2622781 2622784 2622787 2623047 2623050 2623053 2623313 2557780 2557783 2557786 2558046 2558049 2558052 2558312 2558315 2558318 2558321 2558581 2558584 2558587 2558847 2558850 2558853 2558856 2559116 2559119 2559122 2559382 2493849 2493852 2493855 2494115 2494118 2494121 2494381 2494384 2494387 2494390 2494650 2494653 2494656 2494916 2494919 2494922 2494925 2495185 2495188 2495191 2495451 2430427 2430940 2431452 2432221 2432733 2433246 2434014 2434527 2435039 2435808 2436320 2436833 2437601 2438114 2438626 2373859 2374371 2374884 2375652 2376165 2376677 2377190 2377958 2378471 2378983 2379752 2380264 2380777 2381545 2382058 2382570 2317803 2318315 2318828 2319596 2320109 2320621 2321390 2321902 2322415 2323183 2323696 2324208 2324721 2325489 2326002 2326514 2261747 2262259 2262772 2263540 2264053 2264565 2265334 2265846 2266359 2267127 2267640 2268152 2268921 2269433 2269946 2270715 2269946 2269433 2268921 2268152 2267640 2267127 2266359 2265846 2265334 2264565 2264053 2263540 2262772 2262259 2261747 2326514 2326002 2325489 2324721 2324208 2323696 2323183 2322415 2321902 2321390 2320621 2320109 2319596 2318828 2318315 2317803 2382570 2382058 2381545 2380777 2380264 2379752 2378983 2378471 2377958 2377190 2376677 2376165 2375652 2374884 2374371 2373859 2438626 2438114 2437601 2436833 2436320 2435808 2435039 2434527 2434014 2433246 2432733 2432221 2431452 2430940 2430427 2495451 2495191 2495188 2495185 2494925 2494922 2494919 2494916 2494656 2494653 2494650 2494390 2494387 2494384 2494381 2494121 2494118 2494115 2493855 2493852 2493849 2559382 2559122 2559119 2559116 2558856 2558853 2558850 2558847 2558587 2558584 2558581 2558321 2558318 2558315 2558312 2558052 2558049 2558046 2557786 2557783 2557780 2623313 2623053 2623050 2623047 2622787 2622784 2622781 2622778 2622518 2622515 2622512 2622252 2622249 2622246 2622243 2621983 2621980 2621977 2621717 2621714 2621711 2687244";

}
