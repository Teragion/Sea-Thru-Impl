//
// This file is part of Sea-Thru-Impl.
// Copyright (c) 2022 Zeyuan HE (Teragion).
//
// This program is free software: you can redistribute it and/or modify  
// it under the terms of the GNU General Public License as published by  
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but 
// WITHOUT ANY WARRANTY; without even the implied warranty of 
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License 
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//

#include <queue>
#include <set>
#include <utility>
#include <vector>

#include <iostream>

#include <omp.h>

static int xlim;
static int ylim;

static bool precomputed = false;
static std::vector<int> neighborhoods;
static int color = 1;

inline int ind(int x, int y) {
    return x * ylim + y;
}

void find_neighborhood(double* depths, int x, int y, double eps, int color) {
    std::queue<std::pair<int, int> > q;

    q.push(std::make_pair(x, y));
    neighborhoods[ind(x, y)] = color;

    while (!q.empty()) {
        auto cur = q.front();
        q.pop();
        
        double z = depths[ind(cur.first, cur.second)];

        if (cur.first > 0) {
            std::pair<int, int> next{cur.first - 1, cur.second};
            if (neighborhoods[ind(next.first, next.second)] == 0 && std::abs(depths[ind(next.first, next.second)] - z) < eps) {
                neighborhoods[ind(next.first, next.second)] = color;
                q.push(next);
            }
        }
        if (cur.second > 0) {
            std::pair<int, int> next{cur.first, cur.second - 1};
            if (neighborhoods[ind(next.first, next.second)] == 0 && std::abs(depths[ind(next.first, next.second)] - z) < eps) {
                neighborhoods[ind(next.first, next.second)] = color;
                q.push(next);
            }
        }
        if (cur.first < xlim - 1) {
            std::pair<int, int> next{cur.first + 1, cur.second};
            if (neighborhoods[ind(next.first, next.second)] == 0 && std::abs(depths[ind(next.first, next.second)] - z) < eps) {
                neighborhoods[ind(next.first, next.second)] = color;
                q.push(next);
            }
        }
        if (cur.second < ylim - 1) {
            std::pair<int, int> next{cur.first, cur.second + 1};
            if (neighborhoods[ind(next.first, next.second)] == 0 && std::abs(depths[ind(next.first, next.second)] - z) < eps) {
                neighborhoods[ind(next.first, next.second)] = color;
                q.push(next);
            }
        }
    }
}

extern "C" {

void compute_illuminant_map(double* Dc, double* depths, double* illu, double p, double f, double eps, int xlim_in, int ylim_in, int iterations) {
    xlim = xlim_in;
    ylim = ylim_in;

    // Compute and store neighborhood map

    std::vector<double> ac(xlim * ylim, 0.0);
    std::vector<double> ac_p(xlim * ylim, 0.0);
    std::vector<double> ac_new(xlim * ylim, 0.0);

    if (!precomputed) {
        std::cout << "Computing neighborhood map." << std::endl;
        neighborhoods = std::vector<int>(xlim * ylim, 0);

        for (int x = 0; x < xlim; x++) {
            // std::cout << x << std::endl;
            for (int y = 0; y < ylim; y++) {
                if (neighborhoods[ind(x, y)] == 0) {
                    find_neighborhood(depths, x, y, eps, color++);
                }
            }
        }

        precomputed = true;

        std::cout << "Neighborhoods: " << color << std::endl;
    }

    std::cout << "Computing illuminant." << std::endl;

    for (int k = 0; k < iterations; k++) {
        // std::cout << k << std::endl;
        std::vector<double> color_sums(color, 0.0);
        std::vector<int> color_count(color, 0);
        for (int x = 0; x < xlim; x++) {
            for (int y = 0; y < ylim; y++) {
                color_sums[neighborhoods[ind(x, y)]] += ac[ind(x, y)];
                color_count[neighborhoods[ind(x, y)]] += 1;
            }
        }
#pragma omp parallel for
        for (int x = 0; x < xlim; x++) {
            for (int y = 0; y < ylim; y++) {
                ac_p[ind(x, y)] = color_sums[neighborhoods[ind(x, y)]] / color_count[neighborhoods[ind(x, y)]];
                ac_new[ind(x, y)] = Dc[ind(x, y)] * p + ac_p[ind(x, y)] * (1 - p);
            }
        }
        ac.swap(ac_new);
    }

    for (int x = 0; x < xlim; x++) {
        for (int y = 0; y < ylim; y++) {
            illu[ind(x, y)] = ac[ind(x, y)] * f;
        }
    }
}

}