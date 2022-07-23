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
static std::vector<std::vector<std::pair<int, int> > > maps;

inline int ind(int x, int y) {
    return x * ylim + y;
}

std::vector<std::pair<int, int> > find_neighborhood(double* depths, int x, int y, double eps) {
    std::vector<std::pair<int, int> > ret;

    std::queue<std::pair<int, int> > q;
    std::set<std::pair<int, int> > flags;

    q.push(std::make_pair(x, y));

    double z = depths[ind(x, y)];

    while (!q.empty()) {
        auto cur = q.front();
        q.pop();
        if (flags.find(cur) != flags.end()) {
            continue;
        } else {
            flags.emplace(cur);
            if (std::abs(depths[ind(cur.first, cur.second)] - z) < eps) {
                ret.push_back(cur);
                if (cur.first > 0) {
                    q.push({cur.first - 1, cur.second});
                }
                if (cur.second > 0) {
                    q.push({cur.first, cur.second - 1});
                }
                if (cur.first < xlim - 1) {
                    q.push({cur.first + 1, cur.second});
                }
                if (cur.second < ylim - 1) {
                    q.push({cur.first, cur.second + 1});
                }
            }
        }
    }

    return ret;
}

extern "C" {

void compute_illuminant_map(double* Dc, double* depths, double* illu, double p, double f, double eps, int xlim, int ylim, int iterations) {
    // Compute and store neighborhood map

    std::vector<double> ac(xlim * ylim, 0.0);
    std::vector<double> ac_p(xlim * ylim, 0.0);
    std::vector<double> ac_new(xlim * ylim, 0.0);

    if (!precomputed) {
        std::cout << "Computing neighborhood map." << std::endl;

    #pragma omp parallel for
        for (int x = 0; x < xlim; x++) {
            // std::cout << x << std::endl;
            for (int y = 0; y < ylim; y++) {
    #pragma omp critical
                maps.push_back(find_neighborhood(depths, x, y, eps));
            }
        }

        precomputed = true;

    }

    std::cout << "Computing illuminant." << std::endl;

    for (int k = 0; k < iterations; k++) {
        // std::cout << k << std::endl;
#pragma omp parallel for
        for (int x = 0; x < xlim; x++) {
            for (int y = 0; y < ylim; y++) {
                auto nmap = maps[ind(x, y)];
                double sum = 0.0;
                for (const auto & p : nmap) {
                    sum += ac[ind(p.first, p.second)];
                }
                ac_p[ind(x, y)] = sum /= nmap.size();
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