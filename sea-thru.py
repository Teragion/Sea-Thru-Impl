# 
# This file is part of Sea-Thru-Impl.
# Copyright (c) 2022 Zeyuan HE (Teragion).
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# 

import argparse

import numpy as np
import math

import sklearn as sk
import scipy
import scipy.optimize
import scipy.stats

from PIL import Image
import rawpy
import matplotlib
from matplotlib import pyplot as plt
from skimage import exposure

matplotlib.use('TkAgg')

NUM_BINS = 10 # number of bins of depths to find backscatter

# File operations
def read_image(image_path, max_side = 1024):
    image_file_raw = rawpy.imread(image_path).postprocess()
    image_file = Image.fromarray(image_file_raw)
    image_file.thumbnail((max_side, max_side), Image.ANTIALIAS)
    return np.float32(image_file)

def read_depthmap(depthmap_path, max_side = 1024):
    depth_file = Image.open(depthmap_path)
    depth_file.thumbnail((max_side, max_side), Image.ANTIALIAS)
    return np.array(depth_file)

# Preprocessing
def normalize_depth_map(depths, z_min, z_inf):
    """
    Normalize values in the depth map
    """
    z_max = max(np.max(depths), z_inf)
    depths[depths == 0] = z_max
    depths[depths < z_min] = z_min
    return depths

def estimate_far(image):
    """
    Estimates the farthest distance from the image color    
    """
    return None

# Backscatter

# Eq. 10
def predict_backscatter(z, veil, backscatter, recover, attenuation):
    return (veil * (1 - np.exp(-backscatter * z)) + recover * np.exp(-attenuation * z))

def find_reference_points(image, depths, frac=0.01):
    z_max = np.max(depths)
    z_min = np.min(depths)
    z_bins = np.linspace(z_min, z_max, NUM_BINS + 1)
    rgb_norm = np.linalg.norm(image, axis=2) # is using 2-norm correct here?
    ret = []
    for i in range(NUM_BINS):
        lo, hi = z_bins[i], z_bins[i + 1]
        indices = np.where(np.logical_and(depths >= lo, depths < hi))
        bin_rgb_norm, bin_z, bin_color = rgb_norm[indices], depths[indices], image[indices]
        points_sorted = sorted(zip(bin_rgb_norm, bin_z, bin_color[0], bin_color[1], bin_color[2]), key = lambda p : p[0])
        for j in range(len(points_sorted) * frac):
            ret.append(points_sorted[j])
    return np.asarray(ret)

def estimate_channel_backscatter(points, channel, attempts = 50):

    lo = np.array([0, 0, 0, 0])
    hi = np.array([1, 5, 1, 5])

    best_loss = np.inf
    best_coeffs = []

    for _ in range(attempts):
        popt, pcov = scipy.optimize.curve_fit(predict_backscatter, points[:, 1], points[:, channel + 2], 
                                              np.random.random(4) * (hi - lo) + lo, bounds = (lo, hi))
        cur_loss = np.square(predict_backscatter(points[:, 1], *popt) - points[:, channel + 2])
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_coeffs = popt

    print("Found coeffs for channel {channel} with mse {mse}".format(channel = channel, mse = best_loss))
    print("Veil = {Veil}, backscatter = {backscatter}, recover = {recover}, attenuation = {attenuation}".format(
        Veil = popt[0], backscatter = popt[1], recover = popt[2], attenuation = popt[3]))

    return best_coeffs

def estimate_backscatter(image, depths):
    points = find_reference_points(image, depths)
    backscatter_coeffs = []
    for channel in range(3):
        backscatter_coeffs.append(estimate_channel_backscatter(points, channel))

    return backscatter_coeffs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', required = True, help = "Path to original image")
    parser.add_argument('--depth', required = False, help = "Path to depth map")
    parser.add_argument('--mode', required = True, help = "Mode = {{Map, Predict}}")

    args = parser.parse_args()

    if args.mode == "Map":
        # Using given depth map
        print("Using user input depth map")
        original = read_image(args.original)
        depths = read_depthmap(args.depth)
        depths = normalize_depth_map(depths)
    else:
        # Predicting depth using monodepth2
        print("Predicting depth using monodepth2")

