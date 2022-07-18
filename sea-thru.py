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
import queue

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

def estimate_channel_backscatter(points, depths, channel, attempts = 50):

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
        Veil = best_coeffs[0], backscatter = best_coeffs[1], recover = best_coeffs[2], attenuation = best_coeffs[3]))

    Bc_channel = predict_backscatter(depths, *best_coeffs)

    return Bc_channel

def estimate_backscatter(image, depths):
    points = find_reference_points(image, depths)
    backscatter_channels = []
    for channel in range(3):
        backscatter_channels.append(estimate_channel_backscatter(points, depths, channel))

    Bc = np.stack(backscatter_channels, axis = 2)

    return Bc

# Wideband attenuation

def predict_wideband_attenuation(depths, a, b, c, d):
    return a * np.exp(b * depths) + c * np.exp(d * depths)

def predict_z(Ec, depths, a, b, c, d):
    return -np.log(Ec) / (a * np.exp(b * depths) + c * np.exp(d * depths))

def estimate_wideband_attenuation(D, depths):
    """
    Args:
        Dc: direct signal
    """
    Ea = compute_illuminant_map(D, depths)

    att_channels = []

    for channel in range(3):
        att_channels.append(refine_attenuation_estimation(Ea[:, :, channel], depths, channel))
    
    att = np.stack(att_channels, axis = 2)
    
    return att

def refine_attenuation_estimation(Ec, depths, channel, attempts = 10):
    """
    Ec is illuminant map of only 1 channel
    """
    # Curve fitting
    lo = np.array([-5, -10, 0, -10])
    hi = np.array([50, 1, 50, 1])

    best_loss = np.inf
    best_coeffs = []

    for _ in range(attempts):
        popt, pcov = scipy.optimize.curve_fit(lambda Ec, a, b, c, d: predict_z(Ec, depths, a, b, c, d),
                                              Ec, depths, 
                                              np.random.random(4) * (hi - lo) + lo, bounds = (lo, hi))
        cur_loss = np.square(predict_z(Ec, depths, *popt) - depths)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_coeffs = popt

    print("Found coeffs for channel {channel} with mse {mse}".format(channel = channel, mse = best_loss))
    print("a = {a}, b = {b}, c = {c}, d = {d}".format(
        a = best_coeffs[0], b = best_coeffs[1], c = best_coeffs[2], d = best_coeffs[3]))

    att_channel = predict_wideband_attenuation(depths, *best_coeffs)

    return att_channel

def compute_illuminant_map(Dc, depths, iterations = 100, p = 0.7, f = 2, eps = 0.01):
    neighborhood_maps = compute_neighborhood(depths)

    ac = np.zeros_like(Dc)
    ac_p = ac.copy()
    ac_new = ac.copy()

    xlim, ylim, _ = Dc.shape

    for _ in range(iterations):
        for x in range(xlim):
            for y in range(ylim):
                idcs = neighborhood_maps[x][y]
                ac_p[x, y] = np.sum(ac[idcs]) / len(idcs)
        ac_new = Dc * p + ac_p * (1 - p)
        if np.max(np.abs(ac - ac_new)) < eps:
            break
        ac = ac_new

    return ac * f

def compute_neighborhood(depths, epsilon = 0.03):
    """
    This could be very expensive...
    """
    flags = np.zeros_like(depths, dtype = np.intc)
    
    xlim, ylim = depths.shape

    neighborhood_maps = []
    for x in range(xlim):
        row = []
        for y in range(ylim):
            row.append(find_neighborhood(depths, x, y))
        neighborhood_maps.append(row)

def find_neighborhood(depths, x, y, epsilon):
    flags = np.zeros_like(depths, dtype = np.intc)
    xlim, ylim = depths.shape

    z = depths.copy()
    z = np.abs(z - z[x, y])

    q = queue.Queue()
    q.put((x, y))

    ret = []

    while q.not_empty():
        cur_x, cur_y = q.get()
        if flags[cur_x, cur_y]:
            continue
        if z[cur_x, cur_y] < epsilon:
            ret.append((cur_x, cur_y))
            flags[cur_x, cur_y] = 1
            if cur_x > 0:
                q.put((cur_x - 1, cur_y))
            if cur_y > 0:
                q.put((cur_x, cur_y - 1))
            if cur_x < xlim - 1: 
                q.put((cur_x + 1, cur_y))
            if cur_y < ylim - 1:
                q.put((cur_x, cur_y + 1))

    return ret

# Whitebalance

# Whitebalancing with 
# Conversion functions courtesy of https://stackoverflow.com/a/34913974/2721685
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1,2]] += 128
    return ycbcr #np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def wb_ycbcr_mean(data):
    # Convert data and sample to YCbCr
    ycbcr = rgb2ycbcr(data)

    # Calculate mean components
    yc = list(np.mean(ycbcr[:, :, i]) for i in range(3))

    # Center cb and cr components of image based on sample
    for i in range(1,3):
        ycbcr[:, :, i] = np.clip(ycbcr[:, :, i] + (128 - yc[i]), 0, 255)
    
    return ycbcr2rgb(ycbcr)

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

    Ba = estimate_backscatter(original, depths)

    Da = original - Ba

    att = estimate_wideband_attenuation(Da, depths)

    Ja = Da * np.exp(att)

