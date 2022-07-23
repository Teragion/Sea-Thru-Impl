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
import ctypes
import sys

import numpy as np
import math
import queue

import sklearn as sk
import scipy
import scipy.optimize
import scipy.stats

import cv2
from PIL import Image
import rawpy
from skimage import exposure
import matplotlib

from midas_helper import run_midas

lib = np.ctypeslib.load_library('sillu','.')

matplotlib.use('TkAgg')

NUM_BINS = 10 # number of bins of depths to find backscatter

# File operations
def read_image(image_path, max_side = 3840):
    image_file_raw = rawpy.imread(image_path).postprocess()
    image_file = Image.fromarray(image_file_raw)
    image_file.thumbnail((max_side, max_side), Image.ANTIALIAS)
    return np.float64(image_file) / 255.0

def read_depthmap(depthmap_path, size):
    depth_file = Image.open(depthmap_path)
    depths = depth_file.resize(size, Image.ANTIALIAS)
    return np.float64(depths)

# Preprocessing
def normalize_depth_map(depths, z_min, z_inf):
    """
    Normalize values in the depth map
    """
    z_max = max(np.max(depths), z_inf)
    depths[depths == 0] = z_max
    depths[depths < z_min] = z_min
    return depths

def estimate_far(image, frac = 0.2, close = 0.3):
    """
    Estimates the farthest distance from the image color    
    """
    # Chosen luminance formula
    r = image[:, :, 0] * 0.2126
    g = image[:, :, 1] * 0.7152
    b = image[:, :, 2] * 0.0722

    lum = np.sum(np.stack([r, g, b], axis = 2), axis = 2)
    lum.sort(axis = 0)
    print(lum)

    rows = int(frac * lum.shape[0])

    darkest = np.mean(lum[rows:(2 * rows), :], axis = 0)
    brightest = np.mean(lum[-(2 * rows):(-rows), :], axis = 0)

    ratio = np.mean(brightest / darkest)

    return np.log2(ratio) * 10

def refine_depths_from_hint(depths, avg):
    oavg = np.mean(depths)
    depths = depths * (avg / oavg)
    print("Estimated farthest distance is {far}".format(far = np.max(depths)))
    print("Estimated mean distance is {mean}".format(mean = np.mean(depths)))
    return depths

def preprocess_predicted_depths(original, depths):
    far = estimate_far(original)
    print("Estimated farthest distance is {far}".format(far = far))
    ratio = far / np.max(depths)
    print("Estimated mean distance is {mean}".format(mean = np.mean(depths * ratio)))
    return depths * ratio

# Backscatter

# Eq. 10
def predict_backscatter(z, veil, backscatter, recover, attenuation):
    return (veil * (1 - np.exp(-backscatter * z)) + recover * np.exp(-attenuation * z))

def find_reference_points(image, depths, frac = 0.02):
    z_max = np.max(depths)
    z_min = np.min(depths)
    z_bins = np.linspace(z_min, z_max, NUM_BINS + 1)
    rgb_norm = np.linalg.norm(image, axis=2) # is using 2-norm correct here?
    ret = []
    for i in range(NUM_BINS):
        lo, hi = z_bins[i], z_bins[i + 1]
        indices = np.where(np.logical_and(depths >= lo, depths < hi))
        if indices[0].size == 0:
            continue
        bin_rgb_norm, bin_z, bin_color = rgb_norm[indices], depths[indices], image[indices]
        points_sorted = sorted(zip(bin_rgb_norm, bin_z, bin_color[:,0], bin_color[:,1], bin_color[:,2]), key = lambda p : p[0])
        for j in range(math.ceil(len(points_sorted) * frac)):
            ret.append(points_sorted[j])
    return np.asarray(ret)

def estimate_channel_backscatter(points, depths, channel, attempts = 50):

    lo = np.array([0, 0, 0, 0])
    hi = np.array([1, 5, 1, 5])

    best_loss = np.inf
    best_coeffs = []

    for _ in range(attempts):
        try:
            popt, pcov = scipy.optimize.curve_fit(predict_backscatter, points[:, 1], points[:, channel + 2], 
                                                np.random.random(4) * (hi - lo) + lo, bounds = (lo, hi))
            cur_loss = np.mean(np.square(predict_backscatter(points[:, 1], *popt) - points[:, channel + 2]))
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_coeffs = popt
        except RuntimeError as re:
            print(re, file=sys.stderr)

    print("Found coeffs for channel {channel} with mse {mse}".format(channel = channel, mse = best_loss))
    print("Veil = {Veil}, backscatter = {backscatter}, recover = {recover}, attenuation = {attenuation}".format(
        Veil = best_coeffs[0], backscatter = best_coeffs[1], recover = best_coeffs[2], attenuation = best_coeffs[3]))

    Bc_channel = predict_backscatter(depths, *best_coeffs)

    return Bc_channel, best_coeffs

def estimate_backscatter(image, depths):
    points = find_reference_points(image, depths)
    backscatter_channels = []
    backscatter_coeffs = []
    for channel in range(3):
        Bc, coeffs = estimate_channel_backscatter(points, depths, channel)
        backscatter_channels.append(Bc)
        backscatter_coeffs.append(coeffs)

    Ba = np.stack(backscatter_channels, axis = 2)

    return Ba, backscatter_coeffs

# Wideband attenuation

def predict_wideband_attenuation(depths, a, b, c, d):
    return a * np.exp(b * depths) + c * np.exp(d * depths)

def predict_z(x, a, b, c, d):
    Ec, depth = x
    return -np.log(Ec) / (a * np.exp(b * depth) + c * np.exp(d * depth))

def estimate_wideband_attenuation(D, depths):
    """
    Args:
        Dc: direct signal
    """
    # Ea = compute_illuminant_map_plugin(D, depths, p = 0.6, f = 2.0, eps = 0.03)
    Ea = compute_illuminant_map_plugin(D, depths, p = 0.6, f = 2.0, eps = np.mean(depths) / 200.0)
    Ea = np.clip(Ea, 0, None)

    att_channels = []

    for channel in range(3):
        att_channels.append(refine_attenuation_estimation(Ea[:, :, channel], depths, channel))
    
    att = np.stack(att_channels, axis = 2)
    
    return att

def refine_attenuation_estimation(Ec, depths, channel, attempts = 5):
    """
    Ec is illuminant map of only 1 channel
    """
    # Curve fitting
    lo = np.array([-10, -10, -10, -10])
    hi = np.array([100, 1, 100, 1])

    best_loss = np.inf
    best_coeffs = []

    original_shape = depths.shape

    # print(Ec)
    # print(depths)

    Ec.reshape(-1)
    depths.reshape(-1)

    locs = np.where(Ec > 1E-5)
    E = Ec[locs]
    z = depths[locs]

    for _ in range(attempts):
        try:
            popt, pcov = scipy.optimize.curve_fit(predict_z,
                                                (E, z), z, 
                                                np.random.random(4) * (hi - lo) + lo, bounds = (lo, np.inf * np.ones(4)))
            cur_loss = np.mean(np.square(predict_z((E, z), *popt) - z))
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_coeffs = popt
        except RuntimeError as re:
            print(re, file=sys.stderr)

    print("Found coeffs for channel {channel} with mse {mse}".format(channel = channel, mse = best_loss))
    print("a = {a}, b = {b}, c = {c}, d = {d}".format(
        a = best_coeffs[0], b = best_coeffs[1], c = best_coeffs[2], d = best_coeffs[3]))

    depths.reshape(original_shape)
    att_channel = predict_wideband_attenuation(depths, *best_coeffs)

    return att_channel

def compute_illuminant_map_plugin(D, depths, iterations = 100, p = 0.7, f = 2, eps = 0.2):
    """
    Calls C interface for computing illuminant map and returns LSAC result
    """
    func = lib.compute_illuminant_map

    func.restype = None
    func.argtypes = [np.ctypeslib.ndpointer(float, ndim = 2, flags = 'aligned, contiguous'),
                     np.ctypeslib.ndpointer(float, ndim = 2, flags = 'aligned, contiguous'),
                     np.ctypeslib.ndpointer(float, ndim = 2, flags = 'aligned, contiguous, writeable'),
                     ctypes.c_double,
                     ctypes.c_double,
                     ctypes.c_double,
                     ctypes.c_int, 
                     ctypes.c_int,
                     ctypes.c_int]

    a = []

    z = np.require(depths, float, ['ALIGNED', 'CONTIGUOUS'])

    for channel in range(3):
        Dc = np.ascontiguousarray(D[:, :, channel])
        Dc = np.require(Dc, float, ['ALIGNED', 'CONTIGUOUS'])
        ac = np.zeros_like(Dc)
        ac = np.require(ac, float, ['ALIGNED', 'CONTIGUOUS'])
        x, y = depths.shape
        func(Dc, z, ac, p, f, eps, x, y, iterations, dtype = float)
        a.append(ac)

    return np.stack(a, axis = 2)

def compute_illuminant_map(Dc, depths, iterations = 100, p = 0.5, f = 2, eps = 0.03):
    """
    Computes illuminant map and returns LSAC result, very slow, only provided as a reference implementation
    """
    neighborhood_maps = compute_neighborhood(depths)

    ac = np.zeros_like(Dc)
    ac_p = ac.copy()
    ac_new = ac.copy()

    xlim, ylim, _ = Dc.shape

    for _ in range(iterations):
        for x in range(xlim):
            for y in range(ylim):
                idcs = neighborhood_maps[x][y]
                ac_p[x, y] = np.sum(ac[tuple(idcs)], axis = 0) / len(idcs[0])
        ac_new = Dc * p + ac_p * (1 - p)
        if np.max(np.abs(ac - ac_new)) < eps:
            break
        ac = ac_new

    return ac * f

def compute_neighborhood(depths, epsilon = 0.03):
    """
    This could be very expensive...
    """
    
    xlim, ylim = depths.shape

    neighborhood_maps = []
    for x in range(xlim):
        print("Process: {p}".format(p = x / xlim))
        row = []
        for y in range(ylim):
            row.append(find_neighborhood(depths, x, y, epsilon))
        neighborhood_maps.append(row)
    
    return neighborhood_maps

def find_neighborhood(depths, x, y, epsilon):
    flags = np.zeros_like(depths, dtype = np.intc)
    xlim, ylim = depths.shape

    z = depths.copy()
    z = np.abs(z - z[x, y])

    q = queue.Queue()
    q.put((x, y))

    ret = [[], []]

    while not q.empty():
        cur_x, cur_y = q.get()
        if flags[cur_x, cur_y]:
            continue
        flags[cur_x, cur_y] = 1
        if z[cur_x, cur_y] < epsilon:
            ret[0].append(cur_x)
            ret[1].append(cur_y)
            if cur_x > 0:
                q.put((cur_x - 1, cur_y))
            if cur_y > 0:
                q.put((cur_x, cur_y - 1))
            if cur_x < xlim - 1: 
                q.put((cur_x + 1, cur_y))
            if cur_y < ylim - 1:
                q.put((cur_x, cur_y + 1))

    return ret

def recover(Da, att):
    for c in range(3):
        att[:, :, c] = att[:, :, c] * depths

    Ja = Da * np.exp(att)

    Ja = np.clip(Ja, 0, 1)

    return Ja

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
    rgb = im.astype(np.float64)
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

def grey_world(img):
    dg = 1.0 / np.mean(np.sort(img[:, :, 1], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    db = 1.0 / np.mean(np.sort(img[:, :, 2], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    dsum = dg + db
    dg = dg / dsum * 2.
    db = db / dsum * 2.
    img[:, :, 0] *= (db + dg) / 2
    img[:, :, 1] *= dg
    img[:, :, 2] *= db
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', required = True, help = "Path to original image")
    parser.add_argument('--depth', required = False, help = "Path to depth map")
    parser.add_argument('--mode', required = True, help = "Mode = {{Map, Predict}}")
    parser.add_argument('--hint', required = False, help = "Path to depth map as hint")
    parser.add_argument('--size', required = False, type = int, help = "Maximum side of image to shrink")

    args = parser.parse_args()

    if args.size is not None:
        original = read_image(args.original, args.size)
    else:
        original = read_image(args.original)

    if args.mode == "Map":
        # Using given depth map
        print("Using user input depth map")
        depths = read_depthmap(args.depth, (original.shape[1], original.shape[0]))
        depths = normalize_depth_map(depths, 0.1, 6.0)
    else:
        # Predicting depth using MiDaS
        print("Predicting depth using MiDaS")
        depths = run_midas(args.original, "out/", "weights/dpt_large-midas-2f21e586.pt", "dpt_large")
        # depths = run_midas(args.original, "out/", "weights/dpt_hybrid-midas-501f0c75.pt", "dpt_hybrid")
        depths = cv2.resize(depths, dsize = (original.shape[1], original.shape[0]), interpolation = cv2.INTER_CUBIC)
        depths = np.square(depths) # More contrast!
        depths = np.max(depths) / depths # disparity map to depth map
        print(depths)

        if args.hint is not None:
            print("Preprocessing monocular depths esimation with hint")    
            hint_depths = read_depthmap(args.hint, (original.shape[1], original.shape[0]))
            depths = refine_depths_from_hint(depths, np.mean(hint_depths))
        else:
            print("Preprocessing monocular depths esimation without hint")    
            preprocess_predicted_depths(original, depths)

    print("Loaded image and depth map of size {x} x {y}".format(x = original.shape[0], y = original.shape[1]))

    print("Estimating backscatter...")
    Ba, coeffs = estimate_backscatter(original, depths)

    Da = original - Ba
    Da = np.clip(Da, 0, 1)

    D = np.uint8(Da * 255.0)
    backscatter_removed = Image.fromarray(D)
    backscatter_removed.save("out/direct_signal.png")

    print("Estimating wideband attenuation...")
    att = estimate_wideband_attenuation(Da, depths)

    Ja = recover(Da, att / 2)

    Ja = exposure.equalize_adapthist(Ja)

    Ja *= 255.0
    Js = wb_ycbcr_mean(Ja)

    result = Image.fromarray(Js)
    result.save("out/out.png")
    print("Finished.")
