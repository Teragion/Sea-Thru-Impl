from sea_thru import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', required = True, help = "Path to original image")
    parser.add_argument('--depth', required = True, help = "Path to depth map")
    parser.add_argument('--mode', required = True, help = "Mode = {{Map, Predict, Hybrid}}")
    parser.add_argument('--prefix', required = True, help = "Prefix for output files")

    args = parser.parse_args()

    mode = args.mode
    image_path = args.original
    depths_path = args.depth

    original = read_image(image_path, 2048)

    if mode == "Map":
        # Using given depth map
        depths = read_depthmap(depths_path, (original.shape[1], original.shape[0]))
        depths = normalize_depth_map(depths, 0.1, 10.0)

    elif mode == "Predict":
        # Predicting depth using MiDaS
        print("Predicting depth using MiDaS")
        depths = run_midas(image_path, "out/", "weights/dpt_large-midas-2f21e586.pt", "dpt_large")
        # depths = run_midas(image_path, "out/", "weights/dpt_hybrid-midas-501f0c75.pt", "dpt_hybrid")
        depths = cv2.resize(depths, dsize = (original.shape[1], original.shape[0]), interpolation = cv2.INTER_CUBIC)
        # depths = np.square(depths) # More contrast!
        depths = np.max(depths) / depths # disparity map to depth map
        print(depths)

        print("Preprocessing monocular depths esimation with hint")    
        hint_depths = read_depthmap(depths_path, (original.shape[1], original.shape[0]))
        depths = refine_depths_from_hint(depths, np.mean(hint_depths))

    elif mode == "Hybrid":
        print("Loading user input depth map")
        depths = read_depthmap(depths_path, (original.shape[1], original.shape[0]))
        print("Predicting depth using MiDaS")
        pdepths = run_midas(image_path, "out/", "weights/dpt_large-midas-2f21e586.pt", "dpt_large")
        pdepths = cv2.resize(pdepths, dsize = (original.shape[1], original.shape[0]), interpolation = cv2.INTER_CUBIC)
        pdepths = np.max(pdepths) / pdepths # disparity map to depth map
        print("Combining depth maps")
        depths = combine_map_predict(depths, pdepths)

    matplotlib.use('TkAgg')

    fig1, ax2 = plt.subplots()
    x = np.arange(depths.shape[1])
    y = np.flip(np.arange(depths.shape[0]))
    X, Y = np.meshgrid(x, y)
    CS = ax2.contourf(X, Y, depths, 20, cmap=plt.cm.plasma)
    ax2.axis('equal')
    ax2.axis('off')
    plt.title(mode)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig1.colorbar(CS, cax = cax)

    plt.savefig("out/depths_" + args.prefix + mode + ".png", bbox_inches='tight', dpi=300)
    plt.show()