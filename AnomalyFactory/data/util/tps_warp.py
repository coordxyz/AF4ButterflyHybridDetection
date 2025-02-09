import numpy as np

from .warp_image import warp_images
import pdb, random

# def _get_regular_grid(image, points_per_dim):
#     nrows, ncols = image.shape[0], image.shape[1]
#     rows = np.linspace(0, nrows, points_per_dim)
#     cols = np.linspace(0, ncols, points_per_dim)
#     rows, cols = np.meshgrid(rows, cols)
#     return np.dstack([cols.flat, rows.flat])[0]

#---zy: local region tps
def _get_regular_grid(image, points_per_dim):
    nrows, ncols = image.shape[0], image.shape[1]
    offset = random.randint(0, 80)
    row_start = random.randint(0, nrows-points_per_dim*offset)
    row_end = min(nrows, row_start+points_per_dim*offset)
    col_start = random.randint(0, ncols-points_per_dim*offset)
    col_end = min(ncols, col_start+points_per_dim*offset)
    rows = np.linspace(row_start, row_end, points_per_dim)
    cols = np.linspace(col_start, col_end, points_per_dim)
    rows, cols = np.meshgrid(rows, cols)
    return np.dstack([cols.flat, rows.flat])[0]

def _get_regular_grid3d(image, points_per_dim):
    ndepth = image.shape[0] #depth的阶段也统一设为256
    nrows, ncols = image.shape[0], image.shape[1]
    offset = random.randint(0, 80)
    row_start = random.randint(0, nrows-points_per_dim*offset)
    row_end = min(nrows, row_start+points_per_dim*offset)
    col_start = random.randint(0, ncols-points_per_dim*offset)
    col_end = min(ncols, col_start+points_per_dim*offset)
    depth_start = random.randint(0, ndepth-points_per_dim*offset)
    depth_end = min(ndepth, depth_start+points_per_dim*offset)
    rows = np.linspace(row_start, row_end, points_per_dim)
    cols = np.linspace(col_start, col_end, points_per_dim)
    depths = np.linspace(depth_start, depth_end, points_per_dim)
    rows, cols, depths = np.meshgrid(rows, cols, depths)
    return np.dstack([cols.flat, rows.flat, depths.flat])[0]

def _generate_random_vectors(image, src_points, scale):
    dst_pts = src_points + np.random.uniform(-scale, scale, src_points.shape)
    return dst_pts


def _thin_plate_spline_warp(image, src_points, dst_points, keep_corners=True):
    width, height = image.shape[:2]
    if keep_corners:
        corner_points = np.array(
            [[0, 0], [0, width], [height, 0], [height, width]])
        src_points = np.concatenate((src_points, corner_points))
        dst_points = np.concatenate((dst_points, corner_points))

    out = warp_images(src_points, dst_points,
                      np.moveaxis(image, 2, 0),
                      (0, 0, width - 1, height - 1))

    return np.moveaxis(np.array(out), 0, 2)

def _thin_plate_spline_warp3d(image, src_points, dst_points, keep_corners=True):
    width, height = image.shape[:2]
    depth = image.shape[0]
    if keep_corners:
        corner_points = np.array(
            [[0, 0, 0], [0, width,0], [height, 0, 0], [height, width, 0],
            [0, 0, depth], [0, width,depth], [height, 0, depth],
            [height, width, depth]])
        src_points = np.concatenate((src_points, corner_points))
        dst_points = np.concatenate((dst_points, corner_points))
    #尝试：按3d计算tps，但只用x，y对图像warp，看看跟2d有无优势
    out = warp_images3d(src_points, dst_points,
                      np.moveaxis(image, 2, 0),
                      (0, 0, width - 1, height - 1))

    return np.moveaxis(np.array(out), 0, 2)

def tps_warp(image, points_per_dim, scale):
    width, height = image.shape[:2]
    src = _get_regular_grid(image, points_per_dim=points_per_dim)
    dst = _generate_random_vectors(image, src, scale=scale*width)
    out = _thin_plate_spline_warp(image, src, dst)
    return out

def tps_warp3d(image, points_per_dim, scale):
    width, height = image.shape[:2]
    src3d = _get_regular_grid3d(image, points_per_dim=points_per_dim)
    dst3d = _generate_random_vectors(image, src3d, scale=scale*width)
    out = _thin_plate_spline_warp3d(image, src3d, dst3d)
    return out

def tps_warp_2(image, dst, src):
    out = _thin_plate_spline_warp(image, src, dst)
    return out