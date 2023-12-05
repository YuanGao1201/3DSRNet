import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import label
from skimage.morphology import binary_opening
import skimage.morphology as sm
import skimage
import time
from contextlib import contextmanager
import cv2
import os


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def sparse_df_to_arr(arr_expected_shape=None, sparse_df=None, fill_bool=True):
    """
    :param arr_expected_shape: arr's expected shape
    :param sparse_df:
    :return: expected_arr
    """
    expected_arr = np.zeros(arr_expected_shape)
    point_index = sparse_df['z'].values, sparse_df['x'].values, sparse_df['y'].values
    if fill_bool:
        expected_arr[point_index] = 1
    else:
        expected_arr[point_index] = sparse_df['c']

    del point_index
    del sparse_df
    # gc.collect()
    return expected_arr


def arr_to_sparse_df_only(binary_arr=None):
    index = binary_arr.nonzero()
    sparse_df = pd.DataFrame({'x': index[1],
                              'y': index[2],
                              'z': index[0],
                              'v': binary_arr[index]})
    del index
    return sparse_df


def sparse_df_remove_min(sparse_df=None, threshold_min=5000):
    cluster_df = sparse_df.groupby('c').agg({'c': ['count']})
    cluster_df.columns = ['%s.%s' % e for e in cluster_df.columns.tolist()]
    cluster_df.reset_index(inplace=True)
    cluster_df.rename(columns={'index': 'c'})
    cluster_df = cluster_df[cluster_df['c.count'] > threshold_min]
    return sparse_df[sparse_df['c'].isin(cluster_df['c'].values)]


def arr_to_sparse_df(label_arr=None, add_pixel=False, pixel_arr=None, sort=False, sort_key='c.count',
                     keep_by_top=False, top_nth=30, keep_by_threshold=False, threshold_min=2000, allow_debug=False):
    index = label_arr.nonzero()

    sparse_df = pd.DataFrame({'x': index[1],
                              'y': index[2],
                              'z': index[0],
                              'c': label_arr[index]})
    if add_pixel:
        sparse_df['v'] = pixel_arr[index]
    cluster_df = sparse_df.groupby('c').agg({'x': ['mean', 'min', 'max'],
                                             'y': ['mean', 'min', 'max'],
                                             'z': ['mean', 'min', 'max'],
                                             'c': ['count']})
    cluster_df.columns = ['%s.%s' % e for e in cluster_df.columns.tolist()]

    if sort:
        cluster_df.sort_values(sort_key, ascending=False, inplace=True)
    cluster_df.reset_index(inplace=True)
    cluster_df.rename(columns={'index': 'c'})

    if allow_debug:
        print(label_arr[index])

    if keep_by_top:
        cluster_df = cluster_df.head(top_nth)

    if keep_by_threshold:
        cluster_df = cluster_df[cluster_df['c.count'] > threshold_min]

    sparse_df = sparse_df[sparse_df['c'].isin(cluster_df['c'])]

    del index
    return sparse_df, cluster_df


"""
morphology functions.
"""


def source_hu_value_arr_to_binary(value_arr, use_cv=False, hu_threshold=400):
    binary_arr = value_arr.copy()
    binary_arr[binary_arr < hu_threshold] = 0
    binary_arr[binary_arr >= hu_threshold] = 1
    return binary_arr


def morphology_label(binary_arr=None, use_cv=False, transfer_to_binary=False, hu_thresholds=400):
    """
    :param binary_arr:
    :param use_cv:
    :return:
    """
    if use_cv:
        return None
    return skimage.measure.label(binary_arr, connectivity=2)


def morphology_binary_opening(binary_arr, use_cv=False, transfer_to_binary=False, selem=None):
    """
    :param binary_arr: binary_arr: binary array waited be opened.
    :param use_cv: if True, use open-cv, else use scipy.ndimage or skimage
    :param transfer_to_binary: todo
    :param selem: kernel
    :return:
    """
    if use_cv:
        return None
    else:
        return skimage.morphology.binary_opening(binary_arr, selem)


def loop_morphology_binary_opening(binary_arr, use_cv=False, allow_debug=False, opening_times=None):
    """
    :param binary_arr: binary array waited be opened.
    :param use_cv: if True, use open-cv, else use scipy.ndimage or skimage
    :param allow_debug: if True, print some debug info.
    :return:
    """
    r_list = [1, 1, 1, 1]
    # r_list = [1, 10, 100, 1000]
    r = r_list[opening_times]
    binary_arr = morphology_binary_opening(binary_arr, use_cv=use_cv, selem=sm.ball(r))
    if allow_debug:
        print("binary_opening use the biggest radius is {}.".format(r))
    return binary_arr


def plot_separated_bone(top_df, rib_df, shape, scatter_point_list=None):

    top_groupby_df = top_df.groupby(['y', 'z']).agg({'x': 'count'})
    top_groupby_df.columns = ['x.count']
    top_groupby_df.reset_index(inplace=True)
    # print(top_groupby_df.columns)
    top_arr = np.zeros((shape[0], shape[2]))
    top_index = top_groupby_df['z'].values, top_groupby_df['y'].values
    top_arr[top_index] = top_groupby_df['x.count']

    rib_groupby_df = rib_df.groupby(['y', 'z']).agg({'x':'count'})
    rib_groupby_df.columns = ['x.count']
    rib_groupby_df.reset_index(inplace=True)
    rib_arr = np.zeros((shape[0], shape[2]))
    rib_index = rib_groupby_df['z'].values, rib_groupby_df['y'].values
    rib_arr[rib_index] = rib_groupby_df['x.count']

    # scatter_point = np.array(scatter_point_list)

    plt.subplot(121)
    plt.imshow(top_arr+rib_arr)
    # plt.imshow(rib_arr)
    # plt.scatter(x=scatter_point[:, 2], y=scatter_point[:, 0], c='r')
    plt.subplot(122)
    plt.imshow(top_arr)
    plt.show()


def plot_yzd(temp_df=None, shape_arr=None, save=False, save_path=None, line_tuple_list=[]):
    temp_arr = np.zeros(shape_arr)
    temp_df_yz = temp_df.groupby(['y', 'z']).agg({'x': ['count']})
    temp_df_yz.columns = ['%s.%s' % e for e in temp_df_yz.columns.tolist()]
    temp_df_yz.reset_index(inplace=True)
    index = temp_df_yz['z'].values, temp_df_yz['y'].values
    temp_arr[index] = temp_df_yz['x.count'].values
    img = (255 / (temp_arr.max() - temp_arr.min())) * (temp_arr - temp_arr.min())
    # cv2.imwrite(os.path.join('/media/gy/BAC656DAC656970B/RibFrac/RF_remove/restore', 'spine_remaining.png'),
    #             img)
    cv2.imwrite(save_path, img)
    plt.figure()
    plt.imshow(temp_arr)
    for e, f in line_tuple_list:
        # e stands for z index.
        # f stands for y index.
        plt.plot(f, e)
    if save:
        plt.savefig(save_path)
    else:
        raise NotImplementedError


def plot_binary_array(binary_arr, title=None, save=True, save_path=None, line_tuple_list=[]):
    plt.figure()
    plt.title(title, color='red')
    plt.imshow(binary_arr.sum(axis=1))
    temp = binary_arr.sum(axis=1)
    for e, f in line_tuple_list:
        # e stands for z index.
        # f stands for y index.
        plt.plot(f, e)
    if save:
        plt.savefig(save_path)
        img = (255 / (temp.max() - temp.min())) * (temp - temp.min())
        cv2.imwrite(save_path, img)
    else:
        raise NotImplementedError


def plot_3d(image, threshold=0.5):
    """
    :param image:
    :param threshold: greater than threshold will be 1, otherwise 0.
    :return:
    """
    p = image.transpose(2, 1, 0)
    verts, faces = skimage.measure.marching_cubes_classic(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)        # Fancy indexing: `verts[faces]` to generate a collection of triangles
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()
