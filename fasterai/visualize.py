from numpy import ndarray
from matplotlib.axes import Axes
from fastai.conv_learner import *
from fastai.dataset import *

def plot_image_from_ndarray(image: ndarray, axes:Axes=None, figsize=(20,20)):
    if axes is None: 
        _,axes = plt.subplots(figsize=figsize)
    clipped_image =np.clip(image,0,1)
    axes.imshow(clipped_image)
    axes.axis('off')


def plot_images_from_ndarray_pairs(image_pairs: [(ndarray, ndarray)], figsize=(20,20), max_columns=6):
    num_pairs = len(image_pairs)
    num_images = num_pairs * 2
    rows, columns = get_num_rows_columns(num_images, max_columns)

    _, axes = plt.subplots(rows, columns, figsize=figsize)
    for i,(x,y) in enumerate(image_pairs):
        plot_image_from_ndarray(x, axes=axes.flat[i*2])
        plot_image_from_ndarray(y, axes=axes.flat[i*2+1])



def plot_images_from_dataset(ds: FilesDataset, start_idx: int, count: int, figsize=(20,20), max_columns=6):
    idxs = list(range(start_idx, start_idx+count))
    num_images = len(idxs)
    rows, columns = get_num_rows_columns(num_images, max_columns)
    _,axes=plt.subplots(rows,columns,figsize=figsize)

    for idx,ax in zip(idxs, axes.flat):  
        plot_image_from_ndarray(ds.denorm(ds[idx][0])[0], axes=ax)


def plot_image_outputs_from_model(ds: FilesDataset, model, idxs: [int], figsize=(20,20), max_columns=6):
    image_pairs = []

    for idx in idxs:
        x,_=ds[idx]
        preds = model(VV(x[None]))
        image_pairs.append((ds.denorm(x[None])[0], ds.denorm(preds)[0]))

    plot_images_from_ndarray_pairs(image_pairs, figsize=figsize, max_columns=max_columns)

def get_num_rows_columns(num_images: int, max_columns: int):
    columns = min(num_images, max_columns)
    rows = num_images//columns
    rows = rows if rows * columns == num_images else rows + 1
    return rows, columns
