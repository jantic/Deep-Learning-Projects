import multiprocessing
from fastai.dataset import *
from fasterai.files import *
from pathlib import Path
from itertools import repeat
from PIL import Image
from numpy import ndarray

def generate_image_preprocess_path(source_path: Path, is_x:bool, uid: str):
    name = generate_image_preprocess_name(source_path, is_x, uid)
    path = source_path.parent/name
    return path

def generate_image_preprocess_name(source_path: Path, is_x:bool, uid: str):
    return generate_preprocess_name(source_path, is_x, uid)

def transform_image_and_save_new(function, sourcepath: Path, destpath: Path):
    try:
        with Image.open(sourcepath) as image:
            image = function(image)
            image.save(destpath)
    except Exception as ex:
        print(ex)
        
def transform_images_to_new_directory(function, sourceroot: Path, destroot: Path):
    destroot.mkdir(exist_ok=True)
    raw_sourcepaths, _, _ = folder_source(sourceroot.parent, sourceroot.name)
    #First make the destination directories if they don't already exist- we want the subsequent operations to be threadsafe.  Then create
    #another generator of destpaths for use in the image generation
    generate_folders_for_dest(destpaths=dest_path_generator(sourceroot=sourceroot, raw_sourcepaths=raw_sourcepaths, destroot=destroot))   
    destpaths = dest_path_generator(sourceroot=sourceroot, raw_sourcepaths=raw_sourcepaths, destroot=destroot)
    sourcepaths = (sourceroot.parent/Path(raw_sourcepath) for raw_sourcepath in raw_sourcepaths)
    numthreads = multiprocessing.cpu_count()//2
    
    with ThreadPoolExecutor(numthreads) as e:
        try:
            e.map(partial(transform_image_and_save_new, function), sourcepaths, destpaths)
        except Exception as ex:
            print(ex)

def resize_image(im: Image, targ: int):
    r,c = im.size
    ratio = targ/min(r,c)
    sz = (scale_to(r, ratio, targ), scale_to(c, ratio, targ))
    return im.resize(sz, Image.LINEAR)

def to_grayscale_image(image: Image):
    return image.convert('L')


class EasyTensorImage():
    def __init__(self, source_tensor: torch.Tensor, ds:FilesDataset):
        self.array = self._convert_to_denormed_ndarray(source_tensor.data, ds=ds)   
        self.tensor = self._convert_to_denormed_tensor(self.array)
    
    def _convert_to_denormed_ndarray(self, raw_tensor: torch.Tensor, ds:FilesDataset):
        return ds.denorm(to_np(raw_tensor.data))[0]

    def _convert_to_denormed_tensor(self, denormed_array: ndarray):
        return V(np.moveaxis(denormed_array,2,0))

class ModelImageSet():
    @staticmethod
    def get_list_from_model(ds: FilesDataset, model: nn.Module, idxs:[int]):
        image_sets = []

        for idx in idxs:
            x,y=ds[idx]
            orig_tensor = VV(x[None])
            real_tensor = V(y[None])
            gen_tensor = model(orig_tensor)

            orig_easy = EasyTensorImage(orig_tensor, ds)
            real_easy = EasyTensorImage(real_tensor, ds)
            gen_easy = EasyTensorImage(gen_tensor, ds)

            image_set = ModelImageSet(orig_easy,real_easy,gen_easy)
            image_sets.append(image_set)

        return image_sets  

    def __init__(self, orig: EasyTensorImage, real: EasyTensorImage, gen: EasyTensorImage):
        self.orig=orig
        self.real=real
        self.gen=gen