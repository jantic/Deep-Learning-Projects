import multiprocessing
from fastai.dataset import *
from fasterai.files import *
from pathlib import Path
from itertools import repeat
from PIL import Image


def generate_image_preprocess_path(source_path: Path, is_x:bool, size: int, uid: str):
    name = generate_image_preprocess_name(source_path, is_x, size, uid)
    path = source_path.parent/name
    return path

def generate_image_preprocess_name(source_path: Path, is_x:bool, size: int, uid: str):
   return generate_preprocess_name(source_path, is_x, uid) + '_' + str(size)

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

def resize_image(image: Image, size: int):
    return image.resize((size,size))

def to_grayscale_image(image: Image):
    return image.convert('L')