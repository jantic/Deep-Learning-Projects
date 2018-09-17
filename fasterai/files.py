import multiprocessing
import pandas as pd
from fastai.dataset import folder_source
import torch
from pathlib import Path
from itertools import repeat
import re



def generate_preprocess_path(source_path: Path, is_x:bool, uid: str):
    name = generate_preprocess_name(source_path, is_x, uid) 
    path = source_path.parent/name
    return path

def generate_preprocess_name(source_path: Path, is_x:bool, uid: str):
    middle = 'x' if is_x else 'y'
    return f'{source_path.name}_{middle}_{uid}'

def generate_full_path(rootpath: Path, relativepath: Path):
    return rootpath/relativepath


def generate_dest_path(sourceroot: Path, sourcepath: Path, destroot: Path):
    relativepath = sourcepath.relative_to(sourceroot)
    destpath = generate_full_path(destroot, relativepath)
    return destpath

def dest_path_generator(sourceroot: Path, raw_sourcepaths: [Path], destroot: Path):
    return (generate_dest_path(sourceroot=sourceroot, sourcepath=generate_full_path(sourceroot.parent, Path(raw_sourcepath)), destroot=destroot) 
            for raw_sourcepath in raw_sourcepaths)

def generate_folders_for_dest(destpaths: [Path]):
    destdirs = set(destpath.parent for destpath in destpaths)
    
    for destdir in destdirs:
        destdir.mkdir(parents=True, exist_ok=True)

def convert_to_xy_comparable_path(path:Path):
    return Path(*Path(path).parts[1:])

#I'm not a huge fan of the implementation, but it is relatively fast and 
#it filters out unmatched files automatically for us.
def get_matched_xy_file_lists(x_root_path: Path, y_root_path: Path):
    x_orig, x_rel = generate_comparable_path_info(x_root_path)
    x_rel_set = set(x_rel)
    y_orig, y_rel = generate_comparable_path_info(y_root_path)
    y_rel_set = set(y_rel)
    x_filtered =  [orig for orig, rel in zip(x_orig, x_rel) if rel in y_rel_set]
    x_filtered.sort()
    y_filtered =  [orig for orig, rel in zip(y_orig, y_rel) if rel in x_rel_set]
    y_filtered.sort()
    assert(len(x_filtered) == len(y_filtered))
    return (x_filtered, y_filtered)

def generate_comparable_path_info(root_path: Path):
    orig_paths,_,_ = folder_source(root_path.parent, root_path.name)
    path_strs = [str(path) for path in orig_paths]
    root_folder = Path(*root_path.parts[1:]).name
    relative_paths = [re.sub('^' + root_folder, '', path) for path in path_strs]
    return orig_paths, relative_paths


def convert_paths_to_str(paths: [Path]):
    return [str(path) for path in paths]

def save_model(model, path): 
    torch.save(model.state_dict(), path)
