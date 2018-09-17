from fastai.dataset import *
from fasterai.files import *
from fasterai.images import *

class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): 
        return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): 
        return 0

def get_matched_image_model_data(image_size: int, batch_size: int, root_data_path: Path, train_root_path: Path, proj_id: str, 
        keep_pct: float=0.1, random_seed: int=42):
    train_x_path = generate_image_preprocess_path(train_root_path, is_x=True, size=image_size, uid=proj_id)
    train_y_path = generate_image_preprocess_path(train_root_path, is_x=False, size=image_size, uid=proj_id)
    x_paths, y_paths = get_matched_xy_file_lists(train_x_path, train_y_path)
    x_paths_str = convert_paths_to_str(x_paths)
    y_paths_str = convert_paths_to_str(y_paths)
    np.random.seed(random_seed)
    keeps = np.random.rand(len(x_paths_str)) < keep_pct
    fnames_x = np.array(x_paths_str, copy=False)[keeps]
    fnames_y = np.array(y_paths_str, copy=False)[keeps]
    val_idxs = get_cv_idxs(len(fnames_x), val_pct=min(0.01/keep_pct, 0.1))
    ((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, np.array(fnames_x), np.array(fnames_y))
    tfms = tfms_from_stats(inception_stats, image_size, tfm_y=TfmType.PIXEL, aug_tfms=transforms_side_on, sz_y=image_size)
    datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x,trn_y), (val_x,val_y), tfms, path=train_y_path.parent)
    md = ImageData(root_data_path, datasets, batch_size, num_workers=16, classes=None)
    return md