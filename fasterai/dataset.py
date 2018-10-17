from fastai.dataset import *
from fasterai.files import *
from fasterai.images import *

class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path, x_tfms):
        self.y=y
        self.x_tfms=x_tfms
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_x(self, i): 
        x = super().get_x(i)
        for tfm in self.x_tfms:
            x,_ = tfm(x, False)
        return x
    def get_y(self, i): 
        return open_image(os.path.join(self.path, self.y[i]))
    def get_c(self): 
        return 0 

class NoiseToImageFilesDataset(FilesDataset):
    def __init__(self, fnames, y, path, x_tfms=None):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, None, path)
    def get_y(self, i): 
        return open_image(os.path.join(self.path, self.y[i]))
    def get_x(self, i): 
        raw_random = V(torch.randn(3, self.get_sz(),self.get_sz()).normal_(0, 1))
        return F.tanh(raw_random)
    def get_c(self): 
        return 0 


class ImageGenDataLoader():
    def __init__(self, sz:int, bs:int, path:Path, random_seed=None, x_noise:bool=False, 
            keep_pct:float=1.0, x_tfms:[Transform]=[]):
        
        self.md = None
        self.sz = sz
        self.bs = bs 
        self.path = path
        self.x_tfms = x_tfms
        self.x_noise = x_noise
        self.random_seed = random_seed
        self.keep_pct = keep_pct

    def get_model_data(self):
        if self.md is not None:
            return self.md

        fnames_full,label_arr_full,all_labels = folder_source(self.path.parent, self.path.name)
        fnames_full = ['/'.join(Path(fn).parts[-2:]) for fn in fnames_full]
        if self.random_seed is None:
            np.random.seed()
        else:
            np.random.seed(self.random_seed)
        keeps = np.random.rand(len(fnames_full)) < self.keep_pct
        fnames = np.array(fnames_full, copy=False)[keeps]
        val_idxs = get_cv_idxs(len(fnames), val_pct=min(0.01/self.keep_pct, 0.1))
        ((val_x,trn_x),(val_y,trn_y)) = split_by_idx(val_idxs, np.array(fnames), np.array(fnames))
        aug_tfms = [RandomFlip()] 
        tfms = (tfms_from_stats(inception_stats, self.sz, tfm_y=TfmType.PIXEL, aug_tfms=aug_tfms))
        dstype = NoiseToImageFilesDataset if self.x_noise else MatchedFilesDataset
        datasets = ImageData.get_ds(dstype, (trn_x,trn_y), (val_x,val_y), tfms, path=self.path, x_tfms=self.x_tfms)
        self.md = ImageData(self.path.parent, datasets, self.bs, num_workers=16, classes=None)
        return self.md