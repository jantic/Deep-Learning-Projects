from fastai.dataset import *
from fasterai.files import *
from fasterai.images import *


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path, x_tfms=[], x_noise=None):
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

class NoiseVectorToImageDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path:Path, x_tfms=[], x_noise=64):
        self.y=y
        assert(len(fnames)==len(y))
        self.x_noise=x_noise
        super().__init__(fnames, transform, path)
    def get_y(self, i): 
        return open_image(os.path.join(self.path, self.y[i]))
    def get_x(self, i): 
        return np.random.normal(loc=0.0, scale=1.0, size=(self.x_noise,1,1))
    def get_c(self): 
        return 0 
    def get(self, tfm, x, y):
        return (x,y) if tfm is None else (x, tfm(y,y)[1])


class ImageGenDataLoader():
    def __init__(self, sz:int, bs:int, path:Path, random_seed=None, x_noise:int=None, 
            keep_pct:float=1.0, x_tfms:[Transform]=[], file_ext='jpg'):
        
        self.md = None
        self.sz = sz
        self.bs = bs 
        self.path = path
        self.x_tfms = x_tfms
        self.x_noise = x_noise
        self.random_seed = random_seed
        self.keep_pct = keep_pct
        self.file_ext = file_ext

    def get_model_data(self):
        if self.md is not None:
            return self.md

        #fnames_full,label_arr_full,all_labels = folder_source(self.path.parent, self.path.name)
        #fnames_full = ['/'.join(Path(fn).parts[-2:]) for fn in fnames_full]
        fnames_full = [str(path) for path in self.path.glob('**/*.' + self.file_ext)]
        fnames_full = [Path(fname.replace(str(self.path) + '/','')) for fname in fnames_full]

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
        dstype = NoiseVectorToImageDataset if self.x_noise is not None else MatchedFilesDataset
        datasets = ImageData.get_ds(dstype, (trn_x,trn_y), (val_x,val_y), tfms, path=self.path, x_tfms=self.x_tfms, x_noise=self.x_noise)
        self.md = ImageData(self.path.parent, datasets, self.bs, num_workers=16, classes=None)

        #optimization
        #if self.sz<128:
            #self.md = self.md.resize(self.sz*2)
        return self.md