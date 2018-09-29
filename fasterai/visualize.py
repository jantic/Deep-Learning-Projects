from numpy import ndarray
from matplotlib.axes import Axes
from fastai.conv_learner import *
from fastai.dataset import *
from fasterai.wgan import WGANGenTrainingResult, WGANCriticTrainingResult, WGANTrainer
from fasterai.files import *
from IPython.display import display
from tensorboardX import SummaryWriter
import shutil
import statistics

def plot_image_from_ndarray(image: ndarray, axes:Axes=None, figsize=(20,20)):
    if axes is None: 
        _,axes = plt.subplots(figsize=figsize)
    clipped_image =np.clip(image,0,1)
    axes.imshow(clipped_image)
    axes.axis('off')


def plot_images_from_ndarray_pairs(image_pairs: [(ndarray, ndarray)], figsize=(20,20), max_columns=6, immediate_display=True):
    num_pairs = len(image_pairs)
    num_images = num_pairs * 2
    rows, columns = _get_num_rows_columns(num_images, max_columns)

    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    for i,(x,y) in enumerate(image_pairs):
        plot_image_from_ndarray(x, axes=axes.flat[i*2])
        plot_image_from_ndarray(y, axes=axes.flat[i*2+1])

    if immediate_display:
        display(fig)


def plot_images_from_dataset(ds: FilesDataset, start_idx: int, count: int, figsize=(20,20), max_columns=6):
    idxs = list(range(start_idx, start_idx+count))
    num_images = len(idxs)
    rows, columns = _get_num_rows_columns(num_images, max_columns)
    _,axes=plt.subplots(rows,columns,figsize=figsize)

    for idx,ax in zip(idxs, axes.flat):  
        plot_image_from_ndarray(ds.denorm(ds[idx][0])[0], axes=ax)


def plot_image_outputs_from_model(ds: FilesDataset, model: nn.Module, idxs: [int], figsize=(20,20), max_columns=6, immediate_display=True):
    image_pairs = []

    for idx in idxs:
        x,_=ds[idx]
        preds = model(VV(x[None]))
        image_pairs.append((ds.denorm(x[None])[0], ds.denorm(preds)[0]))

    plot_images_from_ndarray_pairs(image_pairs, figsize=figsize, max_columns=max_columns, immediate_display=immediate_display)

def _get_num_rows_columns(num_images: int, max_columns: int):
    columns = min(num_images, max_columns)
    rows = num_images//columns
    rows = rows if rows * columns == num_images else rows + 1
    return rows, columns

class ModelStatsVisualizer():
    def __init__(self, base_dir: Path, module: nn.Module, name: str, stats_iters: int=10):
        self.base_dir = base_dir
        self.name = name
        log_dir = base_dir/name
        clear_directory(log_dir)
        self.tbwriter = SummaryWriter(log_dir=str(log_dir))
        self.hook = module.register_forward_hook(self.forward_hook)
        self.stats_iters = stats_iters
        self.iter_count = 0

    def forward_hook(self, module: nn.Module, input, output): 
        self.iter_count += 1
        if self.iter_count % self.stats_iters != 0:
            return

        self._write_tensorboard_stats(module)  

    def _write_tensorboard_stats(self, module: nn.Module):
        gradients = [x  for x in module.parameters() if x.grad is not None]
        gradient_nps = [to_np(x.data) for x in gradients]
 
        if len(gradients) == 0:
            return 

        avg_norm = sum(x.data.norm() for x in gradients)/len(gradients)
        self.tbwriter.add_scalar('/gradients/avg_norm', avg_norm, self.iter_count)

        median_norm = statistics.median(x.data.norm() for x in gradients)
        self.tbwriter.add_scalar('/gradients/median_norm', median_norm, self.iter_count)

        max_norm = max(x.data.norm() for x in gradients)
        self.tbwriter.add_scalar('/gradients/max_norm', max_norm, self.iter_count)

        min_norm = min(x.data.norm() for x in gradients)
        self.tbwriter.add_scalar('/gradients/min_norm', min_norm, self.iter_count)

        num_zeros = sum((np.asarray(x)==0.0).sum() for x in  gradient_nps)
        self.tbwriter.add_scalar('/gradients/num_zeros', num_zeros, self.iter_count)


        avg_gradient= sum(x.data.mean() for x in gradients)/len(gradients)
        self.tbwriter.add_scalar('/gradients/avg_gradient', avg_gradient, self.iter_count)

        median_gradient = statistics.median(x.data.median() for x in gradients)
        self.tbwriter.add_scalar('/gradients/median_gradient', median_gradient, self.iter_count)

        max_gradient = max(x.data.max() for x in gradients) 
        self.tbwriter.add_scalar('/gradients/max_gradient', max_gradient, self.iter_count)

        min_gradient = min(x.data.min() for x in gradients) 
        self.tbwriter.add_scalar('/gradients/min_gradient', min_gradient, self.iter_count)

    def close(self):
        self.tbwriter.close()
        self.hook.remove()


class WganTrainerStatsVisualizer():
    def __init__(self, base_dir: Path, trainer: WGANTrainer, name: str, stats_iters: int=10, visual_iters: int=100):
        self.base_dir = base_dir
        self.name = name
        log_dir = base_dir/name
        clear_directory(log_dir)
        self.tbwriter = SummaryWriter(log_dir=str(log_dir))
        self.hook = trainer.register_train_loop_hook(self.train_loop_hook)
        self.stats_iters = stats_iters
        self.visual_iters = visual_iters
        self.iter_count = 0

    def train_loop_hook(self, trainer: WGANTrainer, gresult: WGANGenTrainingResult, cresult: WGANCriticTrainingResult): 
        self.iter_count += 1
        if self.iter_count % self.stats_iters != 0:
            return

        self._print_stats_in_jupyter(gresult, cresult)
        self._write_tensorboard_stats(gresult, cresult) 

        if self.iter_count % self.visual_iters != 0:
            return

        self._show_images_in_jupyter(trainer)
        

    def _write_tensorboard_stats(self, gresult: WGANGenTrainingResult, cresult: WGANCriticTrainingResult):
        self.tbwriter.add_scalar('/loss/wdist', cresult.wdist, self.iter_count)
        self.tbwriter.add_scalar('/loss/dfake', cresult.dfake, self.iter_count)
        self.tbwriter.add_scalar('/loss/dreal', cresult.dreal, self.iter_count)
        self.tbwriter.add_scalar('/loss/gpenalty', cresult.gpenalty, self.iter_count)
        self.tbwriter.add_scalar('/loss/gcost', gresult.gcost, self.iter_count)
        self.tbwriter.add_scalar('/loss/gcount', gresult.gcount, self.iter_count)
        self.tbwriter.add_scalar('/loss/conpenalty', cresult.conpenalty, self.iter_count)

    def _print_stats_in_jupyter(self, gresult: WGANGenTrainingResult, cresult: WGANCriticTrainingResult):
        print(f'\nWDist {cresult.wdist}; RScore {cresult.dreal}; FScore {cresult.dfake}' + 
                f'; GCount: {gresult.gcount}; GPenalty: {cresult.gpenalty}; GCost: {gresult.gcost}; ConPenalty: {cresult.conpenalty}')

    def _show_images_in_jupyter(self, trainer: WGANTrainer):
        md = trainer.md
        model = trainer.netG
        #TODO:  Parameterize these
        start_idx=0
        count = 8
        figsize=(20,20)
        max_columns=4
        immediate_display=True
        end_index = start_idx + count
        idxs = list(range(start_idx,end_index))
        plot_image_outputs_from_model(ds=md.val_ds, model=model, idxs=idxs, max_columns=max_columns, immediate_display=immediate_display)
    
    def close(self):
        self.tbwriter.close()
        self.hook.remove()



