from numpy import ndarray
from matplotlib.axes import Axes
from fastai.conv_learner import *
from fastai.dataset import *
from fasterai.wgan import WGANGenTrainingResult, WGANCriticTrainingResult, WGANTrainer
from fasterai.files import *
from fasterai.images import *
from IPython.display import display
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import shutil
import statistics
import cv2


class ModelImageVisualizer():
    def __init__(self):
        return 

    def plot_transformed_image(self, path: Path, model: nn.Module, figsize=(10,10), sz:int=None):
        result = self.get_transformed_image_ndarray(path, model, sz)
        self.plot_image_from_ndarray(result, figsize=figsize)

    def get_transformed_image_ndarray(self, path: Path, model: nn.Module, sz:int=None):
        orig = self.get_model_ready_image_ndarray(path, sz)
        result = model(VV(orig[None])).detach().cpu().numpy()
        result = np.rollaxis(result,1,4)
        return result[0]

    def get_model_ready_image_ndarray(self, path: Path, sz:int=None):
        flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
        im = cv2.imread(path, flags).astype(np.float32)/255
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if sz is not None:
            im = scale_min(im, sz)
        im = np.moveaxis(im,2,0)
        return im

    def plot_image_from_ndarray(self, image: ndarray, axes:Axes=None, figsize=(20,20)):
        if axes is None: 
            _,axes = plt.subplots(figsize=figsize)
        clipped_image =np.clip(image,0,1)
        axes.imshow(clipped_image)
        axes.axis('off')


    def plot_images_from_image_sets(self, image_sets: [ModelImageSet], figsize=(20,20), max_columns=6, immediate_display=True):
        num_sets = len(image_sets)
        num_images = num_sets * 2
        rows, columns = self._get_num_rows_columns(num_images, max_columns)

        fig, axes = plt.subplots(rows, columns, figsize=figsize)
        for i, image_set in enumerate(image_sets):
            self.plot_image_from_ndarray(image_set.orig.array, axes=axes.flat[i*2])
            self.plot_image_from_ndarray(image_set.gen.array, axes=axes.flat[i*2+1])

        if immediate_display:
            display(fig)


    def plot_image_outputs_from_model(self, ds: FilesDataset, model: nn.Module, idxs: [int], figsize=(20,20), max_columns=6, immediate_display=True):
        image_sets = ModelImageSet.get_list_from_model(ds=ds, model=model, idxs=idxs)
        self.plot_images_from_image_sets(image_sets=image_sets, figsize=figsize, max_columns=max_columns, immediate_display=immediate_display)

    def _get_num_rows_columns(self, num_images: int, max_columns: int):
        columns = min(num_images, max_columns)
        rows = num_images//columns
        rows = rows if rows * columns == num_images else rows + 1
        return rows, columns


class ModelGraphVisualizer():
    def __init__(self):
        return 
     
    def write_model_graph_to_tensorboard(self, ds: FilesDataset, model: nn.Module, tbwriter: SummaryWriter):
        try:
            x,_=ds[0]
            tbwriter.add_graph(model, V(x[None]))
        except Exception as e:
            print(("Failed to generate graph for model: {0}. Note that there's an outstanding issue with "
                + "scopes being addressed here:  https://github.com/pytorch/pytorch/pull/12400").format(e))

class ModelStatsVisualizer(): 
    def __init__(self):
        return 

    def write_tensorboard_stats(self, model: nn.Module, iter_count:int, tbwriter: SummaryWriter):
        gradients = [x  for x in model.parameters() if x.grad is not None]
        gradient_nps = [to_np(x.data) for x in gradients]
 
        if len(gradients) == 0:
            return 

        avg_norm = sum(x.data.norm() for x in gradients)/len(gradients)
        tbwriter.add_scalar('/gradients/avg_norm', avg_norm, iter_count)

        median_norm = statistics.median(x.data.norm() for x in gradients)
        tbwriter.add_scalar('/gradients/median_norm', median_norm, iter_count)

        max_norm = max(x.data.norm() for x in gradients)
        tbwriter.add_scalar('/gradients/max_norm', max_norm, iter_count)

        min_norm = min(x.data.norm() for x in gradients)
        tbwriter.add_scalar('/gradients/min_norm', min_norm, iter_count)

        num_zeros = sum((np.asarray(x)==0.0).sum() for x in  gradient_nps)
        tbwriter.add_scalar('/gradients/num_zeros', num_zeros, iter_count)


        avg_gradient= sum(x.data.mean() for x in gradients)/len(gradients)
        tbwriter.add_scalar('/gradients/avg_gradient', avg_gradient, iter_count)

        median_gradient = statistics.median(x.data.median() for x in gradients)
        tbwriter.add_scalar('/gradients/median_gradient', median_gradient, iter_count)

        max_gradient = max(x.data.max() for x in gradients) 
        tbwriter.add_scalar('/gradients/max_gradient', max_gradient, iter_count)

        min_gradient = min(x.data.min() for x in gradients) 
        tbwriter.add_scalar('/gradients/min_gradient', min_gradient, iter_count)

class ImageGenVisualizer():
    def __init__(self):
        self.model_vis = ModelImageVisualizer()

    def output_image_gen_visuals(self, ds: FilesDataset, model: nn.Module, iter_count:int, tbwriter: SummaryWriter, jupyter:bool=False):
        #TODO:  Parameterize these
        start_idx=0
        count = 8
        end_index = start_idx + count
        idxs = list(range(start_idx,end_index))
        image_sets = ModelImageSet.get_list_from_model(ds=ds, model=model, idxs=idxs)
        self.write_tensorboard_images(image_sets=image_sets, iter_count=iter_count, tbwriter=tbwriter)
        if jupyter:
            self._show_images_in_jupyter(image_sets)
    
    def write_tensorboard_images(self, image_sets:[ModelImageSet], iter_count:int, tbwriter: SummaryWriter):
        orig_images = []
        gen_images = []
        real_images = []

        for image_set in image_sets:
            orig_images.append(image_set.orig.tensor)
            gen_images.append(image_set.gen.tensor)
            real_images.append(image_set.real.tensor)

        tbwriter.add_image('orig images', vutils.make_grid(orig_images, normalize=True), iter_count)
        tbwriter.add_image('gen images', vutils.make_grid(gen_images, normalize=True), iter_count)
        tbwriter.add_image('real images', vutils.make_grid(real_images, normalize=True), iter_count)


    def show_images_in_jupyter(self, image_sets:[ModelImageSet]):
        #TODO:  Parameterize these
        figsize=(20,20)
        max_columns=4
        immediate_display=True
        self.model_vis.plot_images_from_image_sets(image_sets, figsize=figsize, max_columns=max_columns, immediate_display=immediate_display)


class WganTrainerStatsVisualizer():
    def __init__(self):
        return

    def write_tensorboard_stats(self, gresult: WGANGenTrainingResult, cresult: WGANCriticTrainingResult, iter_count:int, tbwriter: SummaryWriter):
        tbwriter.add_scalar('/loss/wdist', cresult.wdist, iter_count)
        tbwriter.add_scalar('/loss/dfake', cresult.dfake, iter_count)
        tbwriter.add_scalar('/loss/dreal', cresult.dreal, iter_count)
        tbwriter.add_scalar('/loss/gpenalty', cresult.gpenalty, iter_count)
        tbwriter.add_scalar('/loss/gcost', gresult.gcost, iter_count)
        tbwriter.add_scalar('/loss/gcount', gresult.gcount, iter_count)
        tbwriter.add_scalar('/loss/conpenalty', cresult.conpenalty, iter_count)
        tbwriter.add_scalar('/loss/gaddlloss', gresult.gaddlloss, iter_count)

    def print_stats_in_jupyter(self, gresult: WGANGenTrainingResult, cresult: WGANCriticTrainingResult):
        print(f'\nWDist {cresult.wdist}; RScore {cresult.dreal}; FScore {cresult.dfake}; GAddlLoss {gresult.gaddlloss}; ' + 
                f'GCount: {gresult.gcount}; GPenalty: {cresult.gpenalty}; GCost: {gresult.gcost}; ConPenalty: {cresult.conpenalty}')


class LearnerStatsVisualizer():
    def __init__(self):
        return
    
    def write_tensorboard_stats(self, metrics, iter_count:int, tbwriter:SummaryWriter):
        if isinstance(metrics, list):
            tbwriter.add_scalar('/loss/trn_loss', metrics[0], iter_count)    
            if len(metrics) == 1: return
            tbwriter.add_scalar('/loss/val_loss', metrics[1], iter_count)        
            if len(metrics) == 2: return

            for metric in metrics[2:]:
                name = metric.__name__
                tbwriter.add_scalar('/loss/'+name, metric, iter_count)
                    
        else: 
            tbwriter.add_scalar('/loss/trn_loss', metrics, iter_count)

  





