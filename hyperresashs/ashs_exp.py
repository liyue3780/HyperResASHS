import os
from os.path import join
from types import SimpleNamespace
import SimpleITK as sitk
from matplotlib.gridspec import GridSpec
from typing import Dict, Type, Callable, Any


class LazyPipelineTypeTraits:
    """Traits for handling lazy loading/saving of different object types in the pipeline."""
    @staticmethod
    def loader(path:str) -> Any:
        return None
    
    @staticmethod
    def saver(data: Any, path:str):
        pass


class LazyPipelineImageTraits(LazyPipelineTypeTraits):
    """Traits for handling lazy loading/saving of SimpleITK images in the pipeline."""
    @staticmethod
    def loader(path:str) -> sitk.Image:
        return sitk.ReadImage(path)
    
    @staticmethod
    def saver(data:sitk.Image, path:str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sitk.WriteImage(data, path)
        
        
class LazyPipelineCastToShortImageTraits(LazyPipelineTypeTraits):
    """Traits for handling lazy loading/saving of SimpleITK images in the pipeline."""
    @staticmethod
    def loader(path:str) -> sitk.Image:
        return sitk.ReadImage(path)
    
    @staticmethod
    def saver(data:sitk.Image, path:str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sitk.WriteImage(sitk.Cast(data, sitk.sitkInt16), path)


class LazyPipelineElement:
    """
    A generic lazy-loading pipeline element that defers reading from disk until needed.
    
    Supports multiple object types via a registration system. New types can be registered
    using LazyPipelineElement.register() with custom loader/saver functions.
    
    Example:
        LazyPipelineElement.register(
            sitk.Transform,
            loader=lambda path: sitk.ReadTransform(path),
            saver=lambda transform, path: sitk.WriteTransform(transform, path)
        )
        lazy_xfm = LazyPipelineElement("transform.mat", sitk.Transform)
        transform = lazy_xfm.data  # Loads on first access
    """
    def __init__(self, filename, traits: Type[LazyPipelineTypeTraits]):
        self.filename = filename
        self._data = None
        self.traits = traits
        
    @property
    def data_or_none(self) -> Any:
        """Read the object from disk if it hasn't been read yet, and return it. Returns None if the file does not exist."""
        if self._data is None and os.path.exists(self.filename):
            # Get the loader function for this type
            loader = getattr(self.traits, 'loader', None)
            if loader is None:
                raise ValueError(f"No loader registered for type {self.traits.__name__}")
            self._data = loader(self.filename)
            
        return self._data
    
    @property
    def data(self) -> Any:
        """Read the object from disk if it hasn't been read yet, and return it. Raises exception if data does not exist."""
        if self._data is None:
            if not os.path.exists(self.filename):
                raise FileNotFoundError(f"File does not exist: {self.filename}")

            # Get the loader function for this type
            loader = getattr(self.traits, 'loader', None)
            if loader is None:
                raise ValueError(f"No loader registered for type {self.traits.__name__}")
            self._data = loader(self.filename)
            
        return self._data
    
    @data.setter
    def data(self, value:Any):
        """Set the object and write it to disk immediately"""
        self._data = value
        if self._data is not None:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            saver = getattr(self.traits, 'saver', None)
            if saver is None:
                raise ValueError(f"No saver registered for type {self.traits.__name__}")
            saver(value, self.filename)
        
    def exists(self) -> bool:
        """Check if the image file exists on disk"""
        return os.path.exists(self.filename)
    
    def __str__(self) -> str:
        return f"LazyPipelineElement(filename='{self.filename}', type={self.traits.__name__}, exists={self.exists()}, loaded={self._data is not None})"
        

class LazyImage(LazyPipelineElement):
    """
    Convenience wrapper for lazy-loading SimpleITK images.
    
    Specialized version of LazyPipelineElement for sitk.Image objects. Automatically
    defers reading from disk until accessed via the .data property, and immediately
    writes to disk when set.
    
    Example:
        lazy_img = LazyImage("path/to/image.nii.gz")
        if lazy_img.exists():
            img = lazy_img.data  # Loads on first access
            # Process img...
            lazy_img.data = processed_img  # Saves immediately
    """
    def __init__(self, filename, cast_to_int16=False):
        super().__init__(filename, LazyPipelineCastToShortImageTraits if cast_to_int16 else LazyPipelineImageTraits)      
        

class LazyInt16Image(LazyImage):
    """
    Convenience wrapper for lazy-loading SimpleITK images that should be saved as int16.
    
    Specialized version of LazyImage that automatically casts images to int16 when saving.
    
    Example:
        lazy_img = LazyInt16Image("path/to/image.nii.gz")
        if lazy_img.exists():
            img = lazy_img.data  # Loads on first access
            # Process img...
            lazy_img.data = processed_img  # Saves immediately, automatically cast to int16
    """
    def __init__(self, filename):
        super().__init__(filename, cast_to_int16=True)  
 
 
# This class stores the images generated using processing
class GlobalPipelineElements:
    def __init__(self, case_path:str, nm: SimpleNamespace, prefix:str=''):

        format = lambda fn: fn.format(prefix=prefix)

        # Registration matrices
        self.fn_save_mat_path_t2_to_t1_global = join(case_path, format(nm.t1_to_t2_affine_mat))
        self.fn_template_to_3tt1_affine_matrix = join(case_path, format(nm.template_to_t1_affine_mat))

        # Raw inputs                
        self.t2_whole_img = LazyImage(join(case_path, format(nm.t2_native_img)))
        self.t1_native = LazyImage(join(case_path, format(nm.t1_native_img)))
        
        # Processed images
        self.t1_neck_trim = LazyInt16Image(join(case_path, format(nm.t1_neck_trim_img)))
        self.t1_reg_to_t2 = LazyInt16Image(join(case_path, format(nm.t1_aligned_to_t2_img)))
        self.template_to_3tt1 = LazyInt16Image(join(case_path, format(nm.template_to_t1_warped_img)))
        
        # Final labelfile and volumes
        self.fn_final_labelfile = join(case_path, format(nm.final_labelfile))
        self.fn_final_volumes_csv = join(case_path, format(nm.final_volumes_csv))
    

class TemplatePipelineElements:
    def __init__(self, template_path:str, nm: SimpleNamespace):
        self.template_3tt1 = LazyImage(join(template_path, nm.template_img))
        self.template_roi = { side : LazyImage(join(template_path, nm.template_roi_img.format(side=side))) for side in ['left', 'right'] }


class LocalPipelineElements:
    def __init__(self, case_path:str, side:str, dataset_id: str, nm: SimpleNamespace, prefix:str='',
                 inr_path: str|None=None, nnunet_train_id: int|None = None):

        format = lambda fn: fn.format(side=side, dataset=dataset_id, prefix=prefix)

        # Folders
        # self.dir_local = join(case_path, test_folder, side)
        self.dir_nnunet_input = join(case_path, format(nm.nnunet_input_dir))
        self.dir_nnunet_output = join(case_path, format(nm.nnunet_output_dir))
        
        # Images and matrices generated during preprocessing
        self.roi_in_t1_space = LazyInt16Image(join(case_path, format(nm.template_roi_to_t1_warped_img)))
        self.fn_save_mat_path_t2_to_t1_local = join(case_path, format(nm.t1_to_t2_local_affine_mat))
        self.t2_patch_hyperres = LazyInt16Image(join(case_path, format(nm.t2_hyperres_patch_img)))
        self.t1_patch_warped_hyperres = LazyInt16Image(join(case_path, format(nm.t1_hyperres_patch_img)))
        self.fn_registration_qc = join(case_path, format(nm.registration_qc))
        
        # NNUNet input and output
        self.nnunet_train_id = nnunet_train_id
        self.hl_nnunet_t2_input = join(self.dir_nnunet_input, 'MTL_000_0000.nii.gz')
        self.hl_nnunet_t1_input = join(self.dir_nnunet_input, 'MTL_000_0001.nii.gz')
        self.nnunet_seg = LazyInt16Image(join(self.dir_nnunet_output, 'MTL_000.nii.gz'))
        self.fn_segmentation_qc = join(case_path, format(nm.segmentation_qc))
        
        # Final post-processed segmentation
        self.t2_seg_hyperres = LazyInt16Image(join(case_path, format(nm.final_hyperres_seg)))
        self.t2_seg_native = LazyInt16Image(join(case_path, format(nm.final_native_seg)))
        self.t1_seg_native = LazyInt16Image(join(case_path, format(nm.final_native_t1space_seg)))

        # The input path for INR training is different since INR code expects a certain directory structure
        self.dir_inr_train_input = inr_path
        
        # For training/INR, the primary and secondary modalities cropped at native resolution
        self.input_seg = LazyInt16Image(join(case_path, format(nm.t2_native_gt_seg)))
        self.inr_primary = LazyInt16Image(join(case_path, format(nm.inr_input_t2_lr_img)))
        self.inr_primary_mask = LazyInt16Image(join(case_path, format(nm.inr_input_t2_lr_mask)))
        self.inr_primary_seg = LazyInt16Image(join(case_path, format(nm.inr_input_t2_lr_seg)))
        self.inr_primary_gt = LazyInt16Image(join(case_path, format(nm.inr_input_t2_gt_img)))
        self.inr_secondary = LazyInt16Image(join(case_path, format(nm.inr_input_t1_lr_img)))
        self.inr_secondary_mask = LazyInt16Image(join(case_path, format(nm.inr_input_t1_lr_mask)))
        self.inr_secondary_gt = LazyInt16Image(join(case_path, format(nm.inr_input_t1_gt_img)))
        self.inr_inference_mask = LazyInt16Image(join(case_path, format(nm.inr_input_inference_mask)))
        
        # After INR upsampling is done, the upsampled segmentation is resampled into the space of the
        # T2 hyper-resolution patch.
        self.t2_patch_hyperres_seg = LazyInt16Image(join(case_path, format(nm.t2_hyperres_gt_seg)))
                
class ASHSExperimentBase:
    def __init__(self, config: Dict[str, Any], case_path:str, nm: SimpleNamespace, 
                 sides=['left', 'right'], subject:str|None=None, date:str|None=None,
                 inr_path: Dict[str,str]|None=None, nnunet_train_id: Dict[str,int]|None=None,
                 prefix:str=''):
        self.config = config
        self.case_path = case_path
        self.inr_path = inr_path
        self.nnunet_train_id = nnunet_train_id
        self.dataset_id = 'Dataset{}_{}'.format(config['EXP_NUM'], config['MODEL_NAME'])

        self.subject = subject
        self.date = date
        self.prefix = prefix
        self.qc_title = f'{subject} - {date}' if (subject and date) else f'{subject}' if subject else ''

        self.gpe = GlobalPipelineElements(case_path, nm, prefix=self.prefix)
        self.lpe = { side: LocalPipelineElements(case_path, side, self.dataset_id, nm, prefix=self.prefix,
                                                 inr_path=inr_path[side] if inr_path is not None else None,
                                                 nnunet_train_id=nnunet_train_id[side] if nnunet_train_id is not None else None) 
                    for side in sides }
        self.tpe = TemplatePipelineElements(config['TEMPLATE_PATH'], nm)

