
import os
from os.path import join
from .ashs_exp import ASHSExperimentBase
from .utils.upsample_linear_method import linear_isotropic_upsampling, pad_image_with_world_alignment_in_memory
from .utils.trim_neck import trim_neck_in_memory
from .utils.tool import copy_or_link_file, get_nifti_sform_matrix, set_nifti_sform_matrix
from picsl_greedy import Greedy3D
from picsl_c3d import Convert3D
import yaml
from types import SimpleNamespace
import torch
import time
import shutil
import tempfile
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Protocol, Dict, Literal, Type, Callable, Any, Tuple, List
import sys

_itksnap_label_file_preamble="""
################################################
# ITK-SnAP Label Description File
# File format:
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields:
#    IDX:   Zero-based index
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    MSH:   Label mesh visibility (0 or 1)
#  LABEL:   Label description
################################################"""


# Normalize a simple ITK image to 0-255 range
def normalize_intensity_to_uchar(img: sitk.Image, qtile=0.995) -> sitk.Image:
    c3d = Convert3D()
    c3d.push(img)
    c3d.execute(f'-stretch 0 {qtile*100}% 0 255 -clip 0 255')
    return c3d.peek(-1)

# Rescale intensity so that the image can be safely saved as an image
# of short type without compromizing any details. No clipping is applied
def rescale_intensity_to_short(img: sitk.Image, qtile=0.995) -> sitk.Image:
    c3d = Convert3D()
    c3d.push(img)
    c3d.execute(f'-stretch {(1-qtile)*100}% {qtile*100}% 0 4096')
    return c3d.peek(-1)

# Crop segmentation by margin
def trim_segmentation(seg: sitk.Image, margin_mm:float=10) -> sitk.Image:
    c3d = Convert3D()
    c3d.push(seg)
    c3d.execute(f'-trim {margin_mm}mm')
    return c3d.peek(-1)

# Crop to segmentation and normalize intensity of images
def normalize_and_reslice(img: sitk.Image, ref_img: sitk.Image, qtile=0.995) -> sitk.Image:
    c3d = Convert3D()
    c3d.push(ref_img)
    c3d.push(img)
    c3d.execute(f'-stretch 0 {qtile*100}% 0 255 -clip 0 255 -reslice-identity')
    return c3d.peek(-1)

# Overlay segmentation on top of an image with transparency for QC visualization
def overlay_segmentation_on_image(image: sitk.Image, segmentation: sitk.Image, label_file: str, alpha=0.5) -> sitk.Image:
    
    c3d = Convert3D()
    c3d.push(image)
    c3d.push(segmentation)
    c3d.execute(f'-oli {label_file} {alpha}')
    return sitk.Compose([c3d.peek(i) for i in range(3)])


def generate_ashs_qc(
    volumes: Dict[str, sitk.Image],
    row_labels: Dict[str, str],
    output_path: str,
    title: str|None = None):
    
    # Initialize c3d converter
    c3d = Convert3D()
    
    # Define slice specifications (percentage positions)
    cor_slice_spec = np.linspace(0.0, 1.0, 7)[1:-1] * 100    
    sag_slice_spec = np.linspace(0.0, 1.0, 5)[1:-1] * 100   
    
    # Normalize each of the images
    def normalize(img):
        c3d.push(img)
        c3d.execute('-stretch 0 99.5% 0 255 -clip 0 255 -type uchar')
        return c3d.peek(-1)
    
    # Function to extract slices using c3d
    def extract_slices(img, raicode, slice_spec):
        # Extract the separate components
        n_comp = img.GetNumberOfComponentsPerPixel()
        i_comp = [img] if n_comp == 1 else [ sitk.VectorIndexSelectionCast(img, i) for i in range(n_comp) ]

        # Extract the slices for each component separately
        slices = []
        slice_spec_str = ','.join([f'{pct}%' for pct in slice_spec])
        for i, comp_img in enumerate(i_comp):
            c3d.execute('-clear')
            c3d.push(comp_img)
            c3d.execute(f'-swapdim {raicode} -slice z {slice_spec_str}')
            slices.append([ c3d.peek(i) for i in range(len(slice_spec)) ])
        
        # Slices are now grouped by component, then slice
        return [ sitk.Compose([ slices[c][i] for c in range(n_comp) ]) for i in range(len(slice_spec)) ]        
    
    # Extract the slices in coronal and sagittal views
    slices_cor = { k: extract_slices(v, 'RSA', cor_slice_spec) for k,v in volumes.items() }
    slices_sag = { k: extract_slices(v, 'ASR', sag_slice_spec) for k,v in volumes.items() }
            
    # Create the montage using matplotlib
    n_cor = len(cor_slice_spec)
    n_sag = len(sag_slice_spec)
    n_cols = n_cor + n_sag
    n_rows = len(volumes)
    
    # Compute the extents of different slices
    k1 = next(iter(volumes.keys()))
    w_cor = slices_cor[k1][0].GetSpacing()[0] * slices_cor[k1][0].GetSize()[0]
    w_sag = slices_sag[k1][0].GetSpacing()[0] * slices_sag[k1][0].GetSize()[0]
    h_cor = slices_cor[k1][0].GetSpacing()[1] * slices_cor[k1][0].GetSize()[1]
    aspect = n_rows * h_cor / (n_cor * w_cor + n_sag * w_sag)
    
    fig = plt.figure(figsize=(16, 16 * aspect))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.02, hspace=0.02, 
                  width_ratios=[w_cor]*n_cor + [w_sag]*n_sag)
    
    def plot_slice(ax, img, label=None):
        arr = sitk.GetArrayFromImage(img).squeeze().astype(np.uint8)
        ext = np.array(img.GetSize()) * np.array(img.GetSpacing())
        if img.GetNumberOfComponentsPerPixel() == 1:
           ax.imshow(arr, cmap='gray', aspect=1, extent=[0, ext[0], 0, ext[1]])
        else:
           ax.imshow(arr, aspect=1, extent=[0, ext[0], 0, ext[1]])
        ax.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_xticks(np.linspace(0, ext[0], 7))
        ax.set_yticks(np.linspace(0, ext[1], 7))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params('both', length=0, width=1, which='major')
        if label:
            ax.text(0.02, 0.98, label, transform=ax.transAxes, 
                fontsize=10, va='top', ha='left', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Row 0: Template
    for i, (k,lab) in enumerate(row_labels.items()):
        for j, slice_img in enumerate(slices_cor[k]):
            ax = fig.add_subplot(gs[i, j])
            plot_slice(ax, slice_img, lab if j == 0 else None)
        
        for j, slice_img in enumerate(slices_sag[k]):
            ax = fig.add_subplot(gs[i, n_cor + j])
            plot_slice(ax, slice_img)
            
    # Add title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=96, bbox_inches='tight', facecolor='white', edgecolor='black', pad_inches=0.1)
    plt.close(fig)
    
    
class SegmentationLabelMap():
    def __init__(self, 
                 fn_itksnap_labels: str | None = None, 
                 fn_dataset_json_file: str | None = None):
        
        self.fn_itksnap_labels = fn_itksnap_labels
        if fn_itksnap_labels is not None:
            self.labels = self._parse_itksnap_label_file(fn_itksnap_labels)
        elif fn_dataset_json_file is not None:
            self.labels = self._parse_nnunet_dataset_json(fn_dataset_json_file)
        else:
            raise ValueError("Either fn_itksnap_labels or fn_dataset_json_file must be provided.")
        
    def _parse_itksnap_label_file(self, fn_itksnap_labels: str) -> Dict[int, Dict[str, Any]]:
        label_dict = {}
        with open(fn_itksnap_labels, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    label_id = int(parts[0])
                    color = list(map(int, parts[1:4]))
                    alpha = int(parts[4])
                    name = ' '.join(parts[7:]).strip('"')
                    label_dict[label_id] = {'color': color, 'alpha': alpha, 'name': name}
        return label_dict
    
    def _parse_nnunet_dataset_json(self, fn_dataset_json_file: str) -> Dict[int, Dict[str, Any]]:
        cmap = plt.get_cmap('tab20')
        with open(fn_dataset_json_file, 'r') as f:
            dataset_info = yaml.safe_load(f)
        
        label_dict = {}
        for name, label_id in dataset_info.get('labels', {}).items():
            if name == f'empty_label_{label_id}':
                continue  # Skip empty labels
            elif name == 'background':
                label_dict[label_id] = {'color': [0.0, 0.0, 0.0], 'alpha': 0.0, 'name': name}  # Background is transparent
            else:
                color = [int(c*255) for c in cmap(label_id % 20)[:3]]
                label_dict[label_id] = {'color': color, 'alpha': 1.0, 'name': name}  
        
        return label_dict
    
    def export_itksnap_label_file(self, output_path: str):
        with open(output_path, 'w') as f:
            f.write(_itksnap_label_file_preamble.strip() + '\n')
            for label_id, info in self.labels.items():
                color = info['color']
                alpha = info['alpha']
                name = info['name']
                f.write(f'{label_id:4d} {color[0]:4d} {color[1]:4d} {color[2]:4d}\t{alpha:2d} {alpha:2d} {alpha:2d}\t"{name}"\n')
                
    def to_nnunet_dict_with_contiguous_labels(self):
        nnunet_labels = {'background': 0}
        max_label = max(self.labels.keys())
        for label_id in range(1, max_label + 1):
            name = self.labels.get(label_id, {}).get('name', f'empty_label_{label_id}')
            nnunet_labels[name] = label_id
        return nnunet_labels
    
    def oli_file(self):
        """
        Context manager that provides a temporary ITK-SNAP label file for this label set. 
        If the label set was initialized with an ITK-SNAP label file, that file will be used directly. 
        Otherwise, a temporary file will be created from the dataset.json information and deleted after use.
        """
        class OLIFile:
            def __init__(self, labelset: SegmentationLabelMap):
                self.labelset = labelset
                
            def __enter__(self):
                if self.labelset.fn_itksnap_labels is not None:
                    return self.labelset.fn_itksnap_labels
                
                self.tmp_file = tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt')
                self.labelset.export_itksnap_label_file(self.tmp_file.name)
                return self.tmp_file.name
            
            def __exit__(self, exc_type, exc_value, traceback):
                if hasattr(self, 'tmp_file'):
                    os.remove(self.tmp_file.name)
        
        return OLIFile(self)
    
    @property
    def label_ids(self) -> List[int]:
        return list(self.labels.keys())
    
    @property
    def label_names(self) -> List[str]:
        return [info['name'] for info in self.labels.values()]
    
    
def generate_ashs_segmentation_qc(
    seg: sitk.Image, 
    t1: sitk.Image, 
    t2: sitk.Image, 
    labelset: SegmentationLabelMap,
    output_path: str,
    t1_only = False,
    title: str|None = None,
    trim_margin_mm: float = 3.0):
    
    seg = trim_segmentation(seg, trim_margin_mm)
    t2 = normalize_and_reslice(t2, seg)
    t1 = normalize_and_reslice(t1, seg)
    
    # Obtain a file for the c3d -oli command
    with labelset.oli_file() as label_file:
        if not t1_only:
            vols = {'t2': t2, 't1': t1, 
                    't2s': overlay_segmentation_on_image(t2, seg, label_file), 
                    't1s': overlay_segmentation_on_image(t1, seg, label_file)}
            
            generate_ashs_qc(vols, 
                            row_labels={'t2': 'T2 (native)', 't1': 'T1 to T2', 't2s': 'T2-seg', 't1s': 'T1-seg'},
                            output_path=output_path, title=title)    
        else:
            vols = {'t1': t1, 't1s': overlay_segmentation_on_image(t1, seg, label_file)}
            
            generate_ashs_qc(vols, 
                            row_labels={'t1': 'T1 (native)', 't1s': 'T1-seg'},
                            output_path=output_path, title=title)
    

def generate_ashs_registration_qc(
    template_img: sitk.Image,
    t1_to_t2: sitk.Image,
    t2_img: sitk.Image,
    output_path: str,
    t1_only = False,
    title: str|None = None):

    if not t1_only:    
        vols = {'t2': normalize_intensity_to_uchar(t2_img),
                't1': normalize_intensity_to_uchar(t1_to_t2),
                'temp': normalize_intensity_to_uchar(template_img)}

        generate_ashs_qc(vols, 
                        row_labels={'t2': 'T2 (native)', 't1': 'T1 to T2', 'temp': 'Template to T2'},
                        output_path=output_path, title=title)
    else:
        vols = {'t1': normalize_intensity_to_uchar(t1_to_t2),
                'temp': normalize_intensity_to_uchar(template_img)}

        generate_ashs_qc(vols, 
                        row_labels={'t1': 'T1 (native)', 'temp': 'Template to T1'}, 
                        output_path=output_path, title=title)
    

class Timer:
    def __init__(self) -> None:
        self.t_elapsed = 0
        self.n = 0
        
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        self.t_elapsed += time.time() - self.start_time
        self.n += 1
        self.start_time = None
        
    @property
    def total(self):
        return self.t_elapsed
    
    @property
    def average(self):
        return self.t_elapsed / self.n if self.n > 0 else np.NAN
    
    
# Define a signature for the callback function that can be used to report progress and send 
# updates to the caller during processing. The callback function is meant to be used in 
# conjunction with ITK-SNAP DSS, but not limited to it
class ProgressCallbackType(Protocol):
    def __call__(self, 
                 progress: float|None=None, 
                 progress_range: Tuple[float, float] = (0.0, 1.0), 
                 attachments: Dict[str,str]|None = None,
                 message: str|None = None) -> None: 
        ...
        
def default_progress_callback(progress: float|None=None,
                              progress_range: Tuple[float, float] = (0.0, 1.0), 
                              attachments: Dict[str,str]|None = None,
                              message: str|None = None) -> None:
        pass
        
        
class ASHSProcessor:
    """Common code for preprocessing T2/T1 pairs to generate ROIs for inference/training"""
    def __init__(self, config, t1_only=False, training_mode=False, overwrite_existing=False, save_intermediates=True, create_links=True):
        self.training_mode = training_mode
        self.overwrite_existing = overwrite_existing
        self.save_intermediates = save_intermediates
        self.create_links = create_links
        self.t1_only = t1_only
        self.t1_only_fake_t2_spacing = config.get('T1_ONLY_FAKE_T2_SPACING', 0.4)
        self.t2_cropping = config.get('ASHS_TSE_REGION_CROP', ASHSProcessor.get_config_defaults()['ASHS_TSE_REGION_CROP'])
        self.greedy_num_threads = config.get('GREEDY_NUM_THREADS', 0)
        self.tm_neck = Timer()
        self.tm_reg_t1_t2_whole = Timer()
        self.tm_reg_t1_temp = Timer()
        self.tm_reg_t1_t2_local = Timer()
        self.tm_prep_inr = Timer()
        self.tm_finalize = Timer()
        
        
    @staticmethod
    def get_config_defaults() -> Dict[str, Any]:
        return { 'ASHS_TSE_REGION_CROP': 0.2 }

    def get_close_to_iso_integer_scaling(self, image : sitk.Image):
        """
        Generate a c3d scaling command that will make this image close to isotropic while
        scaling the largest dimension by integer factors. For example, input with spacing
        (0.41,0.39,2.0) will generate scaling command '100x100x500'
        
        Integer scaling is desirable if we want to be able to downsample segmentations back
        to original spacing of the image. But, of course, resulting images may not be very
        isotropuic
        """             
        in_spacing = np.array(image.GetSpacing())
        s_min = np.min(in_spacing)
        scaling = np.floor(in_spacing / s_min + 0.5)
        scaling_str = 'x'.join([f'{100*s}' for s in scaling]) + '%'
        return scaling_str

        
    def preprocess(self, exp: ASHSExperimentBase, callback: ProgressCallbackType = default_progress_callback, progress_range=(0.0, 0.25)):
        """
        Preprocess the T1 and T2 images for a given case, including neck trimming, registration, and ROI cropping.
        """
        nt = self.greedy_num_threads
        gpe, lpe, tpe = exp.gpe, exp.lpe, exp.tpe
        
        # Perform neck trimming if necessary
        if self.overwrite_existing or not gpe.t1_neck_trim.exists():
            with self.tm_neck:
                trimmed_img = trim_neck_in_memory(gpe.t1_native.data)
                gpe.t1_neck_trim.data = rescale_intensity_to_short(trimmed_img)
        else:
            print(f"Neck-trimmed T1 already exists at {gpe.t1_neck_trim.filename}. Skipping neck trimming step.")
            
        callback(progress=0.2, progress_range=progress_range, 
                 message=f"Neck trimming completed in {self.tm_neck.total:.1f} s.")
        
        # Check if nnUNet inputs already exist, and if not, perform the registration steps
        if self.overwrite_existing or not all(lp.t2_patch_hyperres.exists() and lp.roi_in_t1_space.exists() for lp in lpe.values()):
            
            # ------- global T1 to T2 registration using ASHS pipeline -------
            with self.tm_reg_t1_t2_whole:
                
                g = Greedy3D()
                if not self.t1_only:

                    # If specified, crop the T2 image before registration      
                    t2_cropped_img = rescale_intensity_to_short(gpe.t2_whole_img.data)
                    if self.t2_cropping > 0:
                        # Create the cropping command for c3d based on the specified cropping fraction
                        c3d  = Convert3D()
                        c3d.push(t2_cropped_img)
                        c = self.t2_cropping * 100
                        c3d.execute(f'-swapdim RSA -region {c}x{c}x0% {100-2*c}x{100-2*c}x100%')
                        t2_cropped_img = c3d.peek(-1)
                    
                    # Perform the affine registration
                    g.execute(f'-threads {nt} -z -a -dof 6 -ia-identity -m NMI '
                            f'-i t2 t1 -n 100x100x10 -o {gpe.fn_save_mat_path_t2_to_t1_global} ', 
                            t2=t2_cropped_img, t1=gpe.t1_neck_trim.data)
                    
                    # Apply the registration 
                    g.execute(f'-threads {nt} -rf t2 -rm t1 t1_reg_to_t2 '
                            f'-r {gpe.fn_save_mat_path_t2_to_t1_global}', t1_reg_to_t2=None)
                    
                    if self.save_intermediates:
                        gpe.t1_reg_to_t2.data = g['t1_reg_to_t2']
                        
                else:
                    
                    # Write identity matrix for T1 to T2 registration
                    identity_matrix = np.eye(4)
                    np.savetxt(gpe.fn_save_mat_path_t2_to_t1_global, identity_matrix, fmt='%.6f')
                    
                    # Set this variable to avoid Pylance errors
                    t2_cropped_img = gpe.t1_neck_trim.data
                                
            # ------- global T1 to template registration using ASHS pipeline -------
            with self.tm_reg_t1_temp:

                # 1. rigid
                g.execute(f'-threads {nt} -a -dof 6 -m NCC 2x2x2 '
                        f"-i template_3tt1 trim_t1_image "
                        f"-o rigid -n 400x0x0x0 "
                        f"-ia-image-centers -search 400 5 5", 
                        template_3tt1=rescale_intensity_to_short(tpe.template_3tt1.data), 
                        trim_t1_image=gpe.t1_neck_trim.data, rigid=None)
                
                # 2. affine
                g.execute(f'-threads {nt} -a -m NCC 2x2x2 '
                        f'-i template_3tt1 trim_t1_image '
                        f'-o {gpe.fn_template_to_3tt1_affine_matrix} -n 400x80x40x0 '
                        f'-ia rigid')
                
                # 3. deformable
                g.execute(f'-threads {nt} -m NCC 2x2x2 -e 0.5 -n 60x20x0 -sv '
                        f'-i template_3tt1 trim_t1_image -it {gpe.fn_template_to_3tt1_affine_matrix} '
                        f'-o warpfwd -oinv warpinv', 
                        warpfwd=None, warpinv=None)
                
                # 4. apply
                g.execute(f"-threads {nt} -rf trim_t1_image "
                        f"-rm template_3tt1 template_to_3tt1 "
                        f"-rm temp_roi_left t1_roi_left "
                        f"-rm temp_roi_right t1_roi_right "
                        f"-r {gpe.fn_template_to_3tt1_affine_matrix},-1 warpinv", 
                        template_to_3tt1=None, temp_roi_left=tpe.template_roi['left'].data, temp_roi_right=tpe.template_roi['right'].data, 
                        t1_roi_left=None, t1_roi_right=None)
                
                # Read off the ROI images
                t1_roi = { 'left': g['t1_roi_left'], 'right': g['t1_roi_right'] }

                if self.save_intermediates:
                    gpe.template_to_3tt1.data = g['template_to_3tt1']
                    for k, v in t1_roi.items():
                        lpe[k].roi_in_t1_space.data = v
        
            # ------- Perform the cropping based on the ROIs  ------- 
            with self.tm_reg_t1_t2_local:
                
                if not self.t1_only:
                
                    # Pad the T2 image with world alignment
                    t2_padded_img = pad_image_with_world_alignment_in_memory(t2_cropped_img, [40, 40, 40], [40, 40, 40])
                    
                    for side_, lp in lpe.items():
                                                
                        # Determine the target spacing for the T2 upsampling (replace the largest spacing with the second largest one)
                        scaling_str = self.get_close_to_iso_integer_scaling(t2_cropped_img)
                                        
                        # Crop the T2 using the T1 ROI and apply the new spacing
                        c3d = Convert3D()
                        c3d.push(t2_padded_img)
                        c3d.push(t1_roi[side_])
                        c3d.execute(f'-popas ROI_T1 -as T2 -push ROI_T1 -reslice-matrix {gpe.fn_save_mat_path_t2_to_t1_global} -trim 5mm '
                                    f'-resample {scaling_str} -as ROI_T2 -dup -push T2 -reslice-identity -swapdim RPI')
                        lp.t2_patch_hyperres.data = c3d.peek(-1)
                        roi_t2 = c3d.peek(-2)

                        # Compute local registration between T2 and T1
                        g.execute(f'-threads {nt} -z -a -dof 6 -ia {gpe.fn_save_mat_path_t2_to_t1_global} -m NMI '
                                f'-i t2 t1 -gm mask -n 100x50 -o {lp.fn_save_mat_path_t2_to_t1_local} ', 
                                t2=lp.t2_patch_hyperres.data, t1=gpe.t1_neck_trim.data, mask=roi_t2)
                        
                        # Apply the registration to the T1 and resample it into the isotropic space
                        g.execute(f'-threads {nt} -rf t2 -rm t1 t1_reg_to_t2 '
                                f'-r {lp.fn_save_mat_path_t2_to_t1_local}', t1_reg_to_t2=None)
                        lp.t1_patch_warped_hyperres.data = g['t1_reg_to_t2']

                        # For QC purposes, map template all the way to T2 space
                        g.execute(f"-threads {nt} -rf t2 -rm template_3tt1 template_to_t2 "
                                    f"-r {lp.fn_save_mat_path_t2_to_t1_local} {gpe.fn_template_to_3tt1_affine_matrix},-1 warpinv",
                                    template_to_t2=None)
                
                else:
                    
                    # In T1-only mode, we want to resample the T1 patch to near-isotropic spacing that
                    # is close to what the nnUNet model was trained on
                    for side_, lp in lpe.items():
                        c3d = Convert3D()
                        c3d.push(t1_roi[side_])
                        c3d.push(gpe.t1_neck_trim.data)
                        c3d.execute(f'-popas T1 -trim 5mm -resample-mm {self.t1_only_fake_t2_spacing}mm '
                                    f'-swapdim RPI -scale 0 -as T2 -dup -push T1 -reslice-identity')
                        lp.t2_patch_hyperres.data = c3d.peek(-2)
                        lp.t1_patch_warped_hyperres.data = c3d.peek(-1)

                        # For QC purposes, map template all the way to T2 space
                        g.execute(f"-threads {nt} -rf t2 -rm template_3tt1 template_to_t2 "
                                  f"-r {gpe.fn_template_to_3tt1_affine_matrix},-1 warpinv",
                                  t2=lp.t2_patch_hyperres.data, template_to_t2=None)
                    
                        
                for side_, lp in lpe.items():

                    # Write out the cropped T2 image and create links for nnUNet input
                    copy_or_link_file(lp.t2_patch_hyperres.filename, lp.hl_nnunet_t2_input, 
                                    create_links=self.create_links, force_overwrite=True, create_dir=True, quiet=True)
                                    
                    # Write out the cropped T1 and create link for nnUNet input
                    copy_or_link_file(lp.t1_patch_warped_hyperres.filename, lp.hl_nnunet_t1_input, 
                                    create_links=self.create_links, force_overwrite=True, create_dir=True, quiet=True)
                    
                    # Generate registration QC screenshot
                    generate_ashs_registration_qc(
                        template_img=g['template_to_t2'],
                        t1_to_t2=lp.t1_patch_warped_hyperres.data,
                        t2_img=lp.t2_patch_hyperres.data,
                        output_path=lp.fn_registration_qc,
                        t1_only=self.t1_only,
                        title=f"{exp.qc_title} Registration QC - {side_.capitalize()}")
                        
        else:
            print(f"NNUNet input patches already exist for both sides. Skipping registration and cropping steps.")
            
        t_total = self.tm_reg_t1_t2_whole.total + self.tm_reg_t1_temp.total + self.tm_reg_t1_t2_local.total
        callback(progress=1.0, progress_range=progress_range, 
                    attachments={f'{side_.capitalize()} Registration QC': lp.fn_registration_qc for side_, lp in lpe.items()},
                    message=f"Registration and ROI cropping completed in {t_total:.1f} s.")
        
    def prepare_inr(self, exp:ASHSExperimentBase, callback: ProgressCallbackType = default_progress_callback, progress_range=(0.0, 0.25)):
        nt = self.greedy_num_threads
        gpe, lpe = exp.gpe, exp.lpe
        with self.tm_prep_inr:

            # Use the input segmentation to crop the T2 image at native resolution. How much padding to apply
            # is not obvious - we want to provide some context, but not too much to keep the computation reasonable
            for side_, lp in lpe.items():
                c3d = Convert3D()
                c3d.execute(f'-threads {nt}')
                
                # Crop the primary image. We want to keep its native spacing but use the segmentations' bounds
                # to trim the image. Since we allow the manual segmentation to not be in the same spacing as the
                # T2 image we have to first reslice the segmentation into the T2 space, and then use it to crop the T2 image.
                c3d.push(gpe.t2_whole_img.data)
                c3d.push(lp.input_seg.data)
                c3d.execute(f'-trim 5mm -popas S -as T2 -push S -thresh 1 inf 1 0 -reslice-identity -thresh 0.5 inf 1 0 -trim 5mm -as S_T2 -push T2 -reslice-identity -swapdim RPI -as T2P')
                lp.inr_primary.data = c3d.peek(-1)
                
                # Also generate a dummy mask
                c3d.execute(f'-clear -push T2P -scale 0 -shift 1 -as T2M')
                lp.inr_primary_mask.data = c3d.peek(-1)
                
                # Upsample this image to near-isotropic spacing (this is the INR 'ground truth'?)
                scale_cmd = self.get_close_to_iso_integer_scaling(lp.inr_primary.data)
                c3d.execute(f'-clear -push T2P -int 1 -resample {scale_cmd} -as T2GT')
                lp.inr_primary_gt.data = c3d.peek(-1)
                
                # Write out the segmentation and dummy mask
                c3d.execute(f'-clear -push S -swapdim RPI')
                lp.inr_primary_seg.data = c3d.peek(-1)
                
                # Crop the secondary image. Here we first need to define the ROI in the T1 space
                # and then use it to crop the T1 image. The Greedy command applies rigid transform
                # to send the T2 segmentation into the T1 image space.  
                g = Greedy3D()
                g.execute(f'-threads {nt} -ri 0 -rf t1 -rm seg seg_in_t1 -r {lp.fn_save_mat_path_t2_to_t1_local},-1', 
                          t1=gpe.t1_neck_trim.data, seg=lp.input_seg.data, seg_in_t1=None)
                
                # Now crop the secondary image using the segmentation
                c3d.push(gpe.t1_neck_trim.data)
                c3d.push(g['seg_in_t1'])
                c3d.execute(f'-trim 5mm -popas S -insert S 1 -reslice-identity -swapdim RPI')
                t1_native_patch = c3d.peek(-1)
                
                # Finally, we should correct the header of the secondary image to incorporate the rigid 
                # registration, otherwise INR will be confounded by the misalignment between the two modalities.
                S = get_nifti_sform_matrix(t1_native_patch)
                M = np.loadtxt(lp.fn_save_mat_path_t2_to_t1_local)
                S_new = np.linalg.inv(M) @ S
                set_nifti_sform_matrix(t1_native_patch, S_new)
                lp.inr_secondary.data = t1_native_patch
                
                # Resample the T2 mask into the t1 native patch space
                c3d.push(lp.inr_secondary.data)
                c3d.execute(f'-as T1P -push T2M -int 0 -reslice-identity')
                lp.inr_secondary_mask.data = c3d.peek(-1)
                
                # Also generate the target-resolution T1 image
                c3d.execute(f'-push T2GT -push T1P -int 0 -reslice-identity')
                lp.inr_secondary_gt.data = c3d.peek(-1)
                
                # And finally the mask for the INR inference - perhaps we can in the future 
                # limit this to just the area around the segmentation, why upsample whole patch?
                c3d.execute(f'-push T2GT -scale 0 -shift 1')
                lp.inr_inference_mask.data = c3d.peek(-1)
                
                # Populate the links for the INR training directory
                if lp.dir_inr_train_input is not None:
                    d_inr:str = lp.dir_inr_train_input
                    os.makedirs(d_inr, exist_ok=True)
                    
                    # Create all the links
                    for dst, src in {
                        't2_LR': lp.inr_primary.filename,               # Native T2 patch
                        't2_seg_LR': lp.inr_primary_seg.filename,       # Native T2 segmentation
                        't2_mask_LR': lp.inr_primary_mask.filename,     # All ones
                        't1_LR': lp.inr_secondary.filename,             # Native T1 match, header adjusted
                        't1_seg_LR': lp.inr_secondary_mask.filename,    # Same as T1 mask below
                        't1_mask_LR': lp.inr_secondary_mask.filename,   # Region of overlap T2 on T1
                        't2': lp.inr_primary_gt.filename,               # Resampled T2 patch at target resolution (INR GT)
                        't1': lp.inr_secondary_gt.filename,             # Resampled T1 patch at target resolution (INR GT)
                        'brainmask': lp.inr_inference_mask.filename     # Inferencing mask
                    }.items():
                        copy_or_link_file(src, join(lp.dir_inr_train_input, f'{side_}_{dst}.nii.gz'), create_links=True)
                
        callback(progress=1.0, progress_range=progress_range, 
                 message=f"INR preprocessing cropping completed in {self.tm_prep_inr.total:.1f} s.")


    def postprocess(self, exp: ASHSExperimentBase, callback: ProgressCallbackType = default_progress_callback, progress_range=(0.95, 1.0)):
        """
        Postprocess the nnUNet segmentation outputs, including mapping back to native space and generating QC visualizations.
        """
        with self.tm_finalize:
            # Create a new greedy instance for the registration
            g = Greedy3D()
            
            for side_, lp in exp.lpe.items():
                if not self.t1_only:
                    # Apply resampling to original T2 space for each side
                    # Then map it back to the original T2 space using the inverse of the global registration
                    g.execute(f'-threads {self.greedy_num_threads} -rf t2 -ri LABEL 0.2vox '
                                f'-rm final_seg {lp.t2_seg_native.filename} -r ',
                                t2=exp.gpe.t2_whole_img.data, final_seg=lp.nnunet_seg.data)
                    
                else:
                    # Map the T1 segmentation back to the original T1 space
                    g.execute(f'-threads {self.greedy_num_threads} -rf t1 -ri LABEL 0.2vox '
                                f'-rm final_seg {lp.t2_seg_native.filename} -r ',
                                t1=exp.gpe.t1_native.data, final_seg=lp.nnunet_seg.data)

                
                
        callback(progress=1.0, progress_range=progress_range, message=f"Post-processing completed in {self.tm_finalize.total:.1f} s.")
                
