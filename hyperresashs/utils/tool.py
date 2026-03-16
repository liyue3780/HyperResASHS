import SimpleITK as sitk
import numpy as np
import os
from batchgenerators.utilities.file_and_folder_operations import save_json, join
from scipy.ndimage import label as ndi_label
from scipy.linalg import polar
from picsl_c3d import Convert3D
import shutil
import torch

def flip_image(input_path, output_path, flip_axis):
    image_itk = sitk.ReadImage(input_path)
    image_array = sitk.GetArrayFromImage(image_itk)

    # flip it
    image_array = np.flip(image_array, axis=flip_axis)

    # save it
    new_itk = sitk.GetImageFromArray(image_array)
    new_itk.SetOrigin(image_itk.GetOrigin())
    new_itk.SetDirection(image_itk.GetDirection())
    new_itk.SetSpacing(image_itk.GetSpacing())

    sitk.WriteImage(new_itk, output_path)


def convert_each_ground_truth_file_as_continuous(convert_file, nnunet_raw_path):
    # set convert file
    convert_content = [ele_.strip('\n') for ele_ in open(convert_file)]

    # read the ground truth
    gt_path = os.path.join(nnunet_raw_path, 'labelsTr')
    file_list = os.listdir(gt_path)

    # set output path
    for file_ in file_list:
        refseg_path = os.path.join(gt_path, file_)
        replaced_path = os.path.join(gt_path, file_)

        # read the source image
        image_itk = sitk.ReadImage(refseg_path)
        image_array = sitk.GetArrayFromImage(image_itk)

        # create a totally new array
        new_array = np.zeros(image_array.shape)

        # replace each subregion as target label
        for tlt_ in convert_content:
            old_id = int(tlt_.split(',')[0])
            new_id = int(tlt_.split(',')[1])
            new_array = new_array + ((image_array == old_id) + 0) * int(new_id)
            print('replace {} as {}'.format(old_id, new_id))

        # generate new nii file
        new_itk = sitk.GetImageFromArray(new_array)
        new_itk.SetOrigin(image_itk.GetOrigin())
        new_itk.SetDirection(image_itk.GetDirection())
        new_itk.SetSpacing(image_itk.GetSpacing())
        sitk.WriteImage(new_itk, replaced_path)

        print('finish and cover the {}'.format(replaced_path))


def save_label_mapping_to_txt(snap_label_path, save_path):
    snap_content = [ele_.strip('\n') for ele_ in open(snap_label_path) if ele_[0] != '#']

    # new and old label
    convert_list = []
    for ii, ele_ in enumerate(snap_content):
        # get curr id
        curr_id = None
        curr_ele_all_list = ele_.split(' ')
        for kkt_ in curr_ele_all_list:
            if kkt_ != '':
                curr_id = kkt_
                break
        
        # get curr label name
        curr_label = ele_.strip('"').split('"')[-1]
        # convert_list.append('{},{},{}'.format(curr_id, ii, curr_label))  # use continuous labels
        convert_list.append('{},{},{}'.format(curr_id, curr_id, curr_label))  # use original labels
    
    with open(save_path, 'w') as f_:
        for ele_ in convert_list:
            f_.write(ele_ + '\n')


def make_nnunet_dataset_json(convert_file, output_folder):
    # read convert file
    convert_content = [ele_.strip('\n') for ele_ in open(convert_file)]

    # count the number of training set
    labeltr_folder = os.path.join(output_folder, 'labelsTr')
    numtraining = len(os.listdir(labeltr_folder))

    # make different dict
    ## channel name
    modality_list = []
    file_list = os.listdir(os.path.join(output_folder, 'imagesTr'))
    for file_ in file_list:
        modality_ = file_.split('_')[2].strip('.nii.gz')
        if modality_ not in modality_list:
            modality_list.append(modality_)
    channel_names = {}
    for ii in range(len(modality_list)):
        channel_names[str(ii)] = modality_list[ii]

    ## specific label
    labels = {}
    for ele_ in convert_content:
        new_id = ele_.split(',')[1]
        label_name = ele_.split(',')[2]

        if label_name == 'Clear Label':
            label_name = 'background'
        
        if label_name in labels:
            if labels[label_name] != int(new_id):
                raise ValueError('Label conflicts!')
        else:
            labels[label_name] = int(new_id)
    
    ## other information
    dataset_json = {
        'channel_names': channel_names,  
        'labels': labels,
        'numTraining': numtraining,
        'file_ending': ".nii.gz",
    }

    # fill those skipped labels
    fill_labels(dataset_json)
    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)


def fill_labels(d):
    labels = d["labels"]

    value_to_key = {v: k for k, v in labels.items()}
    max_val = max(labels.values())
    
    new_labels = {}
    for i in range(max_val + 1):
        if i in value_to_key:
            new_labels[value_to_key[i]] = i
        else:
            new_labels[f"empty_label_{i}"] = i
    
    d["labels"] = new_labels
    return d

def linear_resample_to_spacing_using_itkimage(image_primary, new_spacing):
    original_spacing = image_primary.GetSpacing()
    original_size = image_primary.GetSize()

    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    new_spacing = [ospc * osz / nsz for ospc, osz, nsz in zip(original_spacing, original_size, new_size)]

    c3d = Convert3D()
    c3d.push(image_primary)
    c3d.execute(f"-resample {new_size[0]}x{new_size[1]}x{new_size[2]}")
    resampled_image = c3d.peek(-1)

    return resampled_image

def copy_or_link_file(src, dst, create_links=True, force_overwrite=False, relative_links=True, create_dir=False, quiet=False):
    """
    Copy or create a symbolic link for a file, with options to control behavior when the destination already exists.
    Parameters:
    - src: Source file path (existing file to copy or to which link will point)
    - dst: Destination file path (where to copy or create link)
    - create_links: If True, create a symbolic link instead of copying the file. Default is True.
    - force_overwrite: If True, overwrite the destination file/link if it already exists. Default is False.
    - relative_links: If True and create_links is True, create a relative symbolic link. Default is True.
    - create_dir: If True, create the parent directory of the destination if it does not exist. Default is False.
    - quiet: If True, suppress output messages. Default is False.
    """
    # Implement the following logic:
    #   - If existing file is different mode than desired, delete it first
    #   - If existing file is same mode as desired and is a file, and force_overwrite is False, skip copying/linking
    #   - If existing file is same mode as desired and is a link, and force_overwrite is False, skip copying/linking but only if it's referencing the same source
    qprint = print if not quiet else lambda *args, **kwargs: None
    dir = os.path.dirname(dst)
    if create_dir:
        os.makedirs(dir, exist_ok=True)
        
    if os.path.exists(dst):
        if create_links and os.path.islink(dst):
            existing_src = os.readlink(dst)
            if existing_src == os.path.abspath(src) and not force_overwrite:
                return
            else:
                qprint(f"Existing link at {dst} points to {existing_src}, which is different from desired source {src}. Removing it.")
                os.remove(dst)
        elif not create_links and os.path.isfile(dst):
            if not force_overwrite:
                return
            else:
                qprint(f"Existing file at {dst} will be overwritten.")
                os.remove(dst)
        else:
            qprint(f"Existing file at {dst} is of a different type than desired. Removing it.")
            os.remove(dst)
            
    if create_links:
        if relative_links:
            # Compute the relative path of source from the destination directory
            dir_fd = os.open(os.path.abspath(dir), os.O_RDONLY )
            os.symlink(os.path.relpath(src, dir), os.path.relpath(dst, dir), dir_fd=dir_fd)
        else:
            os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy(src, dst)
        
def get_nifti_sform_matrix(img:sitk.Image) -> np.ndarray:
    """Get the NIFTI RAS-space affine transformation matrix from a SimpleITK image."""
    # Get the direction cosines, spacing, and origin
    m_dir = np.array(img.GetDirection()).reshape(3, 3)
    m_scale = np.diag(np.array(img.GetSpacing()))
    v_origin = np.array(img.GetOrigin())

    # Convert from LPS to RAS by negating the first two axes
    m_lps_to_ras = np.diag([-1, -1, 1]) 
    
    # Combine the transformations to get the affine matrix
    A = m_lps_to_ras @ m_dir @ m_scale    
    b = m_lps_to_ras @ v_origin
    
    # Construct 4x4 homogeneous matrix
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = b

    # Return the matrix
    return M

def set_nifti_sform_matrix(img:sitk.Image, M:np.ndarray):
    """Set the NIFTI RAS-space affine transformation matrix for a SimpleITK image."""
    # Extract the rotation+scaling (A) and translation (b) components from the input matrix
    A = M[:3, :3]
    b = M[:3, 3]

    # Convert from RAS to LPS by negating the first two axes
    m_ras_to_lps = np.diag([-1, -1, 1])
    A_lps = m_ras_to_lps @ A
    b_lps = m_ras_to_lps @ b
    
    # Compute polar decomposition to split A_lps into rotation and scaling components
    U, P = polar(A_lps)

    # Set the direction cosines, spacing, and origin based on the input matrix
    img.SetDirection(U.flatten())
    img.SetSpacing(P.diagonal())
    img.SetOrigin(b_lps)
    
    
def nnunet_get_num_cpu_threads(user_value) -> int:
    max_threads = os.cpu_count() or 1
    if user_value < 1:
        return max_threads
    else:
        return min(user_value, max_threads)


def nnunet_configure_device(device:str, num_cpu_threads: int) -> torch.device:
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print(f"Auto-detected device: {device}")
    
    if device == 'cpu':
        # let's allow torch to use hella threads
        nt = nnunet_get_num_cpu_threads(num_cpu_threads)
        print(f"Using {nt} CPU threads for nnUNet training.")
        if torch.get_num_threads() != nt:
            torch.set_num_threads(nt)
        return torch.device('cpu')
    elif device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        return torch.device(device)
    elif device == 'mps':
        return torch.device('mps')
    else:
        raise ValueError(f"Unsupported device specified: {device}. Please use 'cpu', 'mps', or 'cuda'.")
    
    
