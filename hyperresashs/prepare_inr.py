import SimpleITK as sitk
from os.path import join
from picsl_c3d import Convert3D
import os
from pathlib import Path
import numpy as np
import yaml
from types import SimpleNamespace
from .utils.upsample_inr_method import create_link
from picsl_greedy import Greedy3D


def resample_using_auto_adjusted_spacing(image, new_spacing, interpolator=sitk.sitkLinear):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    c3d = Convert3D()
    c3d.push(image)
    c3d.execute(f"-resample {new_size[0]}x{new_size[1]}x{new_size[2]}")
    resampled_image = c3d.peek(-1)

    return resampled_image


def resample_using_ref(image, ref_img, interpolator=sitk.sitkLinear):
    # this function supposes two images are aligned with each other well
    # resample one array to another's image size
    # ref image's info is copied to the new upsampled image
    new_spacing = ref_img.GetSpacing()
    new_size = ref_img.GetSize()

    c3d = Convert3D()
    c3d.push(image)
    c3d.execute(f"-resample {new_size[0]}x{new_size[1]}x{new_size[2]}")
    resampled_image = c3d.peek(-1)

    return resampled_image


def register_t1_without_resampling(t1_path, t2_path, t1_registered_path):
    # make intermediate files
    parent_path = Path(t1_path).parent
    t2_fake_path = join(parent_path, 'fake_t2_in_t1_resolution.nii.gz')
    registration_mat = join(parent_path, 'registration_matrix.mat')

    # create fake t2 resolution image
    t2_image = sitk.ReadImage(t2_path)
    t1_image = sitk.ReadImage(t1_path)

    new_resolution = t1_image.GetSpacing()
    resampled_image = resample_using_auto_adjusted_spacing(t2_image, new_resolution)  # to keep the same bounding box, the voxel spacing must be adjusted
    sitk.WriteImage(resampled_image, t2_fake_path)

    # register t1 to t2 but keep its own resolution
    g = Greedy3D()
    g.execute(f'-d 3 -a -m NMI -i {t2_path} {t1_path} -dof 6 -o {registration_mat} -ia-identity -n 100x50')

    g = Greedy3D()
    g.execute(f"-d 3 -rf {t2_fake_path} -rm {t1_path} {t1_registered_path} -r {registration_mat}")


def make_hr_gt(t1_registered_path, t2_path, t1_gt_path, t2_gt_path):
    # t2
    t2_image = sitk.ReadImage(t2_path)
    t2_original_spacing = t2_image.GetSpacing()
    new_common_spacing = (t2_original_spacing[0], t2_original_spacing[0], t2_original_spacing[2])  # isotropic
    t2_gt_image = resample_using_auto_adjusted_spacing(t2_image, new_common_spacing)
    sitk.WriteImage(t2_gt_image, t2_gt_path)

    # t1
    registered_t1_image = sitk.ReadImage(t1_registered_path)
    t1_gt_image = resample_using_ref(registered_t1_image, t2_gt_image)  # this step is different with before
    sitk.WriteImage(t1_gt_image, t1_gt_path)


def make_supervision_mask(t1_registered_path, t1_mask_registered_path, t2_path, t2_mask):
    # t1: the tilted mask
    ct1_image = sitk.ReadImage(t1_registered_path)
    ct1_array = sitk.GetArrayFromImage(ct1_image)
    ct1_all_one_array = (ct1_array > 0) + 0
    ct1_all_one_image = sitk.GetImageFromArray(ct1_all_one_array)
    ct1_all_one_image.CopyInformation(ct1_image)
    sitk.WriteImage(ct1_all_one_image, t1_mask_registered_path)

    # t1: whole image mask
    ct2_image = sitk.ReadImage(t2_path)
    ct2_array = sitk.GetArrayFromImage(ct2_image)
    ct2_all_one_array = np.zeros(ct2_array.shape) + 1
    ct2_all_one_image = sitk.GetImageFromArray(ct2_all_one_array)
    ct2_all_one_image.CopyInformation(ct2_image)
    sitk.WriteImage(ct2_all_one_image, t2_mask)


def make_gt_mask(t1_gt_path, t2_gt_path, mask_gt_path):
    # the whole brain mask: find the intersection between ct1 and ct2
    ct1_gt_image = sitk.ReadImage(t1_gt_path)
    ct1_gt_array = sitk.GetArrayFromImage(ct1_gt_image)
    ct1_gt_all_one_array = (ct1_gt_array > 0) + 0

    ct2_gt_image = sitk.ReadImage(t2_gt_path)
    ct2_gt_array = sitk.GetArrayFromImage(ct2_gt_image)
    ct2_gt_all_one_array = np.zeros(ct2_gt_array.shape) + 1

    brainmask_array = ct1_gt_all_one_array * ct2_gt_all_one_array
    brainmask_image = sitk.GetImageFromArray(brainmask_array)
    brainmask_image.CopyInformation(ct2_gt_image)
    sitk.WriteImage(brainmask_image, mask_gt_path)


def create_slink(input_path, target_path):
    if os.path.islink(target_path):
        os.unlink(target_path)
    os.symlink(input_path, target_path)


def excuate_one_case(data_path, exp_path, case_name, names):
    # input files
    t2_path = join(data_path, names.inr_primary)
    t1_path = join(data_path, names.inr_secondary)
    t2_seg_path = join(data_path, names.inr_primary_seg)

    # registration files
    t1_registered_path = join(data_path, 'registered_secondary.nii.gz')

    # masks
    t1_mask_registered_path = join(data_path, 'registered_secondary_mask.nii.gz')
    t2_mask = join(data_path, 'primary_mask.nii.gz')

    # gt files
    t2_gt_path = join(data_path, 'primary_gt.nii.gz')
    t1_gt_path = join(data_path, 'secondary_gt.nii.gz')
    mask_gt_path = join(data_path, 'gt_mask.nii.gz')

    # registration t1
    register_t1_without_resampling(t1_path, t2_path, t1_registered_path)

    # make fake gt
    make_hr_gt(t1_registered_path, t2_path, t1_gt_path, t2_gt_path)

    # create mask using segmentation (common subregions)
    make_supervision_mask(t1_registered_path, t1_mask_registered_path, t2_path, t2_mask)
    make_gt_mask(t1_gt_path, t2_gt_path, mask_gt_path)

    # export to exp folder
    case_path = join(exp_path, case_name)
    os.makedirs(case_path, exist_ok=True)

    create_slink(t2_path, join(case_path, f'{case_name}_t2_LR.nii.gz'))
    create_slink(t2_seg_path, join(case_path, f'{case_name}_t2_seg_LR.nii.gz'))
    create_slink(t2_mask, join(case_path, f'{case_name}_t2_mask_LR.nii.gz'))

    create_slink(t1_registered_path, join(case_path, f'{case_name}_t1_LR.nii.gz'))
    create_slink(t1_mask_registered_path, join(case_path, f'{case_name}_t1_seg_LR.nii.gz'))  # fake t1 seg for completing input
    create_slink(t1_mask_registered_path, join(case_path, f'{case_name}_t1_mask_LR.nii.gz'))

    create_slink(t2_gt_path, join(case_path, f'{case_name}_t2.nii.gz'))
    create_slink(t1_gt_path, join(case_path, f'{case_name}_t1.nii.gz'))
    create_slink(mask_gt_path, join(case_path, f'{case_name}_brainmask.nii.gz'))


class INRPreprocess():
    def __init__(self, config):
        self.config = config

        with open(config['FILE_NAME_CONFIG']) as f:
            settings = yaml.safe_load(f)
            self.nm = SimpleNamespace(**settings)

        exp_name = str(config['EXP_NUM']) + config['MODEL_NAME']

        self.data_path = join(config['PREPARE_RAW_PATH'], exp_name, 'images')

        self.inr_processing_path = join(config['INR_PATH'], exp_name, self.nm.INR_PREPROCESSING)
        self.preliminary_path = join(config['INR_PATH'], exp_name, self.nm.INR_PRELIMINARY_PATH)
        self.upsampling_path = join(config['INR_PATH'], exp_name, self.nm.INR_UPSAMPLING_PATH)

        # special setting
        self.model_name = 'MLPv2WithEarlySeg'
        self.config_name = 'config.yaml'
        self.template_config = 'config_inr/template.yaml'
        self.case_list = os.listdir(self.data_path)
    
    def make_inr_folders(self):
        case_list = self.case_list
        for case_ in case_list:
            case_path = join(self.data_path, case_)

            # set three important files
            t2_file = join(case_path, self.nm.primary)
            t1_file = join(case_path, self.nm.secondary)
            seg_file = join(case_path, self.nm.seg)

            target_case_folder = join(self.inr_processing_path, case_)
            os.makedirs(target_case_folder, exist_ok=True)
            target_t2_file = join(target_case_folder, self.nm.inr_primary)
            target_t1_file = join(target_case_folder, self.nm.inr_secondary)
            target_seg_file = join(target_case_folder, self.nm.inr_primary_seg)

            # copy corresponding image
            if os.path.islink(target_t2_file):
                os.unlink(target_t2_file)
            create_link(t2_file, target_t2_file)

            if os.path.islink(target_t1_file):
                os.unlink(target_t1_file)
            create_link(t1_file, target_t1_file)

            if os.path.islink(target_seg_file):
                os.unlink(target_seg_file)
            create_link(seg_file, target_seg_file)
    
    def make_config(self, case_name, data_directory, save_path):
        with open(self.template_config, "r") as f:
            config = yaml.safe_load(f)

        if "SETTINGS" in config:
            config["SETTINGS"]["DIRECTORY"] = data_directory
            config["SETTINGS"]["SAVE_PATH"] = join(self.upsampling_path, case_name)
        
        if "DATASET" in config:
            config["DATASET"]["SUBJECT_ID"] = case_name
        
        if "MODEL" in config:
            config["MODEL"]["MODEL_CLASS"] = self.model_name
        
        if "TRAINING" in config:
            config["TRAINING"]["EPOCHS"] = 60

        with open(save_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    
    def make_inr_data(self):
        case_list = self.case_list
        for case_ in case_list:
            print(case_)
            case_path = join(self.inr_processing_path, case_)
            excuate_one_case(case_path, self.preliminary_path, case_, self.nm)  # make image data
            self.make_config(case_, self.preliminary_path, join(self.inr_processing_path, case_, self.config_name))  # make config
    
    def create_inr_script(self):
        """Create the shell script for running INR upsampling with correct paths from config"""
        # Read template file
        template_path = join(Path(__file__).parent, 'shell', 'run_inr_upsampling_template.sh')
        
        if not os.path.exists(template_path):
            print(f'Warning: Template file not found: {template_path}')
            return
        
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Get the project root directory and submodule path
        project_root = Path(__file__).parent
        inr_repo_path = join(str(project_root), 'submodules', 'multi_contrast_inr')
        
        # Replace placeholders with actual values from config
        script_content = template_content.replace('{INR_PATH}', self.config['INR_PATH'])
        script_content = script_content.replace('{EXP_NUM}', str(self.config['EXP_NUM']))
        script_content = script_content.replace('{MODEL_NAME}', self.config['MODEL_NAME'])
        script_content = script_content.replace('{INR_REPO_PATH}', str(inr_repo_path))
        
        # Create shell directory if it doesn't exist
        shell_dir = join(Path(__file__).parent, 'shell')
        os.makedirs(shell_dir, exist_ok=True)
        
        # Write the script with experiment name in filename
        exp_name = str(self.config['EXP_NUM']) + self.config['MODEL_NAME']
        script_filename = f'run_inr_upsampling_{exp_name}.sh'
        script_path = join(shell_dir, script_filename)
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make it executable
        os.chmod(script_path, 0o755)
        
        print(f'Created INR upsampling script: {script_path}')
    
    def run_inr_upsampling(self):
        """run INR upsampling for all cases"""
        import subprocess
        
        exp_name = str(self.config['EXP_NUM']) + self.config['MODEL_NAME']
        base_dir = join(self.config['INR_PATH'], exp_name, self.nm.INR_PRELIMINARY_PATH)
        
        # get project root and INR repo path
        project_root = Path(__file__).parent
        inr_repo_path = join(str(project_root), 'submodules', 'multi_contrast_inr')
        
        # get all case folders from training_preparation directory
        if not os.path.exists(base_dir):
            raise ValueError(f'Training preparation directory not found: {base_dir}. Please run prepare_inr stage first.')
        
        all_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(join(base_dir, f))])
        
        print(f'Found {len(all_folders)} total cases, processing all cases')
        
        for case_name in all_folders:
            # config file is in preprocess folder, not training_preparation
            config_path = join(self.config['INR_PATH'], exp_name, self.nm.INR_PREPROCESSING, case_name, self.config_name)
            if not os.path.exists(config_path):
                print(f'Warning: Config file not found for {case_name}: {config_path}')
                continue
            
            print(f'Running INR upsampling for: {case_name}')
            inr_main_path = join(inr_repo_path, 'main.py')
            
            # run the INR main.py with the config
            cmd = ['python', inr_main_path, '--config', config_path, '--logging']
            result = subprocess.run(cmd, cwd=inr_repo_path)
            
            if result.returncode != 0:
                print(f'Error: INR upsampling failed for {case_name}')
                raise RuntimeError(f'INR upsampling failed for {case_name} with return code {result.returncode}')
            
            print(f'Completed INR upsampling for: {case_name}')
    
    def execute(self):
        self.make_inr_folders()
        self.make_inr_data()
        self.create_inr_script()

    