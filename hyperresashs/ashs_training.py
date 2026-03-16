import os
from os.path import join
from .utils.upsample_inr_method import create_link
from .utils.upsample_linear_method import linear_isotropic_upsampling, pad_image_with_world_alignment_in_memory
from .utils.trim_neck import trim_neck_in_memory
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
from typing import Protocol, Dict, Literal, Type, Callable, Any, List
from .ashs_exp import ASHSExperimentBase
from .ashs_preproc import ASHSProcessor, Timer, SegmentationLabelMap
from .utils.tool import copy_or_link_file, nnunet_configure_device, nnunet_get_num_cpu_threads
import pandas as pd
import re
from importlib.resources import files
from argparse import Namespace
import sys
import json
from tqdm import tqdm
from sklearn.model_selection import KFold
from . import __version__

def process_manifest(manifest_csv: str) -> pd.DataFrame:
    """
    Validate the manifest CSV file for training preprocessing, replacing 
    relative paths with absolute paths, and return a DataFrame.
    
    Raises an error if the manifest is invalid
    
    Parameters:
    -----------
    manifest_csv : str
        Path to the manifest CSV file to validate.
    
    The manifest CSV should have the following columns:
        - 'id': Unique identifier for each case (e.g., subject ID)
        - 'date': [Optional] date for the case (e.g., scan date)
        - 'tse': Path to the TSE image for the case
        - 'mprage': Path to the MPRAGE image for the case
        - 'seg_left': Path to the left hemisphere segmentation for the case
        - 'seg_right': Path to the right hemisphere segmentation for the case
    """
    required_columns = {'id', 'tse', 'mprage', 'seg_left', 'seg_right'}
    
    df = pd.read_csv(manifest_csv)
    
    # Check for required columns
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Generate a no-date column
    if 'date' in df.columns:
        df['date'] = np.where(pd.isna(df.date), 'nodate', df.date)
    else:
        df['date'] = 'nodate'
    
    # Check that file paths exist as either an absolute path or relative to the manifest file
    manifest_dir = os.path.dirname(manifest_csv)
    def fix_filename(path):
        if not os.path.isabs(path):
            path = os.path.join(manifest_dir, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File specified in manifest .csv does not exist': {path}")
        return path
        
    for col in ['tse', 'mprage', 'seg_left', 'seg_right']:
        df[col] = df[col].apply(fix_filename)
        
    # Set the index
    df = df.set_index(['id','date'])
        
    print("Manifest validation successful.")
    return df


class HyperASHSTraining:
    """
    This class is used to organize HyperASHS model training 
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing settings for the ASHS processing and training. 
        This should include at least the 'FILE_NAME_CONFIG' key pointing to the YAML file with filename settings.
    manifest_file : str
        Path to the manifest CSV file. See below
    output_dir : str
        Directory where the preprocessed data will be saved. The function will create subdirectories for each case.
    overwrite_existing : bool, optional
        Whether to overwrite existing preprocessed files if they already exist. Default is False.
    save_intermediates : bool, optional
        Whether to save intermediate files generated during preprocessing steps. Default is False.
    create_links : bool, optional
        Whether to create symbolic links to the original files in the working directory instead of copying them. Default is True.

    Manifest file format:
    ---------------
    The manifest file should be a CSV with the following columns:
        - 'id': Unique identifier for each case (e.g., subject ID)
        - 'date': [Optional] date for the case (e.g., scan date)
        - 'tse': Path to the TSE image for the case
        - 'mprage': Path to the MPRAGE image for the case
        - 'seg_left': Path to the left hemisphere segmentation for the case
        - 'seg_right': Path to the right hemisphere segmentation for the case
        
    Output directory structure:
    ---------------
    The function will create a subdirectory for each case in the output directory, with the following structure:
        output_dir/
            preproc/
                subject1/
                    date1/
                        Preprocessed files for subject1 date1 (e.g., registered TSE, MPRAGE, segmentations) 
            inr_training/
                subject1_date1/
                    Files prepared for INR training (e.g., cropped TSE, MPRAGE, segmentations)
                    
    """
    
    def __init__(self, config: Dict[str, Any], manifest_file: str, label_file: str, xval_file: str, 
                 output_dir: str, 
                 overwrite_existing=False, save_intermediates=False, create_links=True):
        self.config = config.copy()
        self.manifest_file = manifest_file
        self.output_dir = output_dir
        self.stats_dir = join(output_dir, 'final', 'stats')
        self.overwrite_existing = overwrite_existing
        self.save_intermediates = save_intermediates
        self.create_links = create_links
        self.label_file = label_file
        self.labels = SegmentationLabelMap(fn_itksnap_labels=label_file)
        self.xval_file = xval_file
        
        # Complete the config with missing keys. This ensures that the keys that have to be exported 
        # along with the atlas are included in the config, even if the user did not explicitly specify them.
        # This is so that if in future versions defaults change, we still have the old defaults for the atlases 
        # that were trained with those defaults.
        self._complete_config()

        # TODO: this is redundant with the ASHSInference class. Consider refactoring to avoid this redundancy.
        # Should really load the namespace and pass as part of the config
        with open(config['FILE_NAME_CONFIG']) as f:
            settings = yaml.safe_load(f)
            self.nm = SimpleNamespace(**settings)
            
        # Load the manifest file and validate it, replacing relative paths with absolute paths
        self.df = process_manifest(manifest_file)

        # Define top-level folders for preprocessed data and INR training data
        self.dir_preproc = join(output_dir, 'preproc')
        self.dir_inr_training = join(output_dir, 'inr_training')
        
        # Extract the dataset id for nnUNet
        self.nnunet_dsid = 'Dataset{}_{}'.format(config['EXP_NUM'], config['MODEL_NAME'])
        self.nnunet_trainer = config.get('TRAINER', 'ModAugUNetTrainer')
        self.nnunet_trid = f'{self.nnunet_trainer}__nnUNetPlans__3d_fullres'
        
        # Number of CPU threads to use for nnunet
        self.nnunet_threads = nnunet_get_num_cpu_threads(int(config.get('NNUNET_NUM_THREADS', 8)))
        
        # Set up the final directories
        os.makedirs(self.stats_dir, exist_ok=True)

        # Set up the directories for nnUNet
        self.dir_nnunet_base = join(self.output_dir, 'nnunet_training')
        self.dir_nnunet = {k:join(self.dir_nnunet_base, f'nnUNet_{k}') for k in ['raw', 'preprocessed', 'results']}
        for k, d in self.dir_nnunet.items():
            os.makedirs(d, exist_ok=True)
            os.environ[f'nnUNet_{k}'] = os.path.abspath(d)
        
        # Initialize the experiment dictionary to store the ASHSExperimentBase objects for each case
        self.d_exp = {}
        self.d_exp_by_side = {}
        for i, (key,row) in enumerate(self.df.iterrows()):
            
            # This is to avoid PyLance typing errors
            subject, date = key if isinstance(key, tuple) else (str(key), 'nodate')
            
            # Create a folder in the output directory for this case
            case_path = join(self.dir_preproc, subject, date)
            inr_path_map = {side: join(self.dir_inr_training, f'{subject}_{date}_{side}') for side in ['left', 'right']}
            os.makedirs(case_path, exist_ok=True)
            
            # Create the ASHS experiment representation
            self.d_exp[(subject,date)] = ASHSExperimentBase(self.config, case_path, self.nm, 
                                                            subject=subject, date=date, 
                                                            inr_path=inr_path_map,
                                                            nnunet_train_id={'left': i*2, 'right': i*2+1})
            
            # Also store the experiments by side for easy access during INR training
            for side in ['left', 'right']:
                self.d_exp_by_side[(subject, date, side)] = self.d_exp[(subject,date)]
            
        
    # Make sure all the keys that are required for inference are included in the config    
    def _complete_config(self):
        
        # Check that all the required keys are present
        defaults = {
            'EXP_NUM': None,
            'MODEL_NAME': None,
            'UPSAMPLING_METHOD': 'INRUpsampling',
            'TRAINER': 'ModAugUNetTrainer',
            'CONDITION': 'in_vivo',
            'ASHS_TSE_REGION_CROP': ASHSProcessor.get_config_defaults()['ASHS_TSE_REGION_CROP'],
        }
        
        for key, default in defaults.items():
            if key not in self.config:
                if default is not None:
                    self.config[key] = default
                else:
                    raise ValueError(f'Missing required config key: {key}')
            
    def _filter_as_int(self, filter:str|None) -> int:
        if filter is not None:
            try:
                return int(filter)
            except:
                pass
        return -1
            
    
    def _filter_cases_by_subject(self, filter=None):
        filter_idx = self._filter_as_int(filter)
        for k, ((subject, date), exp) in enumerate(self.d_exp.items()):
            if filter_idx >= 0:
                if k == filter_idx:
                    yield ((subject, date), exp)
                    return
            else:
                case_id = f'{subject}_{date}' if date != 'nodate' else f'{subject}'
                if filter is not None and not re.search(filter, case_id):
                    continue
                yield ((subject, date), exp)
            
    
    def _filter_cases_by_side(self, filter=None):
        # The filter string may be an index
        filter_idx = self._filter_as_int(filter)
        for k, ((subject, date, side), exp) in enumerate(self.d_exp_by_side.items()):
            if filter_idx >= 0:
                if k == filter_idx:
                    yield ((subject, date, side), exp)
                    return
            else:
                case_id = f'{subject}_{date}_{side}' if date != 'nodate' else f'{subject}_{side}'
                if filter is not None and not re.search(filter, case_id):
                    continue
                yield ((subject, date, side), exp)


    def preprocess(self, filter=None):
        """
        Execute the preprocessing steps for each case in the manifest file, including 
        registration, neck trimming, and preparation for INR training.
        """ 
        print('-' * 60)
        print('HyperASHS Train Stage 1: Preprocessing and registration')
        print('-' * 60)
        
        # Create a preprocessing/registration worker
        reg = ASHSProcessor(self.config, training_mode=True,
                            overwrite_existing=self.overwrite_existing, 
                            save_intermediates=self.save_intermediates, 
                            create_links=self.create_links) 
        
        # Perform initial processing steps for each case (registration, INR preprocessing, and nnUNet preprocessing)
        d_filter = dict(self._filter_cases_by_subject(filter))
        for i, ((subject, date), exp) in enumerate(d_filter.items()):

            print('=' * 40)
            print(f'Preprocessing case {i+1}/{len(d_filter)}: {subject} - {date}')
            print('=' * 40)

            # Link or copy the input files to the working directory folder
            for col, dest in [('mprage', exp.gpe.t1_native), 
                              ('tse', exp.gpe.t2_whole_img),
                              ('seg_left', exp.lpe['left'].input_seg),
                              ('seg_right', exp.lpe['right'].input_seg)]:
                copy_or_link_file(self.df.loc[(subject, date), col], dest.filename, 
                                  create_links=self.create_links, force_overwrite=self.overwrite_existing, 
                                  relative_links=False)

            # Execute the registration and preprocessing steps (neck trimming, global and local registration, ROI cropping)
            reg.preprocess(exp)
            reg.prepare_inr(exp)
            
    def train_inr(self, filter=None, device='cuda', random_seed:int|None=None, batch_size:int|None=None, num_epochs:int|None=None):
        """
        Train the INR model for each case in the manifest file using the preprocessed data.
        """ 
        # Import the INR trainig code
        from .submodules.multi_contrast_inr.main import main as inr_main
        from .submodules.multi_contrast_inr.main import parse_args as inr_parse_args
        
        # Print a banner
        print('-' * 60)
        print('HyperASHS Train Stage 2: INR segmentation upsampling')
        print('-' * 60)
        
        # Read the template YAML file
        config_temp = files('hyperresashs').joinpath('config_templates/config_inr_template.yaml')
        with config_temp.open('r') as f:
            config = yaml.safe_load(f)
            
        # Set common fields in the config file
        config["SETTINGS"]["DIRECTORY"] = self.dir_inr_training
        config["SETTINGS"]["NUM_WORKERS"] = self.nnunet_threads
        config["MODEL"]["MODEL_CLASS"] = 'MLPv2WithEarlySeg'
        config["TRAINING"]["EPOCHS"] = 60
        
        # Settings for INR are taken from parameter, if absent then config
        batch_size = batch_size or self.config.get('INR_BATCH_SIZE', None)
        if batch_size is not None:
            config["TRAINING"]["BATCH_SIZE"] = batch_size
            
        num_epochs = num_epochs or self.config.get('INR_EPOCHS', None)
        if num_epochs is not None:
            config["TRAINING"]["EPOCHS"] = num_epochs
            
        random_seed = random_seed or self.config.get('INR_RANDOM_SEED', None)
        if random_seed is not None:
            config["TRAINING"]["SEED"] = random_seed
        
        # Run INR for each subject
        d_filter = dict(self._filter_cases_by_side(filter))           
        for i, ((subject, date, side), exp) in enumerate(d_filter.items()):
            case_id = f'{subject}_{date}_{side}'
            lp = exp.lpe[side]
                
            # Check if the file can be skipped because the output already exists
            if self.overwrite_existing is False and exp.lpe[side].t2_patch_hyperres_seg.exists():
                print(f'Skipping case {case_id} because output already exists and overwrite_existing=True')
                continue
            

            print('=' * 40)
            print(f'Training INR for case {i+1}/{len(d_filter)}: {subject} - {date} - {side}')
            
            # Create a config for this case        
            inr_work_dir = lp.dir_inr_train_input     
            inr_result_dir = join(inr_work_dir, 'result')   
            config["DATASET"]["SUBJECT_ID"] = case_id
            config["SETTINGS"]["SAVE_PATH"] = inr_result_dir
            
            # If overwriting, force dataset regeneration in case something has changed
            if self.overwrite_existing:
                shutil.rmtree(join(inr_result_dir, 'preprocessed_data'), ignore_errors=True)
                        
            # Write the config file
            inr_config = join(inr_work_dir, 'config.yaml')
            with open(inr_config, 'w') as f:
                yaml.safe_dump(config, f, sort_keys=False)
                            
            saved_args = sys.argv
            sys.argv = ['test', '--config', inr_config]

            # Time the INR
            with Timer() as tm_inr:            
                inr_main(inr_parse_args())
                
            sys.argv = saved_args
            print(f'INR training completed for case {case_id} in {tm_inr.total:.1f} seconds.')                
            
            # Finally, copy the trained model to the case folder and perform any necessary cleanup.
            # Sicne the INR ROI does not have the same full context as the ROI defined based on the 
            # template, the segmentaton needs to be resampled to the original ROI space.
            
            # Find the INR output - take the latest file as we may have run with different seeds.
            inr_result_img_dir = join(inr_result_dir, 'images', case_id, config["MODEL"]["MODEL_CLASS"])
            last_epoch = config["TRAINING"]["EPOCHS"]-1
            inr_files = sorted(os.scandir(inr_result_img_dir), key=lambda x: x.stat().st_mtime)
            fn_inr_final_seg = join(inr_result_img_dir, [x.name for x in inr_files if x.name.endswith(f'e{last_epoch}__seg.nii.gz')][-1])
            
            # Resample the INR output back to the original space of the T2 hyper-resolution patch
            # In the process, also extract the single largest connected component
            # TODO: this might not work for some atlases!!!
            c3d = Convert3D()
            c3d.push(lp.t2_patch_hyperres.data)      # hyper_primary
            c3d.execute(f"{fn_inr_final_seg} -as S -dup -thresh 1 inf 1 0 -comp -thresh 1 1 1 0 -times -int 0 -reslice-identity")
            lp.t2_patch_hyperres_seg.data = c3d.peek(-1)   
            
            # Compute overlap between the upsampled segmentation and what was input
            c3d.push(lp.input_seg.data)
            c3d.execute('-push S -int 0 -reslice-identity')
            ovl = sitk.LabelOverlapMeasuresImageFilter()
            ovl.Execute(sitk.Cast(lp.input_seg.data, sitk.sitkInt16), sitk.Cast(c3d.peek(-1), sitk.sitkInt16))
            dice = { label: ovl.GetDiceCoefficient(label) for label in self.labels.label_ids }
            
            # Write the overlap to a file
            with open(join(inr_result_img_dir, 'inr_lr_overlap.json'), 'wt') as f:
                json.dump({'label_dice': dice, 'total_dice': ovl.GetDiceCoefficient()}, f, indent=4, sort_keys=False)
            
            print(f'Dice Coefficient between input and upsampled segmentation for case {case_id}: {ovl.GetDiceCoefficient():.4f}')
            print('=' * 40)
            
            
    def validity_check_inr_results(self) -> bool:
        d_filter = dict(self._filter_cases_by_side())      
        n_failed = 0     
        for i, ((subject, date, side), exp) in enumerate(d_filter.items()):
            case_id = f'{subject}_{date}_{side}'
            lp = exp.lpe[side]
            
            # Check that the main output exists
            if not lp.t2_patch_hyperres_seg.exists():
                print(f'INR result missing for case {case_id}: {lp.t2_patch_hyperres_seg.filename}')
                n_failed += 1
                continue
            
            # Check that the overlap file exists
            inr_result_img_dir = join(lp.dir_inr_train_input, 'result', 'images', case_id, 'MLPv2WithEarlySeg')
            ovl_file = join(inr_result_img_dir, 'inr_lr_overlap.json')
            if not os.path.exists(ovl_file):
                print(f'INR overlap file missing for case {case_id}')
                n_failed += 1
                continue
            
            # Check that the dice coefficient is above a reasonable threshold (e.g., 0.95)
            with open(ovl_file, 'r') as f:
                ovl_data = json.load(f)
                total_dice = ovl_data.get('total_dice', 0)
                if total_dice < 0.95:
                    print(f'INR upsampling does not match input segmentation for case {case_id}: {total_dice:.4f}. '
                          f'Try rerunning stage 2 (-s 2) for this case (-F {case_id}) with a different random seed (-R).')
                    n_failed += 1
                    
        if n_failed == 0:
            print('Validity check for INR stage successful')
            return True
        else:
            print(f'Validity check for INR stage found {n_failed} failed cases.')
            return False


    def _make_nnunet_dataset_json(self, fn_dataset_json):
        
        # Count total number of training datasets.
        d_filter = dict(self._filter_cases_by_side())           
        max_nnunet_id = -1
        for i, ((subject, date, side), exp) in enumerate(d_filter.items()):
            lp = exp.lpe[side]
            max_nnunet_id = max(max_nnunet_id, lp.nnunet_train_id)
            
        # Get the names of the modalities. We will just use the scheme that Yue
        # set up, i.e., "0": "0000", etc
        channel_names = { f'{i}': f'{i:04d}' for i in range(2) }
        
        # The complete JSON content
        with open(fn_dataset_json, 'w') as f:
            json.dump({
                'channel_names': channel_names,  
                'labels': self.labels.to_nnunet_dict_with_contiguous_labels(),
                'numTraining': max_nnunet_id+1,
                'file_ending': ".nii.gz"
                }, f, indent=4, sort_keys=False)
            
        print(f"nnUNet dataset.json file created at {fn_dataset_json} with {max_nnunet_id+1} training cases.")
        
        
    def _make_xval_splits(self, fn_xval_splits):
        
        # Read the subject ids from the xval file. Each row is a fold, each column is a subject
        # whose data will be held out in that fold.
        unique_subjects = list(set([key[0] for key in self.df.index]))
        subj_splits = []
        if self.xval_file is not None:
            with open(self.xval_file, 'r') as f:
                for line in f:
                    subj_splits.append(line.strip().split(' '))
        else:
            # If not specified, we just are going to split based on the subject ids, not
            # accounting for the fact that some subjects have multiple dates. So this means
            # that folds might be unequal in size. 
            random_state = self.config.get('XVAL_RANDOM_STATE', 42)
            kf = KFold(n_splits=self.config.get('XVAL_NUM_FOLDS', 5), shuffle=True, random_state=random_state)
            for f_train, f_test in kf.split(unique_subjects):
                subj_splits.append([unique_subjects[x] for x in f_test])
        
        # Based on the subjects splits, assign individual nnUnet inputs to splits
        nnunet_splits = [ {'train':[], 'val':[]} for x in subj_splits ]
        for ((subject, date, side), exp) in self._filter_cases_by_side():
            for i, subj_split in enumerate(subj_splits):
                nnunet_id = f'MTL_{exp.lpe[side].nnunet_train_id:03d}'
                if subject in subj_split:
                    nnunet_splits[i]['val'].append(nnunet_id)
                else:
                    nnunet_splits[i]['train'].append(nnunet_id)
                    
        # Generate the split file in the nnUNet preprocessed directory
        with open(fn_xval_splits, 'w') as f:
            json.dump(nnunet_splits, f, indent=4)
            
        print(f"nnUNet cross-validation splits file created at {fn_xval_splits} with {len(subj_splits)} folds.")
    
        
    def _nnunet_plan_and_preprocess(self):
        from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
        
        # Read the number of threads
        nt = self.nnunet_threads
        print(f"Using {nt} threads for nnUNet planning and preprocessing.")
        
        # --- step 1: figerprints ---
        expno = int(self.config['EXP_NUM'])
        extract_fingerprints([expno], 'DatasetFingerprintExtractor', nt, True, False, False)

        # --- step 2: generate the plan ---
        preproc = self.config.get('NNUNET_PREPROSSOR', 'DefaultPreprocessor')
        plans_identifier = plan_experiments([expno], 'ExperimentPlanner', nt, preproc, None, None)
        if plans_identifier is None:
            raise ValueError('Planning failed, no plans identifier returned. Please check the previous steps for errors.')

        # --- step 3: run preprocess ---
        configurations = ['3d_fullres']
        np = [(nt+1)//2]
        print('Preprocessing...')
        preprocess([expno], plans_identifier, configurations, np, False)
        
    
    def _nnunet_run(self, fold:int, device: torch.device):
        from nnunetv2.run.run_training import run_training        
        run_training(self.nnunet_dsid, '3d_fullres', fold, self.nnunet_trainer, 
                     device=device, continue_training = not self.overwrite_existing)
    
    
    def prepare_nnunet(self):
        
        # Print a banner
        print('-' * 60)
        print('HyperASHS Stage 3: nnUNet training data preparation')
        print('-' * 60)
        
        # Create the 'imagesTr' and 'labelsTr' folders in    the preprocessed directory
        raw_dir = join(self.dir_nnunet['raw'], self.nnunet_dsid)
        for what in ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs']:
            os.makedirs(join(raw_dir, what), exist_ok=True)
            
        # Create the dataset.json file
        fn_dataset_json = join(raw_dir, 'dataset.json')
        self._make_nnunet_dataset_json(fn_dataset_json)
        
        # Export each experiment into its own folder in a sequentially numbered format
        d_filter = dict(self._filter_cases_by_side()) 
        
        # Use tqdm to track progress:
        for ((subject, date, side), exp) in tqdm(d_filter.items(), desc="Organizing nnUNet training data"):
            lp = exp.lpe[side]

            # Collect list of source and destinations paths for nnUNet training
            copy_path_list = []
            copy_path_list.append(
                ('t2', lp.t2_patch_hyperres, 
                 join(raw_dir, 'imagesTr', f'MTL_{lp.nnunet_train_id:03d}_0000.nii.gz')))
            copy_path_list.append(
                ('t1', lp.t1_patch_warped_hyperres, 
                 join(raw_dir, 'imagesTr', f'MTL_{lp.nnunet_train_id:03d}_0001.nii.gz')))
            copy_path_list.append(
                ('seg', lp.t2_patch_hyperres_seg,
                 join(raw_dir, 'labelsTr', f'MTL_{lp.nnunet_train_id:03d}.nii.gz')))

            # Check if existing results can be used to skip processing
            if not self.overwrite_existing and all(os.path.exists(dest) for _, _, dest in copy_path_list):
                continue

            # Copy left sides, flip right sides
            for what, src, dest in copy_path_list:
                c3d = Convert3D()
                c3d.push(src.data)                
                if side == 'right':
                    c3d.execute(f"-swapdim RPI -flip x -o {dest}")
                else:
                    c3d.execute(f"-swapdim RPI -o {dest}")
                    
        # If overwriting, erase contents of preproc and result directories
        if self.overwrite_existing:
            shutil.rmtree(join(self.dir_nnunet['preprocessed'], self.nnunet_dsid), ignore_errors=True)
            shutil.rmtree(join(self.dir_nnunet['results'], self.nnunet_dsid), ignore_errors=True)
                    
        # Execute the planning code
        self._nnunet_plan_and_preprocess()
                    
        # Create the cross-validation splits file
        fn_xval_splits = join(self.dir_nnunet['preprocessed'], self.nnunet_dsid, 'splits_final.json')
        self._make_xval_splits(fn_xval_splits)
        

    def train_nnunet(self, filter=None, device: str = 'auto'):
        """
        Train the nnUNet model. The filter can be used to specify a subset of folds.
        """
        fn_xval_splits = join(self.dir_nnunet['preprocessed'], self.nnunet_dsid, 'splits_final.json')
        with open(fn_xval_splits, 'r') as f:
            splits = json.load(f)
            
        actual_device = nnunet_configure_device(device, self.nnunet_threads)
        print(f"Using device {actual_device} for nnUNet training.")

        for fold in range(len(splits)):
            if filter is not None and not re.search(filter, f'fold{fold}'):
                continue
            
            print('=' * 60)
            print(f'HyperResASHS Stage 4: Training nnUNet for fold {fold}')
            print('=' * 60)
            
            with Timer() as tm_fold:            
                self._nnunet_run(fold, device=actual_device)
                print(f'Fold {fold} training completed in {tm_fold.total:.1f} seconds.')
                
                
    def _nnunet_get_fold_details(self):
        
        # Check that all the folds have been generated
        fn_xval_splits = join(self.dir_nnunet['preprocessed'], self.nnunet_dsid, 'splits_final.json')
        with open(fn_xval_splits, 'r') as f:
            splits = json.load(f)
        
        fold_details = []
        for fold, fold_split in enumerate(splits):    
            fold_details.append(SimpleNamespace(
                train_ids = fold_split['train'], val_ids = fold_split['val'],
                checkpoint_final = join(self.dir_nnunet['results'], self.nnunet_dsid, self.nnunet_trid, f'fold_{fold}', 'checkpoint_final.pth'),
                validation_summary = join(self.dir_nnunet['results'], self.nnunet_dsid, self.nnunet_trid, f'fold_{fold}', 'validation', 'summary.json')))
            
        return fold_details
                
                
    def validity_check_nnunet_results(self) -> bool:
            
        fold_details = self._nnunet_get_fold_details()
        n_failed = 0
        d_val_dice = {}
        for fold, fd in enumerate(fold_details):
            fold_dir = join(self.dir_nnunet['results'], self.nnunet_dsid, self.nnunet_trid, f'fold_{fold}')
            
            # Check that the checkpoint and summary files exist
            if not os.path.exists(fd.checkpoint_final):
                print(f'Missing trained nnUNet model for fold {fold} at {fd.checkpoint_final}')
                n_failed += 1
                continue
            if not os.path.exists(fd.validation_summary):
                print(f'Missing validation summary for fold {fold} at {fd.validation_summary}')
                n_failed += 1
                continue
            
            # Load the Dice scores from the summary file  
            with open(fd.validation_summary, 'r') as f:
                summary = json.load(f)
                for i, nid in enumerate(fd.val_ids):
                    d_val_dice[nid] = summary['metric_per_case'][i]['metrics']
        
        if n_failed > 0:
            print(f'Validity check for nnUNet training stage found {n_failed}/{len(fold_details)} incomplete folds')  
            return False  
        
        # Compute table of summary Dice measures
        c_labels = [label['name'] for id, label in self.labels.labels.items() if id > 0 ]
        d_val_dice_for_df = { k:[] for k in ['id', 'date', 'side'] + c_labels }
        d_filter = dict(self._filter_cases_by_side()) 
        for ((subject, date, side), exp) in d_filter.items():
            lp = exp.lpe[side]
            nnunet_id = f'MTL_{lp.nnunet_train_id:03d}'
            val_dice = d_val_dice[nnunet_id]
            d_val_dice_for_df['id'].append(subject)
            d_val_dice_for_df['date'].append(date)
            d_val_dice_for_df['side'].append(side)
            for label_id, label in self.labels.labels.items():
                if label_id > 0:
                    d_val_dice_for_df[label['name']].append(val_dice[f'{label_id}']['Dice'])
        df_val_dice = pd.DataFrame(d_val_dice_for_df)
        df_val_dice['mean'] = df_val_dice[c_labels].mean(1)
        df_val_dice.to_csv(join(self.stats_dir, 'nnunet_validation_dice.csv'), index=False)
        
        # Create a summary table of Dice by label using pandas wide to tall and aggregating by label
        df_val_dice_long = df_val_dice.melt(id_vars=['id', 'date', 'side'], value_vars=c_labels, var_name='label', value_name='dice')
        df_val_dice_summary = df_val_dice_long.groupby('label')['dice'].agg(['mean', 'std', 'min', 'max'])
        df_val_dice_summary.to_csv(join(self.stats_dir, 'nnunet_validation_dice_summary.csv'))        
        
        # Print warning if any Dice scores are very low (< 0.1)
        n_warn = 0
        for i, row in df_val_dice.iterrows():
            if row['mean'] < 0.1:
                print(f"Warning: Case {row['id']}_{row['date']}_{row['side']} has low mean Dice score of {row['mean']:.4f} across all labels.")
                n_warn += 1

        if n_warn > 0:
            print(f'Validity check for nnUNet completed with {n_warn} warnings.')
        else:
            print(f'Validity check for nnUNet training stage successful.')
            
        return True
    
                
    def finalize(self, full_metadata: Dict[str, Any]):
        """
        Finalize the training by copying the trained nnUNet models to the case folders and performing any necessary cleanup.
        """ 
        print('-' * 60)
        print('HyperASHS Train Stage 5: Finalization and packaging')
        print('-' * 60)
        
        # Copy the folds to the target destination
        fold_details = list(self._nnunet_get_fold_details())
        atlas_dir = join(self.output_dir, 'final', 'atlas')
        for fold, fd in tqdm(enumerate(fold_details), desc="Copying nnUNet checkpoints to final atlas"):
            fold_dir = join(atlas_dir, self.nnunet_trid, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            copy_or_link_file(fd.checkpoint_final, join(fold_dir, 'checkpoint_final.pth'), 
                              create_links=False, force_overwrite=True, quiet=True)
            
        # Copy required .json files
        for fn in ['dataset.json', 'plans.json']:
            fn_src = join(self.dir_nnunet['results'], self.nnunet_dsid, self.nnunet_trid, fn)
            fn_dst = join(atlas_dir, self.nnunet_trid, fn)
            copy_or_link_file(fn_src, fn_dst, create_links=False, force_overwrite=True, quiet=True)             
        
        # Write the ITK-SNAP labels file to the final directory
        self.labels.export_itksnap_label_file(join(atlas_dir, 'itksnap_labels.txt'))
        
        # Create a .gitattributes file to specify that .pth files should be treated as large files if using git-lfs
        with open(join(atlas_dir, '.gitattributes'), 'w') as f:
            f.write('*.pth filter=lfs diff=lfs merge=lfs -text\n')
            
        # Write out the atlas.yaml file with the complete metadata
        full_metadata = full_metadata.copy()  
        
        # Ensure relevant config keys for inference are in the metadata, while the other fields
        # that are only relevant for training (e.g., local directories) are not includes
        full_metadata['config'] = {
            key: self.config[key] for key in ['EXP_NUM', 'MODEL_NAME', 'UPSAMPLING_METHOD', 
                                              'TRAINER', 'CONDITION', 'ASHS_TSE_REGION_CROP']
        }
            
        # Write the software version information to the metadata
        full_metadata['hyperresashs'] = {'version': __version__}
        
        # Write the metadata to a YAML file in the atlas directory
        with open(join(atlas_dir, 'atlas.yaml'), 'w') as f:
            yaml.safe_dump(full_metadata, f, sort_keys=False)
            
        # Write an dummy README.md file to the atlas directory
        with open(join(atlas_dir, 'README.md'), 'w') as f:
            f.write('---')
            f.write('license: cc-by-nc-4.0')
            f.write('---')
            
        # Print what we have done!
        print(f'Finalized atlas saved to {atlas_dir}')
                    
            
                    
            
            
                    
            
            
            
            
            
            
                
                         
            
