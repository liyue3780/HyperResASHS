import os
import shutil
from os.path import join
from .utils.upsample_inr_method import copy_inr_upsample_seg, copy_inr_linear_image, create_link, correct_shift
from .utils.upsample_greedy_method import greedy_upsample_segmentation
from .utils.upsample_linear_method import linear_isotropic_upsampling
from .utils.tool import flip_image, save_label_mapping_to_txt, make_nnunet_dataset_json, convert_each_ground_truth_file_as_continuous
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json
import numpy as np
from pathlib import Path
from collections import OrderedDict
import yaml
from types import SimpleNamespace
import csv
from picsl_c3d import Convert3D
from picsl_greedy import Greedy3D
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess


class PreprocessorBase():
    def __init__(self, config) -> None:
        self.config = config
        self.config['NNUNET_RAW_PATH'] = os.environ.get('nnUNet_raw')
        
        with open(config['FILE_NAME_CONFIG']) as f:
            settings = yaml.safe_load(f)
            self.nm = SimpleNamespace(**settings)

        # global info
        self.model_name = str(config['EXP_NUM']) + config['MODEL_NAME']
        self.auxiliary = True

        # prepare path
        self.preparation_folder = os.path.join(config['PREPARE_RAW_PATH'], self.model_name)
        self.preparation_file_folder = os.path.join(self.preparation_folder, 'files')
        self.preparation_image_folder = os.path.join(self.preparation_folder, 'images')
        os.makedirs(self.preparation_file_folder, exist_ok=True)
        os.makedirs(self.preparation_image_folder, exist_ok=True)

        # Upsampling
        self.upsampling_method = config['UPSAMPLING_METHOD']
        inr_exp_name = str(config['EXP_NUM']) + config['MODEL_NAME']
        self.inr_preliminary_path = join(config['INR_PATH'], inr_exp_name, self.nm.INR_PRELIMINARY_PATH)
        self.inr_upsampling_path = join(config['INR_PATH'], inr_exp_name, self.nm.INR_UPSAMPLING_PATH)

        # nnunet
        self.nnunet_raw_date_path = os.path.join(self.config['NNUNET_RAW_PATH'], 'Dataset{}_{}'.format(config['EXP_NUM'], config['MODEL_NAME']))
        self.snap_label_path = config['SNAP_LABEL_PATH']
        self.cv_path = config['CV_FILE']
                    
    def resampling(self):
        case_list = os.listdir(self.preparation_image_folder)
        for case_ in case_list:
            print('{} hyper'.format(case_))
            case_path = join(self.preparation_image_folder, case_)

            if self.upsampling_method == 'INRUpsampling':
                if os.path.exists(join(self.inr_upsampling_path, case_)):
                    copy_inr_upsample_seg(case_path, self.inr_upsampling_path, self.nm)
                    copy_inr_linear_image(case_path, self.inr_preliminary_path, self.nm)
                    correct_shift(case_path, self.config['INR_CORRECTION_PARAM'], self.nm)
                else:
                    raise ValueError('Please run INR first!')

            elif self.upsampling_method == 'None':
                create_link(join(case_path, self.nm.primary), join(case_path, self.nm.hyper_primary))
                create_link(join(case_path, self.nm.secondary), join(case_path, self.nm.hyper_secondary))
                create_link(join(case_path, self.nm.seg), join(case_path, self.nm.hyper_primary_seg))
            
            elif self.upsampling_method == 'GreedyUpsampling':
                linear_isotropic_upsampling(join(case_path, self.nm.primary), join(case_path, self.nm.secondary),
                                            join(case_path, self.nm.hyper_primary), join(case_path, self.nm.hyper_secondary))
                greedy_upsample_segmentation(case_path, self.nm, s_param=0.75)
    
    def register_to_primary(self):
        case_list = os.listdir(self.preparation_image_folder)
        for case_ in case_list:
            case_path = join(self.preparation_image_folder, case_)
            target_file = join(case_path, self.nm.hyper_primary)
            moving_file = join(case_path, self.nm.hyper_secondary)
            save_mat_path = os.path.join(case_path, self.nm.auxiluary_to_primary_matrix)
            output_file_path = os.path.join(case_path, self.nm.auxiluary_to_primary_registered)

            g = Greedy3D()
            g.execute(f"-d 3 -a -m NMI -i {target_file} {moving_file} -dof 6 -o {save_mat_path} -ia-identity -n 100x50")

            g = Greedy3D()
            g.execute(f"-d 3 -rf {target_file} -rm {moving_file} {output_file_path} -r {save_mat_path}")

    def sort_case_list(self, case_list):
        case_list.sort(key=lambda x: int(x[0:3]))

    def get_id_side(self, case_name):
        case_id = case_name[0:3]
        side_ = case_name.split('_')[-1].strip('.nii.gz')
        return case_id, side_

    def prepare_nnunet(self):
        os.makedirs(os.path.join(self.nnunet_raw_date_path, 'imagesTr'), exist_ok=True)
        os.makedirs(os.path.join(self.nnunet_raw_date_path, 'imagesTs'), exist_ok=True)
        os.makedirs(os.path.join(self.nnunet_raw_date_path, 'labelsTr'), exist_ok=True)
        os.makedirs(os.path.join(self.nnunet_raw_date_path, 'labelsTs'), exist_ok=True)

        # copy the image patch (after flipping)
        case_list = os.listdir(self.preparation_image_folder)
        self.sort_case_list(case_list)
        
        nnunet_id = 1
        for case_ in case_list:
            case_id, side_ = self.get_id_side(case_)
            copy_path_list = []

            # image path
            source_primary_file = os.path.join(self.preparation_image_folder, case_, self.nm.hyper_primary)
            target_primary_file = os.path.join(self.nnunet_raw_date_path, 'imagesTr', 'MTL_' + "%03.0d" % int(nnunet_id) + '_0000.nii.gz')
            copy_path_list.append([source_primary_file, target_primary_file])

            if self.auxiliary:
                source_auxiliary_file = os.path.join(self.preparation_image_folder, case_, self.nm.auxiluary_to_primary_registered)
                target_auxiliary_file = os.path.join(self.nnunet_raw_date_path, 'imagesTr', 'MTL_' + "%03.0d" % int(nnunet_id) + '_0001.nii.gz')
                copy_path_list.append([source_auxiliary_file, target_auxiliary_file])

            # segmentation path
            source_seg = os.path.join(self.preparation_image_folder, case_, self.nm.hyper_primary_seg)
            target_seg = os.path.join(self.nnunet_raw_date_path, 'labelsTr', 'MTL_' + "%03.0d" % int(nnunet_id) + '.nii.gz')
            copy_path_list.append([source_seg, target_seg])

            # copy or flip to nnunet  # TODO: both in vivo and ex vivo
            for one_path_pair in copy_path_list:
                if side_ == 'left':
                    shutil.copyfile(one_path_pair[0], one_path_pair[1])
                elif side_ == 'right':
                    flip_image(one_path_pair[0], one_path_pair[1], 2)
            
            # write nnunet id
            with open(os.path.join(self.preparation_image_folder, case_, 'nnunet_id.txt'), 'w') as f_:
                f_.write(str(nnunet_id))
            nnunet_id = nnunet_id + 1

            print('{}: {}'.format(case_id, side_))

    def get_maximal_segmentation(self, input_path, output_path):
        seg_itk = sitk.ReadImage(input_path)
        seg_array = sitk.GetArrayFromImage(seg_itk)
        seg_mask = (seg_array > 0) + 0

        new_wm_temp_itk = sitk.GetImageFromArray(seg_mask)
        component_image = sitk.ConnectedComponent(new_wm_temp_itk)
        sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
        largest_component_binary_image = sorted_component_image == 1
        new_wm_largest_in_roi = sitk.GetArrayFromImage(largest_component_binary_image)
        masked_seg = seg_array * new_wm_largest_in_roi

        masked_seg_itk = sitk.GetImageFromArray(masked_seg)
        masked_seg_itk.SetOrigin(seg_itk.GetOrigin())
        masked_seg_itk.SetDirection(seg_itk.GetDirection())
        masked_seg_itk.SetSpacing(seg_itk.GetSpacing())
        sitk.WriteImage(masked_seg_itk, output_path)
    
    def remove_outer_seg(self):
        nnunet_label_path = os.path.join(self.nnunet_raw_date_path, 'labelsTr')
        file_list = os.listdir(nnunet_label_path)

        for file_ in file_list:
            file_path = join(nnunet_label_path, file_)
            self.get_maximal_segmentation(file_path, file_path)

    def preprocess_labels(self):
        continuous_correspondance = join(self.preparation_file_folder, 'continuous_correspondance.txt')
        save_label_mapping_to_txt(self.snap_label_path, continuous_correspondance)
        make_nnunet_dataset_json(continuous_correspondance, self.nnunet_raw_date_path)  # generate dataset.json
        convert_each_ground_truth_file_as_continuous(continuous_correspondance, self.nnunet_raw_date_path)  # change the label of files
    
    def process_cross_validation(self):
        cv_dict = load_json(self.cv_path)
        case_list = os.listdir(self.preparation_image_folder)

        nnunet_dict = {}
        for case in case_list:
            case_number = ''.join(filter(str.isdigit, case))

            # search nnunet id
            case_nnunet_id = [ele_ for ele_ in open(join(self.preparation_image_folder, case, 'nnunet_id.txt'))][0]

            # create five folds
            for fold_name, fold_content in cv_dict.items():
                if case_number in fold_content:
                    if fold_name not in nnunet_dict:
                        nnunet_dict[fold_name] = []
                    nnunet_dict[fold_name].append('MTL_{}'.format("%03.0d" % int(case_nnunet_id)))
        
        # create nnunet split
        new_nnunet_dict = [ele_ for ele_ in nnunet_dict.values()]
        final_split = []
        for ii in range(5):
            test_list = new_nnunet_dict[ii]
            train_list = []
            for jj in range(5):
                if jj != ii:
                    train_list.extend(new_nnunet_dict[jj])
            
            curr_dict = OrderedDict()
            curr_dict['train'] = train_list
            curr_dict['val'] = test_list
            final_split.append(curr_dict)
        
        save_json(final_split, join(self.nnunet_raw_date_path, 'splits_final.json'))
    
    def nnunet_plan(self):
        # --- step 1: figerprints ---
        extract_fingerprints([int(self.config['EXP_NUM'])], 'DatasetFingerprintExtractor', 8, True, False, False)

        # --- step 2: generate the plan ---
        plans_identifier = plan_experiments([int(self.config['EXP_NUM'])], 'ExperimentPlanner', 8, self.config['NNUNET_PREPROSSOR'], None, None)
        if plans_identifier is None:
            raise ValueError('Planning failed, no plans identifier returned. Please check the previous steps for errors.')

        # --- step 3: run preprocess ---
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        configurations = ['3d_fullres']
        np = [default_np[c] if c in default_np.keys() else 4 for c in configurations]
        print('Preprocessing...')
        preprocess([int(self.config['EXP_NUM'])], plans_identifier, configurations, np, False)

        # --- step 4: move cross-validation json ---
        nnunet_model_name = self.nnunet_raw_date_path.split('/')[-1]
        shutil.copyfile(join(self.nnunet_raw_date_path, 'splits_final.json'),
                        join(Path(self.config['NNUNET_RAW_PATH']).parent.parent.absolute(), 'nnUNet_preprocessed', nnunet_model_name, 'splits_final.json'))
    
    def create_nnunet_training_script(self):
        """Create the shell script for running nnUNet training with correct parameters from config"""
        # read template file
        template_path = join(Path(__file__).parent, 'shell', 'train_nnunet_template.sh')
        
        if not os.path.exists(template_path):
            print(f'Warning: Template file not found: {template_path}')
            return
        
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # replace placeholders with actual values from config
        script_content = template_content.replace('{EXP_NUM}', str(self.config['EXP_NUM']))
        script_content = script_content.replace('{TRAINER}', self.config['TRAINER'])
        
        # create shell directory if it doesn't exist
        shell_dir = join(Path(__file__).parent, 'shell')
        os.makedirs(shell_dir, exist_ok=True)
        
        # write the script with experiment name in filename
        exp_name = str(self.config['EXP_NUM']) + self.config['MODEL_NAME']
        script_filename = f'train_nnunet_{exp_name}.sh'
        script_path = join(shell_dir, script_filename)
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # make it executable
        os.chmod(script_path, 0o755)
        
        print(f'Created nnUNet training script: {script_path}')
    
    def run_nnunet_training(self):
        """run nnU-Net training for all 5 folds"""
        import subprocess
        
        exp_num = str(self.config['EXP_NUM'])
        trainer = self.config['TRAINER']
        
        print(f'Running nnU-Net training for dataset {exp_num} with trainer {trainer}')
        print(f'Training all 5 folds (0-4)')
        
        for fold in range(5):
            print(f'Training fold {fold}...')
            cmd = ['nnUNetv2_train', exp_num, '3d_fullres', str(fold), '-tr', trainer]
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                print(f'Error: nnU-Net training failed for fold {fold}')
                raise RuntimeError(f'nnU-Net training failed for fold {fold} with return code {result.returncode}')
            
            print(f'Completed training for fold {fold}')
        
        print('All folds completed successfully')

class PreprocessorInVivo(PreprocessorBase):
    def __init__(self, config):
        super().__init__(config)
        self.primary_ashs_path = config['PRIMARY_ASHS_PATH']
        self.secondary_ashs_path = config['SECOND_ASHS_PATH']
    
    def prepare_patch_data_from_ashs_package(self):
        target_path = self.preparation_image_folder
        case_list = os.listdir(self.primary_ashs_path)
        case_list.sort(key=lambda x: int(x[5:]))

        for case_ in case_list:
            case_id = case_[5:]

            for side_ in ['left', 'right']:
                curr_folder_name = case_id + '_' + side_
                curr_folder_path = os.path.join(target_path, curr_folder_name)
                os.makedirs(curr_folder_path, exist_ok=True)

                # copy image
                source_file = os.path.join(self.primary_ashs_path, case_, 'tse_native_chunk_{}.nii.gz'.format(side_))
                target_file_primary = os.path.join(curr_folder_path, self.nm.primary)
                shutil.copyfile(source_file, target_file_primary)

                c3d = Convert3D()
                c3d.execute(f"{target_file_primary} -swapdim RPI -o {target_file_primary}")

                # copy seg
                source_file = os.path.join(self.primary_ashs_path, case_, 'tse_native_chunk_{}_seg.nii.gz'.format(side_))
                target_file_seg = os.path.join(curr_folder_path, self.nm.seg)
                shutil.copyfile(source_file, target_file_seg)

                c3d = Convert3D()
                c3d.execute(f"{target_file_seg} -swapdim RPI -o {target_file_seg}")

                # copy secondary image
                source_file = os.path.join(self.secondary_ashs_path, case_, 'tse_native_chunk_{}.nii.gz'.format(side_))
                target_file_secondary = os.path.join(curr_folder_path, self.nm.secondary)
                shutil.copyfile(source_file, target_file_secondary)

                c3d = Convert3D()
                c3d.execute(f"{target_file_secondary} -swapdim RPI -o {target_file_secondary}")

                print('{}: {}'.format(case_id, side_))
    
    def execute(self):
        self.resampling()
        self.register_to_primary()
        self.prepare_nnunet()
        self.remove_outer_seg()
        self.preprocess_labels()
        self.process_cross_validation()
        self.nnunet_plan()
        self.create_nnunet_training_script()


class PreprocessorExVivo(PreprocessorBase):
    def __init__(self, config):
        super().__init__(config)
        self.whole_brain_path = config['WHOLE_BRAIN_PATH']
        self.label_path = config['LABEL_PATH']
        self.roi_path = config['CROP_ROI_PATH']
    
    def prepare_patch_data_from_exvivo_input(self):
        case_list = os.listdir(self.roi_path)
        for case_ in case_list:
            print(case_)
            case_name = case_.strip('.nii.gz')
            t2_global_path = join(self.whole_brain_path, f"{case_name}_iv_t2.nii.gz")
            case_roi_path = join(self.roi_path, case_name)

            # ------- t2 -------
            case_path = join(self.preparation_image_folder, case_name)
            os.makedirs(case_path, exist_ok=True)

            target_primary_path = join(case_path, self.nm.primary)
            c3d = Convert3D()
            c3d.execute(f"{case_roi_path} {t2_global_path} -reslice-identity -o {target_primary_path}")

            c3d = Convert3D()
            c3d.execute(f'{target_primary_path} -swapdim RPI -o {target_primary_path}')

            # ------- t1 -------
            t1_global_path = join(self.whole_brain_path, f"{case_name}_iv_t1.nii.gz")
            target_t1_roi = join(case_path, self.nm.secondary)
            c3d = Convert3D()
            c3d.execute(f"{t1_global_path} {case_roi_path} -reslice-identity {t1_global_path} -multiply -trim 5vox -o {target_t1_roi}")

            c3d = Convert3D()
            c3d.execute(f'{target_t1_roi} -swapdim RPI -o {target_t1_roi}')

            # ------- seg -------
            for file_ in os.listdir(self.label_path):
                if case_[0:6] in file_:
                    seg_file = join(self.label_path, file_)
                    target_seg_path = join(case_path, self.nm.seg)
                    
                    # read original segmentation to get unique labels
                    original_seg_itk = sitk.ReadImage(seg_file)
                    original_seg_array = sitk.GetArrayFromImage(original_seg_itk)
                    unique_labels = np.unique(original_seg_array)
                    unique_labels = unique_labels[unique_labels > 0]  # exclude background (0)
                    
                    if len(unique_labels) == 0:
                        # no labels found, create empty segmentation
                        c3d = Convert3D()
                        c3d.execute(f"{case_roi_path} {seg_file} -reslice-identity -o {target_seg_path}")
                        c3d = Convert3D()
                        c3d.execute(f'{target_seg_path} -swapdim RPI -o {target_seg_path}')
                    else:
                        # create temporary directory for binary masks
                        temp_dir = join(case_path, 'temp_binary_masks')
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        # create one-hot binary masks and reslice each
                        resliced_binary_masks = []
                        label_mapping = {}  # maps index to original label value
                        
                        for idx, label_val in enumerate(unique_labels):
                            # create binary mask for this label
                            binary_mask = (original_seg_array == label_val).astype(np.uint8)
                            binary_itk = sitk.GetImageFromArray(binary_mask)
                            binary_itk.CopyInformation(original_seg_itk)
                            
                            # save temporary binary mask
                            temp_binary_path = join(temp_dir, f'label_{idx}.nii.gz')
                            sitk.WriteImage(binary_itk, temp_binary_path)
                            
                            # reslice binary mask to match ROI
                            temp_resliced_path = join(temp_dir, f'label_{idx}_resliced.nii.gz')
                            c3d = Convert3D()
                            c3d.execute(f"{case_roi_path} {temp_binary_path} -reslice-identity -o {temp_resliced_path}")
                            
                            # read resliced binary mask
                            resliced_binary_itk = sitk.ReadImage(temp_resliced_path)
                            resliced_binary_array = sitk.GetArrayFromImage(resliced_binary_itk)
                            resliced_binary_masks.append(resliced_binary_array)
                            label_mapping[idx] = label_val
                        
                        # stack binary masks and use argmax to recover labels
                        stacked_masks = np.stack(resliced_binary_masks, axis=0)  # shape: (num_labels, H, W, D)
                        label_indices = np.argmax(stacked_masks, axis=0)  # shape: (H, W, D)
                        
                        # map indices back to original label values
                        label_map_array = np.zeros_like(label_indices, dtype=np.uint8)
                        for idx, label_val in label_mapping.items():
                            label_map_array[label_indices == idx] = label_val
                        
                        # handle background: if no mask has value > 0.5, it's background
                        max_values = np.max(stacked_masks, axis=0)
                        label_map_array[max_values < 0.5] = 0
                        
                        # create final segmentation image
                        final_seg_itk = sitk.GetImageFromArray(label_map_array)
                        final_seg_itk.CopyInformation(resliced_binary_itk)
                        sitk.WriteImage(final_seg_itk, target_seg_path)
                        
                        # swap dimensions to RPI
                        c3d = Convert3D()
                        c3d.execute(f'{target_seg_path} -swapdim RPI -o {target_seg_path}')
                        
                        # clean up temporary files
                        shutil.rmtree(temp_dir)

    def sort_case_list(self, case_list):
        case_list.sort(key=lambda x: int(x[0:6]))

    def get_id_side(self, case_name):
        case_id = case_name[0:6]
        last_letter = case_name[-1]
        if last_letter == "L":
            side_ = 'left'
        elif last_letter == 'R':
            side_ = 'right'
        return case_id, side_

    def preprocess_labels(self):
        continuous_correspondance = join(self.preparation_file_folder, 'continuous_correspondance.txt')
        label_csv = self.config['SNAP_LABEL_PATH']
        with open(label_csv, "r") as infile, open(continuous_correspondance, "w", newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            for row in reader:
                selected = [row[0], row[1], row[3]]  # 1st, 2nd, 4th columns
                writer.writerow(selected)
        
        make_nnunet_dataset_json(continuous_correspondance, self.nnunet_raw_date_path)  # generate dataset.json
        convert_each_ground_truth_file_as_continuous(continuous_correspondance, self.nnunet_raw_date_path)  # change the label of files

    def create_auxillary_channels(self, auxiliary_id):
        # ------- make the auxiliary channels from big-label model -------
        # set the path of big label gt
        # search the nnunet id full name
        nnunet_dataset_name = None
        for ele_ in os.listdir(self.config['NNUNET_RAW_PATH']):
            if 'Dataset' + str(auxiliary_id) in ele_:
                nnunet_dataset_name = ele_
        label_source = join(self.config['NNUNET_RAW_PATH'], nnunet_dataset_name, 'labelsTr')
        print(label_source)

        # set save path
        save_path = os.path.join(self.nnunet_raw_date_path, 'imagesTr')
        print(save_path)

        # read channel num from dataset.json
        auxiliary_dataset_json = join(self.config['NNUNET_RAW_PATH'], nnunet_dataset_name, 'dataset.json')
        auxiliary_label_map = load_json(auxiliary_dataset_json)["labels"]
        max_label_idx = np.array(list(auxiliary_label_map.values())).max()
        max_channel_idx = max_label_idx + 1

        # read gt files one by one and create the auxiliary channels for small label model
        gt_list = os.listdir(label_source)
        for gt_file in gt_list:
            gt_file_path = join(label_source, gt_file)
            gt_itk = sitk.ReadImage(gt_file_path)
            gt_array = sitk.GetArrayFromImage(gt_itk)

            # Use the big label (even big CA) as new channels
            for ii in range(1, max_channel_idx):
                curr_label_mask = (gt_array == ii) + 0
                mask_itk = sitk.GetImageFromArray(curr_label_mask)
                mask_itk.CopyInformation(gt_itk)
                curr_file_save_path = join(save_path, gt_file.strip(".nii.gz") + "_%04.0d" % int(ii+1) + ".nii.gz")
                sitk.WriteImage(mask_itk, curr_file_save_path)
    
        # ------- remake small-label model's dataset.json -------
        main_dataset_json = join(self.nnunet_raw_date_path, 'dataset.json')
        main_dataset_dict = load_json(main_dataset_json)
        inverted_auxiliary_label_map = {value: key for key, value in auxiliary_label_map.items()}
        for ii in range(2, max_channel_idx+1):
            main_dataset_dict['channel_names'][str(ii)] = inverted_auxiliary_label_map[ii-1]
        save_json(main_dataset_dict, main_dataset_json)

    def execute(self):
        self.resampling()
        self.register_to_primary()
        self.prepare_nnunet()
        self.remove_outer_seg()
        self.preprocess_labels()
        self.process_cross_validation()
    
    def execute_nnunet_plan(self):
        self.nnunet_plan()
        self.create_nnunet_training_script()


