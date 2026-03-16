import os
from os.path import join
from .utils.upsample_inr_method import create_link
from .utils.upsample_linear_method import linear_isotropic_upsampling, pad_image_with_world_alignment
from picsl_greedy import Greedy3D
from picsl_c3d import Convert3D
from .utils.trim_neck import trim_neck
import yaml
from types import SimpleNamespace
from batchgenerators.utilities.file_and_folder_operations import *
import torch
import time
import shutil
import tempfile


class ModelTester():
    def __init__(self, config):
        self.config = config

        with open(config['FILE_NAME_CONFIG']) as f:
            settings = yaml.safe_load(f)
            self.nm = SimpleNamespace(**settings)

        self.test_path = config['TEST_PATH']
        self.test_folder = 'Dataset{}_{}'.format(config['EXP_NUM'], config['MODEL_NAME'])
        self.upsampling_method = config['UPSAMPLING_METHOD']
        self.dataset_id = config['EXP_NUM']
        self.trainer = config['TRAINER']

        # ROI determination (template path)
        self.template_3tt1 = join(config['TEMPLATE_PATH'], self.nm.template)
        self.template_roi_left = join(config['TEMPLATE_PATH'], self.nm.left_roi_file)
        self.template_roi_right = join(config['TEMPLATE_PATH'], self.nm.right_roi_file)

    def download_model_from_huggingface(self, hf_repo_id, target_path):
        try:
            from huggingface_hub import snapshot_download
            import requests
        except ImportError:
            raise ImportError("huggingface_hub is not installed. Please install it with: pip install huggingface_hub")
        
        print(f"Model not found locally. Downloading from Hugging Face: {hf_repo_id}")
        
        os.makedirs(target_path, exist_ok=True)
        temp_dir = tempfile.mkdtemp()
        try:
            print(f"Downloading model to temporary location: {temp_dir}")
            
            disable_ssl = os.environ.get('HF_HUB_DISABLE_SSL_VERIFY', '0').lower() in ('1', 'true', 'yes')
            original_request = None
            if disable_ssl:
                print("warning: ssl verification is disabled (hf_hub_disable_ssl_verify=1)")
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                original_request = requests.Session.request
                def patched_request(self, method, url, **kwargs):
                    kwargs['verify'] = False
                    return original_request(self, method, url, **kwargs)
                requests.Session.request = patched_request
            
            try:
                downloaded_path = snapshot_download(
                    repo_id=hf_repo_id,
                    local_dir=temp_dir,
                    repo_type="model"
                )
            except Exception as ssl_error:
                error_str = str(ssl_error).lower()
                if 'ssl' in error_str or 'certificate' in error_str or 'cert' in error_str:
                    if original_request:
                        requests.Session.request = original_request
                    raise RuntimeError(
                        f"ssl certificate verification failed. this often happens in corporate networks with proxies.\n"
                        f"to disable ssl verification (less secure), set environment variable:\n"
                        f"  export HF_HUB_DISABLE_SSL_VERIFY=1\n"
                        f"then run your command again.\n"
                        f"original error: {str(ssl_error)}"
                    )
                raise
            finally:
                if original_request:
                    requests.Session.request = original_request
            
            downloaded_contents = os.listdir(temp_dir)
            
            model_folder = None
            for item in downloaded_contents:
                item_path = join(temp_dir, item)
                if os.path.isdir(item_path):
                    if os.path.exists(join(item_path, 'dataset.json')) and os.path.exists(join(item_path, 'plans.json')):
                        model_folder = item_path
                        break
            
            if model_folder is None:
                if os.path.exists(join(temp_dir, 'dataset.json')) and os.path.exists(join(temp_dir, 'plans.json')):
                    model_folder = temp_dir
                else:
                    raise ValueError(f"Could not find model structure in downloaded repository. Contents: {downloaded_contents}")
            
            parent_dir = os.path.dirname(target_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            for item in os.listdir(model_folder):
                src = join(model_folder, item)
                dst = join(target_path, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            
            required_files = ['dataset.json', 'plans.json']
            for req_file in required_files:
                if not os.path.exists(join(target_path, req_file)):
                    raise ValueError(f"Required file {req_file} not found in downloaded model")
            
            fold_found = False
            for item in os.listdir(target_path):
                if item.startswith('fold_') and os.path.isdir(join(target_path, item)):
                    if os.path.exists(join(target_path, item, 'checkpoint_final.pth')):
                        fold_found = True
                        break
            
            if not fold_found:
                raise ValueError("No fold checkpoints found in downloaded model")
            
            print(f"Successfully downloaded and installed model to: {target_path}")
            
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to download model from Hugging Face ({hf_repo_id}): {str(e)}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def resample_test_with_date(self, subject_id=None):
        subject_list = os.listdir(self.test_path)
        if subject_id:
            if subject_id not in subject_list:
                raise ValueError(f'subject_id {subject_id} not found in test path {self.test_path}')
            subject_list = [subject_id]
        
        for subject_ in subject_list:
            subject_path = join(self.test_path, subject_)
            date_list = os.listdir(subject_path)

            for date_ in date_list:
                date_path = join(subject_path, date_)

                print(subject_ + '_' + date_)
                self.run_inference_for_one_case(date_path)
    
    def check_roi_existence(self, case_path):
        existence_flag = 1
        for side_ in ['left', 'right']:
            t1_roi_path = join(case_path, f'trim_{side_}_roi_in_3tt1.nii.gz')
            t2_roi_path = join(case_path, f'trim_{side_}_roi_in_3tt2.nii.gz')
            if not os.path.exists(t1_roi_path) or not os.path.exists(t2_roi_path):
                existence_flag = 0
        return existence_flag
    
    def trim_neck_for_original_3tt1(self, case_path):
        input_image = join(case_path, self.nm.t1_whole_img)
        output_image = join(case_path, self.nm.t1_name_after_triming_neck)
        trim_neck(input_image, output_image)
        print('finish trimming neck in 3T-T1')
    
    def create_roi(self, case_path):
        template_to_3tt1_rigid_matrix = join(case_path, self.nm.rigid_matrix)
        template_to_3tt1_affine_matrix = join(case_path, self.nm.affine_matrix)
        template_to_3tt1_deformable_field = join(case_path, self.nm.deformable_field)
        template_to_3tt1_inverse_deformable_field = join(case_path, self.nm.deformable_field_inverse)
        trim_t1_image = join(case_path, self.nm.t1_name_after_triming_neck)

        g = Greedy3D()

        # 1. rigid
        g.execute('-d 3 -threads 24 -a -dof 6 -m NCC 2x2x2 '
                  f"-i {self.template_3tt1} {trim_t1_image} "
                  f"-o {template_to_3tt1_rigid_matrix} -n 400x0x0x0 "
                  "-ia-image-centers -search 400 5 5")
        
        # 2. affine
        g.execute('-d 3  -threads 24 -a -m NCC 2x2x2 '
                  f'-i {self.template_3tt1} {trim_t1_image} '
                  f'-o {template_to_3tt1_affine_matrix} -n 400x80x40x0 '
                  f'-ia {template_to_3tt1_rigid_matrix}')
        
        # 3. run it first
        intermidate_output_before_deformable = join(case_path, 'roi_step_t1_to_template_before_deformable.nii.gz')
        g.execute(f'-d 3  -threads 24 -r {template_to_3tt1_affine_matrix} -rf {self.template_3tt1} '
                  f'-rm {trim_t1_image} {intermidate_output_before_deformable}')

        # 4. deformable
        g.execute('-d 3  -threads 24 -m NCC 2x2x2 -e 0.5 -n 60x20x0 '
                  f'-i {self.template_3tt1} {intermidate_output_before_deformable} '
                  f'-o {template_to_3tt1_deformable_field} '
                  f'-oinv {template_to_3tt1_inverse_deformable_field}')
        
        # 5. apply
        g.execute(f"-d 3  -threads 24 -rm {trim_t1_image} {join(case_path, 'roi_step_3tt1_to_template.nii.gz')} "
                  f'-rf {self.template_3tt1} -r {template_to_3tt1_deformable_field} {template_to_3tt1_affine_matrix}')
        
        # map the ROI to the T1 image
        g.execute(f"-d 3  -threads 24 -rf {trim_t1_image} "
                  f"-rm {self.template_3tt1} {join(case_path, self.nm.template_to_3tt1)} "
                  f"-rm {self.template_roi_left} {join(case_path, self.nm.global_roi_in_3tt1_XYZ.replace('XYZ', 'left'))} "
                  f"-rm {self.template_roi_right} {join(case_path, self.nm.global_roi_in_3tt1_XYZ.replace('XYZ', 'right'))} "
                  f"-r {template_to_3tt1_affine_matrix},-1 {template_to_3tt1_inverse_deformable_field}")

        # delete large files
        os.remove(template_to_3tt1_deformable_field)
        os.remove(template_to_3tt1_inverse_deformable_field)
    
    def cropping(self, case_path):
        for side_ in ['left', 'right']:
            # crop 3tt1
            t1_roi = join(case_path, self.nm.global_roi_in_3tt1_XYZ.replace('XYZ', side_))
            t1_whole_img = join(case_path, self.nm.t1_whole_img)

            t1_local_patch = join(case_path, self.nm.trim_roi_in_3tt1_XYZ.replace('XYZ', side_))
            c3d = Convert3D()
            c3d.execute(f'{t1_roi} -trim 5vox {t1_whole_img} -reslice-identity -o {t1_local_patch}')

            # crop 3tt2
            t2_padded_img = join(case_path, self.nm.t2_padded_img)
            t2_local_patch = join(case_path, self.nm.trim_roi_in_3tt2_XYZ.replace('XYZ', side_))
            c3d = Convert3D()
            c3d.execute(f'{t2_padded_img} {t1_roi} -reslice-identity -trim 5vox {t2_padded_img} -reslice-identity -o {t2_local_patch}')

    def run_inference_for_one_case(self, case_path):
        # create the folder for hyper-resolution inference
        hyper_test_path = join(case_path, self.test_folder)
        os.makedirs(hyper_test_path, exist_ok=True)

        if self.check_roi_existence(case_path) == 0:
            # ------- global registration (from T1 to T2) new added pipeline -------
            t2_whole_img = join(case_path, self.nm.t2_whole_img)
            t1_whole_img_before_registration = join(case_path, self.nm.t1_native_img)

            save_mat_path_t2_to_t1_global = join(case_path, 'global_matrix_3tt2_to_3tt1.mat')
            g = Greedy3D()
            g.execute(f'-d 3  -threads 24 -a -m NMI -i {t1_whole_img_before_registration} {t2_whole_img} -dof 6 -o {save_mat_path_t2_to_t1_global} -ia-identity -n 100x50')

            t1_registered_path = join(case_path, self.nm.t1_whole_img)
            g = Greedy3D()
            g.execute(f'-d 3  -threads 24 -rf {t1_whole_img_before_registration} -rm {t1_whole_img_before_registration} {t1_registered_path} -r {save_mat_path_t2_to_t1_global},-1')

            # ------- extract ROI -------
            self.trim_neck_for_original_3tt1(case_path)
            self.create_roi(case_path)
            t2_padded_img = join(case_path, self.nm.t2_padded_img)
            pad_image_with_world_alignment(t2_whole_img, t2_padded_img, [40, 40, 40], [40, 40, 40])
            self.cropping(case_path)

        # iterate through the sides
        side_list = ['left', 'right']
        for side_ in side_list:
            start = time.time()  # start counting the time

            side_path = join(hyper_test_path, side_)
            nnunet_input_folder = join(side_path, 'input')
            os.makedirs(nnunet_input_folder, exist_ok=True)

            # ------- resampling -------
            primary_file = join(case_path, self.nm.trim_roi_in_3tt2_XYZ.replace('XYZ', side_))
            secondary_file = join(case_path, self.nm.trim_roi_in_3tt1_XYZ.replace('XYZ', side_))

            # image path in resmapling workspace
            primary_target = join(side_path, self.nm.primary)
            secondary_target = join(side_path, self.nm.secondary)
            c3d = Convert3D()
            c3d.execute(f'{primary_file} -swapdim RPI -o {primary_target}')

            c3d = Convert3D()
            c3d.execute(f'{secondary_file} -swapdim RPI -o {secondary_target}')
            primary_upsampled = join(side_path, self.nm.hyper_primary)
            secondary_upsampled = join(side_path, self.nm.hyper_secondary)
            
            if self.upsampling_method == 'INRUpsampling' or self.upsampling_method == 'GreedyUpsampling':
                linear_isotropic_upsampling(primary_target, secondary_target, primary_upsampled, secondary_upsampled)

            elif self.upsampling_method == 'None':
                create_link(primary_target, primary_upsampled)
                create_link(secondary_target, secondary_upsampled)

            # ------- registration -------
            target_file = join(side_path, self.nm.hyper_primary)
            moving_file = join(side_path, self.nm.hyper_secondary)
            save_mat_path = os.path.join(side_path, self.nm.reg_mat)
            output_file_path = os.path.join(side_path, self.nm.hyper_secondary_after_registertion)

            g = Greedy3D()
            g.execute(f"-d 3  -threads 24 -a -m NMI -i {target_file} {moving_file} -dof 6 -o {save_mat_path} -ia-identity -n 100x50")

            g = Greedy3D()
            g.execute(f'-d 3  -threads 24 -rf {target_file} -rm {moving_file} {output_file_path} -r {save_mat_path}')

            # ------- nnunet input -------
            create_link(target_file, join(nnunet_input_folder, 'MTL_000_0000.nii.gz'))
            create_link(output_file_path, join(nnunet_input_folder, 'MTL_000_0001.nii.gz'))

            # ------- nnunet inference -------
            # command
            print(f'start running inference for {side_path}')
            nnunt_output_path = join(side_path, 'output')

            # nnUNet prediction
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            from batchgenerators.utilities.file_and_folder_operations import load_json
            from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            predictor = nnUNetPredictor(verbose=True, device=device)
            nnunet_model = join(os.environ.get("nnUNet_results"), f"Dataset{self.config['EXP_NUM']}_{self.config['MODEL_NAME']}", f"{self.config['TRAINER']}__nnUNetPlans__3d_fullres")
            
            dataset_json_path = join(nnunet_model, 'dataset.json')
            plans_json_path = join(nnunet_model, 'plans.json')
            model_complete = (os.path.exists(dataset_json_path) and 
                            os.path.exists(plans_json_path))
            
            if model_complete:
                fold_folders = [item for item in os.listdir(nnunet_model) 
                               if item.startswith('fold_') and os.path.isdir(join(nnunet_model, item))]
                if fold_folders:
                    for fold_folder in fold_folders:
                        checkpoint_path = join(nnunet_model, fold_folder, 'checkpoint_final.pth')
                        if not os.path.exists(checkpoint_path):
                            model_complete = False
                            break
                else:
                    model_complete = False
            
            if not model_complete:
                hf_repo_id = self.config.get('HF_MODEL_REPO')
                if hf_repo_id:
                    print(f"Model not found or incomplete at {nnunet_model}, downloading from Hugging Face...")
                    self.download_model_from_huggingface(hf_repo_id, nnunet_model)
                else:
                    raise FileNotFoundError(
                        f"Model not found or incomplete at {nnunet_model} and HF_MODEL_REPO not specified in config. "
                        f"Please either:\n"
                        f"  1. Place the trained model at the expected path, or\n"
                        f"  2. Add HF_MODEL_REPO to your config file to enable automatic download from Hugging Face."
                    )
            
            use_folds = predictor.auto_detect_available_folds(nnunet_model, 'checkpoint_final.pth')
            dataset_json = load_json(join(nnunet_model, 'dataset.json'))
            plans = load_json(join(nnunet_model, 'plans.json'))
            plans_manager = PlansManager(plans)

            parameters = []
            for i, f in enumerate(use_folds):
                f = int(f) if f != 'all' else f
                checkpoint = torch.load(join(nnunet_model, f'fold_{f}', 'checkpoint_final.pth'),
                                        map_location=torch.device('cpu'))
                if i == 0:
                    configuration_name = checkpoint['init_args']['configuration']
                    inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                        'inference_allowed_mirroring_axes' in checkpoint.keys() else None

                parameters.append(checkpoint['network_weights'])

            configuration_manager = plans_manager.get_configuration(configuration_name)

            # restore network
            num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
            network = nnUNetTrainer.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                enable_deep_supervision=False
            )

            predictor.manual_initialization(
                network, plans_manager, configuration_manager, parameters,
                dataset_json, self.config['TRAINER'],
                inference_allowed_mirroring_axes)

            # predictor.initialize_from_trained_model_folder(nnunet_model, None)
            predictor.predict_from_files(
                nnunet_input_folder,
                nnunt_output_path,
                save_probabilities=False,
                overwrite=True, num_processes_preprocessing=1, num_processes_segmentation_export=1)

            end = time.time()  # end counting the time
            elapsed_time = end - start
            with open(join(nnunt_output_path, "elapsed_time.txt"), "w") as f_:
                f_.write(str(elapsed_time))

    def execute(self, subject_id=None):
        # --- run inference
        self.resample_test_with_date(subject_id=subject_id)