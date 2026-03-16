import os
from os.path import join
from .ashs_exp import ASHSExperimentBase
from .ashs_preproc import ASHSProcessor, generate_ashs_segmentation_qc, ProgressCallbackType, default_progress_callback, Timer, SegmentationLabelMap
from .utils.tool import copy_or_link_file, nnunet_configure_device
import yaml
from types import SimpleNamespace
import torch
import time
import numpy as np
import shutil
import tempfile
import pandas as pd
import SimpleITK as sitk

class HyperASHSInference():
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
        self.nnunet_trid = f"{self.trainer}__nnUNetPlans__3d_fullres"
        self.nnunet_model = join(self.config.get('ATLAS_PATH'), self.nnunet_trid)
        self.dataset_json_path = join(self.nnunet_model, 'dataset.json')
       
        # Number of threads for Greedy
        self.greedy_num_threads = config.get('GREEDY_NUM_THREADS', 0)
        
        # Read ITK-SNAP label mapping for visualization, either from the ITK-SNAP label file
        # provided in the model (optional) or from the dataset.json of the nnUNet model.
        self.labelset = SegmentationLabelMap(
            fn_itksnap_labels=config.get('ITKSNAP_LABEL_FILE'), 
            fn_dataset_json_file=self.dataset_json_path)


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

                self.run_inference_for_one_case(date_path, subject=subject_, date=date_)
    

    def run_inference_for_one_case(self, mprage:str, tse:str|None, case_path:str, 
                                   subject:str|None=None, date:str|None=None,
                                   save_intermediates: bool = True, overwrite_existing: bool = False,
                                   create_links: bool = True,
                                   callback: ProgressCallbackType = default_progress_callback, 
                                   device:str = 'auto'):
        
        # Do we have a TSE at inference? If not, we switch to the T1-ASHS pipeline, which substitutes
        # an empty image for the TSE and performs other operations slightly differently.
        t1_only = tse is None
        
        # Create the ASHS experiment representation
        exp = ASHSExperimentBase(self.config, case_path, self.nm, subject=subject, date=date, 
                                 prefix=f'{subject}_{date}_' if (subject and date) else f'{subject}_' if subject else '')

        # Create a preprocessing/registration worker
        reg = ASHSProcessor(self.config, t1_only=t1_only,
                            overwrite_existing=overwrite_existing, 
                            save_intermediates=save_intermediates, 
                            create_links=create_links) 

        # Create timers for all elements of the pipeline            
        tm_all, tm_nnunet = Timer(), Timer()
        with tm_all:
            
            # Copy or link the raw input images
            for dst, src in [(exp.gpe.t1_native, mprage), (exp.gpe.t2_whole_img, tse)]:        
                if src is not None:
                    copy_or_link_file(src, dst.filename, create_links=create_links, force_overwrite=overwrite_existing, relative_links=False)
            
            # Execute the registration and preprocessing steps (neck trimming, global and local registration, ROI cropping)
            print('--- HyperResASHS Stage 1: Neck Trim and Registration ---')
            reg.preprocess(exp, callback=callback, progress_range=(0.0, 0.25))

            # ------- nnunet inference -------
            with tm_nnunet:
                torch_device = nnunet_configure_device(device, int(self.config.get('NNUNET_NUM_THREADS', 8)))
                for i_side_, (side_, lp) in enumerate(exp.lpe.items()):

                    # command
                    print(f'--- HyperResASHS Stage 2-{side_}: nnUNet Inference ---')
                    start = time.time()

                    # nnUNet prediction
                    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
                    from batchgenerators.utilities.file_and_folder_operations import load_json
                    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
                    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
                    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

                    # Number of CPU threads to use for nnunet
                    predictor = nnUNetPredictor(verbose=True, device=torch_device)
                    use_folds = predictor.auto_detect_available_folds(self.nnunet_model, 'checkpoint_final.pth')
                    dataset_json = load_json(join(self.nnunet_model, 'dataset.json'))
                    plans = load_json(join(self.nnunet_model, 'plans.json'))
                    plans_manager = PlansManager(plans)

                    parameters = []
                    configuration_name = None
                    inference_allowed_mirroring_axes = None
                    for i, f in enumerate(use_folds):
                        f = int(f) if f != 'all' else f
                        checkpoint = torch.load(join(self.nnunet_model, f'fold_{f}', 'checkpoint_final.pth'),
                                                map_location=torch.device('cpu'), weights_only=False)
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
                    print(f'Predicting from {lp.dir_nnunet_input} using model {self.nnunet_model} with configuration {configuration_name} and folds {use_folds}...')
                    predictor.predict_from_files(
                        lp.dir_nnunet_input,
                        lp.dir_nnunet_output,
                        save_probabilities=False,
                        overwrite=True, num_processes_preprocessing=1, num_processes_segmentation_export=1)

                    end = time.time()  # end counting the time
                    elapsed_time = end - start
                    with open(join(lp.dir_nnunet_output, "elapsed_time.txt"), "w") as f_:
                        f_.write(str(elapsed_time))
                        
                    # Copy the nnUNet output segmentation to the final segmentation location
                    # (links are a bad idea because user might want to delete the nnUNet output folder)
                    copy_or_link_file(lp.nnunet_seg.filename, lp.t2_seg_hyperres.filename, 
                                      create_links=False, force_overwrite=True, create_dir=True)
                        
                    # Generate a QC screenshot for the nnUNet output
                    generate_ashs_segmentation_qc(
                        seg = lp.nnunet_seg.data,
                        t1 = lp.t1_patch_warped_hyperres.data,
                        t2 = lp.t2_patch_hyperres.data,
                        labelset = self.labelset,
                        output_path = lp.fn_segmentation_qc,
                        t1_only=t1_only,
                        title = f"{exp.qc_title} Segmentation QC - {side_.capitalize()}")
                        
                    # Report being done with part of the nnUNet processing 
                    callback(progress=0.5 * (i_side_+1), progress_range=(0.25, 0.75),
                             message=f"nnUNet Inference for {side_} completed in {elapsed_time:.1f} s.",
                             attachments={f'{side_.capitalize()} Segmentation QC': lp.fn_segmentation_qc})
            
            # ------- final post-processing -------
            print('--- HyperResASHS Stage 3: Post-Processing ---')
            reg.postprocess(exp, callback=callback, progress_range=(0.95, 1.0))
            
            # Compute and export CSV of structures in the segmentations
            vols = { 'side': [], 'subfield': [], 'volume': [] }
            for side_, lp in exp.lpe.items():
                seg = sitk.GetArrayFromImage(lp.nnunet_seg.data)
                voxel_size = np.prod(lp.nnunet_seg.data.GetSpacing())
                for label_id, label_data in self.labelset.labels.items():
                    if label_id != 0:
                        vols['side'].append(side_)
                        vols['subfield'].append(label_data['name'])
                        vols['volume'].append(np.sum(seg == label_id) * voxel_size)
            df_vols = pd.DataFrame(vols)
            if exp.subject:
                if exp.date:
                    df_vols.insert(0, 'date', exp.date)        
                df_vols.insert(0, 'subject', exp.subject)
                
            df_vols.to_csv(exp.gpe.fn_final_volumes_csv, index=False)
            
            # Save the label mapping used for this case (for visualization in ITK-SNAP)
            self.labelset.export_itksnap_label_file(exp.gpe.fn_final_labelfile)
                    
        print(f'ASHS inference completed for case: {case_path}')
        print(f'  Time in neck trimming: {reg.tm_neck.total:.2f}s')
        print(f'  Time in global T1/T2 reg: {reg.tm_reg_t1_t2_whole.total:.2f}s')
        print(f'  Time in T1/template reg: {reg.tm_reg_t1_temp.total:.2f}s')
        print(f'  Time in local T1/T2 reg: {reg.tm_reg_t1_t2_local.total:.2f}s')
        print(f'  Time in nnUNet: {tm_nnunet.total:.2f}s')
        print(f'  Total time: {tm_all.total:.2f}s')

    def execute(self, subject_id=None):
        # --- run inference
        self.resample_test_with_date(subject_id=subject_id)