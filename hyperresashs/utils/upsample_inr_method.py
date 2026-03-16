from os.path import join
import os
import shutil
import SimpleITK as sitk
from picsl_greedy import Greedy3D

def copy_inr_upsample_seg(case_path, upsampling_path, names):
    case_name = case_path.split('/')[-1]
    upsampling_case_seg_path = join(upsampling_path, case_name, 'images', case_name, 'MLPv2WithEarlySeg')
    files = os.listdir(upsampling_case_seg_path)

    # ----- copy the segmentation -----
    inr_seg_img_path = join(upsampling_case_seg_path, [x for x in files if 'e59__seg.nii.gz' in x][0])
    shutil.copyfile(inr_seg_img_path, join(case_path, names.inr_hyper_primary_seg))

    # ----- copy the t2 image -----
    inr_t2_img_path = join(upsampling_case_seg_path, [x for x in files if 'e59__ct2.nii.gz' in x][0])
    shutil.copyfile(inr_t2_img_path, join(case_path, names.inr_hyper_primary_img))

    print(f'copied the {case_name}')


def create_link(source_path, target_path):
    if os.path.lexists(target_path):  # covers symlinks and real files
        if os.path.islink(target_path):
            os.unlink(target_path)  # remove the symlink only
        else:
            os.remove(target_path)
    link_target = os.path.realpath(source_path) if os.path.islink(source_path) else source_path
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    os.symlink(link_target, target_path)


def copy_inr_linear_image(case_path, inr_preliminary, names):
    case_name = case_path.split('/')[-1]
    case_preliminary_folder = join(inr_preliminary, case_name)

    t2_path = join(case_preliminary_folder, f"{case_name}_t2.nii.gz")
    t1_path = join(case_preliminary_folder, f"{case_name}_t1.nii.gz")

    create_link(t2_path, join(case_path, names.hyper_primary))
    create_link(t1_path, join(case_path, names.hyper_secondary))


def correct_shift(case_path, registration_param, names):
    primary_linear_img_path = join(case_path, names.hyper_primary)
    inr_t2_img_path = join(case_path, names.inr_hyper_primary_img)
    inr_seg_img_path = join(case_path, names.inr_hyper_primary_seg)

    inr_linear_reg_matrix = join(case_path, names.inr_to_linear_reg_matrix)
    shift_corrected_t2_path = join(case_path, names.inr_hyper_primary_img_shift_corrected)
    hyper_seg_path = join(case_path, names.hyper_primary_seg)

    if registration_param == 'None':
        create_link(inr_t2_img_path, shift_corrected_t2_path)
        create_link(inr_seg_img_path, hyper_seg_path)
        return

    elif registration_param == 'rigid':
        g = Greedy3D()
        g.execute(f"-d 3 -a -m NCC 2x2x2 -i {primary_linear_img_path} {inr_t2_img_path} -dof 6 -o {inr_linear_reg_matrix} -ia-image-centers -n 100x50x10")

    elif registration_param == 'affine':
        g = Greedy3D()
        g.execute(f"-d 3 -a -m NCC 2x2x2 -i {primary_linear_img_path} {inr_t2_img_path} -o {inr_linear_reg_matrix} -ia-image-centers -n 100x50x10")

    else:
        raise

    g = Greedy3D()
    g.execute(f"-d 3 -rf {primary_linear_img_path} -rm {inr_t2_img_path} {shift_corrected_t2_path} -ri LABEL 0.2vox -rm {inr_seg_img_path} {hyper_seg_path} -r {inr_linear_reg_matrix}")