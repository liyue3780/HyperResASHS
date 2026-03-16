import SimpleITK as sitk
from .tool import linear_resample_to_spacing_using_itkimage
import numpy as np


def linear_isotropic_upsampling(t2_path, t1_path, t2_output_path, t1_output_path):
    # t2
    t2_image = sitk.ReadImage(t2_path)
    t2_original_spacing = t2_image.GetSpacing()
    new_common_spacing = (t2_original_spacing[0], t2_original_spacing[0], t2_original_spacing[2])  # isotropic
    t2_upsampled_image = linear_resample_to_spacing_using_itkimage(t2_image, new_common_spacing)
    sitk.WriteImage(t2_upsampled_image, t2_output_path)

    # t1
    t1_image = sitk.ReadImage(t1_path)
    t1_upsampled_image = linear_resample_to_spacing_using_itkimage(t1_image, new_common_spacing)
    sitk.WriteImage(t1_upsampled_image, t1_output_path)


def pad_image_with_world_alignment_in_memory(image, pad_lower, pad_upper):
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    # Compute offset in physical space
    offset_vector = direction @ (np.array(pad_lower) * spacing)
    new_origin = origin - offset_vector

    # Apply zero padding
    padded = sitk.ConstantPad(image, pad_lower, pad_upper, 0)

    # Restore spatial geometry
    padded.SetSpacing(tuple(spacing))
    padded.SetOrigin(tuple(new_origin))
    padded.SetDirection(image.GetDirection())

    return padded


def pad_image_with_world_alignment(image_path, output_path, pad_lower, pad_upper):
    image = sitk.ReadImage(image_path)
    padded = pad_image_with_world_alignment_in_memory(image, pad_lower, pad_upper)
    sitk.WriteImage(padded, output_path)