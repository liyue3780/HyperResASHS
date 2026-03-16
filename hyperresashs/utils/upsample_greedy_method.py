from picsl_greedy import Greedy3D
from os.path import join
import numpy as np


def greedy_upsample_segmentation(work_space, names, s_param=0.75):
    reference_img = join(work_space, names.hyper_primary)
    low_res_seg = join(work_space, names.seg)
    high_res_seg = join(work_space, names.hyper_primary_seg)
    identical_matrix = join(work_space, 'identical_matrix.mat')
    I = np.eye(4)
    np.savetxt(identical_matrix, I, fmt='%.0f')

    g = Greedy3D()
    g.execute(f'-d 3 -rf {reference_img} -ri LABEL {s_param}vox -rm {low_res_seg} {high_res_seg} -r {identical_matrix}')