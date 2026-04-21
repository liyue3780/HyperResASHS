import SimpleITK as sitk
import pandas as pd
import numpy as np
from picsl_c3d import Convert3D
from typing import List, Dict, Any

def ashs_cleanup(config: Dict[str, Any], seg: sitk.Image, stats_csv=None) -> sitk.Image:
    """
    Clean up nnUNet segmentation by taking connected components of specific groups of labels, and optionally dilating them. 
    The config should specify the label groups, the number of components to keep, and the dilation radius. For example:
        
    cleanup:
        label_groups:
            - name: Hippo_MTL_GM
            labels: [10, 11, 12, 13, 21, 22, 23, 24, 28, 31, 32, 33, 34, 38]
            components: 1
            dilation: 0
            - name: Amygdala
            labels: [18]
            components: 1
            dilation: 0
            - name: WM
            labels: [20]
            components: 1
            dilation: 0
            - name: All
            labels: []
            dilation: 5
            components: 1
    """
    c3d = Convert3D()
    c3d.push(seg)
    
    # Apply each cleanup rule
    for group in config['label_groups']:
        name = group['name'] 
        labels = group['labels'] # type: List[int]
        components = group.get('components', 1)
        dilation = group.get('dilation', 0)
        
        
        # Command to retain the labels
        labcmd = f'-retain-labels {" ".join(map(str, labels))}' if len(labels) > 0 else ''
        
        # Command to dilate the mask
        dilcmd = f'-dilate 1 {dilation}x{dilation}x{dilation}' if dilation > 0 else ''
            
        # Retain the labels, compute connected components, and keep the specified number; finally multiply the component mask with the labels
        c3d.execute(f'-as S {labcmd} -thresh 1 inf 1 0 -as B {dilcmd} -comp -thresh {components+1} inf 0 1 -as M -push S -times')
        
    # Compute the volume of each label that has been removed
    cln = c3d.peek(-1)
    
    # Optionally save a CSV with the volume of each label before and after cleanup
    if stats_csv is not None:
        arr_seg, arr_cln = sitk.GetArrayFromImage(seg), sitk.GetArrayFromImage(cln)
        stats = {'label': [], 'n_seg': [], 'n_cln': [], 'removed': [], 'percent_removed': []}
        for label in np.unique(sitk.GetArrayFromImage(seg)):
            if label > 0:
                n_seg = (arr_seg == label).sum()
                n_cln = (arr_cln == label).sum()
                if n_seg - n_cln > 0:
                    stats['label'].append(label)
                    stats['n_seg'].append(n_seg)
                    stats['n_cln'].append(n_cln)
                    stats['removed'].append(n_seg - n_cln)
                    stats['percent_removed'].append((n_seg - n_cln) / n_seg)
        df = pd.DataFrame(stats)
        df.to_csv(stats_csv, index=False)

    return cln
        
        
if __name__ == "__main__":
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up nnUNet segmentation for ASHS')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input segmentation image')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to output cleaned segmentation image')
    parser.add_argument('-s', '--stats', type=str, help='Optional path to CSV file to save label volume statistics before and after cleanup')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    seg = sitk.ReadImage(args.input)
    cleaned_seg = ashs_cleanup(config, seg, stats_csv=args.stats)
    sitk.WriteImage(cleaned_seg, args.output)        
        
        
        
        