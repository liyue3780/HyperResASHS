#!/usr/bin/env python3
"""
trim_neck: Brain MRI neck removal tool

Python implementation of the trim_neck.sh script.
Uses SimpleITK and Convert3D (picsl_c3d) for image processing.

Script by Paul Yushkevich and Sandhitsu Das. Modified by Philip Cook.
Python port for HyperResASHS.
"""

import argparse
import os
import sys
import tempfile
import numpy as np
import SimpleITK as sitk
from picsl_c3d import Convert3D


def get_image_info(image_path):
    """Get image dimensions using SimpleITK."""
    img = sitk.ReadImage(image_path)
    size = img.GetSize()
    return [size[0], size[1], size[2]]


def trim_neck_in_memory(input_image:sitk.Image, headlen=150, clearance=10, verbose=False) -> sitk.Image:
    """
    Trim neck from brain MRI image. This version operates on SimpleITK Image objects in memory 
    rather than file paths and does not generate intermediate outputs. 
    
    Parameters:
    -----------
    input_image : sitk.Image
        Path to input T1 image
    headlen : float
        Length (sup-inf) of the head that should be captured in mm [default: 150]
    clearance : float
        Clearance above head that should be captured in mm [default: 10]
    verbose : bool
        Enable verbose/debug output [default: False]
    Returns:
    --------
    sitk.Image
        Trimmed image
    """
    verbose_prefix = '-verbose ' if verbose else ''

    c3d = Convert3D()
    c3d.push(input_image)
    c3d.execute(f'{verbose_prefix} -neck-trim-head-height {headlen} -neck-trim-top-clearance {clearance} '
                f'-as T1 -neck-trim -trim 0vox -push T1 -reslice-identity')
    return c3d.peek(-1)
    

def trim_neck(input_image:str, output_image:str, 
              headlen=150, clearance=10, 
              mask_out:str|None=None, mask_trim=False, 
              working_dir:str|None=None, verbose=False):
    """
    Trim neck from brain MRI image.
    
    Parameters:
    -----------
    input_image : str
        Path to input T1 image
    output_image : str
        Path to output trimmed image
    headlen : float
        Length (sup-inf) of the head that should be captured in mm [default: 150]
    clearance : float
        Clearance above head that should be captured in mm [default: 10]
    mask_out : str, optional
        Path to save mask indicating trimmed region
    mask_trim : bool
        If True, replace trimmed region with zeros rather than cropping [default: False]
    working_dir : str, optional
        Directory for intermediate files. If None, uses temp directory.
    verbose : bool
        Enable verbose/debug output [default: False]
    """
    # Simplified version with new c3d -neck-trim command
    verbose_prefix = '-verbose ' if verbose else ''

    c3d = Convert3D()
    
    # Execute algorithm, which outputs the mask
    c3d.execute(f'{verbose_prefix} -neck-trim-head-height {headlen} -neck-trim-top-clearance {clearance} '
                f'{input_image} -as T1 -neck-trim')
    
    # Write the mask if asked
    if mask_out is not None:
        sitk.WriteImage(c3d.peek(-1), mask_out)
    
    # Write the trimmed image or the masked image depending on mask_trim
    if mask_trim:
        c3d.execute(f'-push T1 -multiply -o {output_image}')
    else:
        c3d.execute(f'-trim 0vox -push T1 -reslice-identity -o {output_image}')

def main():
    parser = argparse.ArgumentParser(
        description='trim_neck: Brain MRI neck removal tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Script by Paul Yushkevich and Sandhitsu Das. Modified by Philip Cook.
Python port for HyperResASHS.
        """
    )
    
    parser.add_argument('input_image', help='Input T1 image path')
    parser.add_argument('output_image', help='Output trimmed image path')
    parser.add_argument('-l', '--headlen', type=float, default=150,
                        help='Length (sup-inf) of the head that should be captured in mm [default: 150]')
    parser.add_argument('-c', '--clearance', type=float, default=10,
                        help='Clearance above head that should be captured in mm [default: 10]')
    parser.add_argument('-m', '--mask-out', dest='mask_out', type=str, default=None,
                        help='Location to save mask indicating trimmed region')
    parser.add_argument('-r', '--mask-trim', dest='mask_trim', action='store_true',
                        help='Output replaces trimmed region with zeros, rather than cropping input')
    parser.add_argument('-w', '--working-dir', dest='working_dir', type=str, default=None,
                        help='Location to store intermediate files. Default: [TMPDIR]')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Verbose/debug mode')
    
    args = parser.parse_args()
    
    trim_neck(
        args.input_image,
        args.output_image,
        headlen=args.headlen,
        clearance=args.clearance,
        mask_out=args.mask_out,
        mask_trim=args.mask_trim,
        working_dir=args.working_dir,
        verbose=args.debug
    )


if __name__ == '__main__':
    main()

