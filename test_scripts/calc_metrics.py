# Calculates some image quality metrics based of 
# https://pypi.org/project/image-similarity-measures/
# 
# Usage: 
# 1. create directory DeepDrill/test_img/
# 2. load images into the directory with the format img_IMAGENAME_original.jpg where IMAGENAME is the name of the image
# 3. call this script with the "init" argument (e.g. python test_script/calc_metrics.py init). Downscaled versions of the original images will be created.
# 4. call this script with metric names as arguments (e.g. python test_script/calc_metrics.py psnr rmse) 
#
# You can change the scale_factor in this file to change the overall scaling 

import cv2 
import os
import glob
import sys
from collections import defaultdict
import image_similarity_measures.evaluate as img_eval
from prettytable import PrettyTable
import progressbar

test_img_dir = "test_img"
scale_factor = 0.125
name_to_cv2_scaling = {"nn":cv2.INTER_NEAREST,
                       "linear":cv2.INTER_LINEAR,
                       "lanczos4":cv2.INTER_LANCZOS4,
                       "cubic":cv2.INTER_CUBIC}
                       #"original":cv2.INTER_NEAREST}

def get_files():
    return glob.glob(os.path.join(test_img_dir, '*_original.jpg'))

def get_img_variants(img, scale_factor=0.5):
    scaled_imgs = {}
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    new_dimensions = (new_width, new_height)

    for k,v in name_to_cv2_scaling.items():
        img_variant = cv2.resize(img, new_dimensions, interpolation=v)
        scaled_imgs[k] = img_variant
    
    return scaled_imgs

def create_files():
    imgs = glob.glob(os.path.join(test_img_dir, '*_original.jpg'))

    for img in imgs:
        base_name = os.path.basename(img)
        img_name = base_name.split('_')[1]

        original = cv2.imread(img) 
        scaled_imgs = get_img_variants(original,scale_factor=scale_factor)
       
        for k,v in scaled_imgs.items():
            file_name = f'img_{img_name}_{k}.jpg'
            cv2.imwrite(os.path.join(test_img_dir, file_name), v)

def compare_files(metrics):
    imgs = glob.glob(os.path.join(test_img_dir, '*.jpg'))
    results = PrettyTable(["Image", "Metric", "Scaling", "Value"])


    # group files
    groups = defaultdict(list)
    for img in imgs:
        base_name = os.path.basename(img)
        parts = base_name.split('_')
        image_name = parts[1].split('.')[0]
        groups[image_name].append(base_name)

    # compare original with scaled 
    with progressbar.ProgressBar(max_value=len(imgs) * len(metrics)) as bar:
        for _,v in groups.items():
            original_path = os.path.join(test_img_dir, [file for file in v if "original" in file][0])
            other_paths = [os.path.join(test_img_dir, file) for file in v if "original" not in file]
            other_paths.sort()

            original = cv2.imread(original_path) 

            for path in other_paths:
                base_name = os.path.basename(path)
                parts = base_name.split('_')
                interpolation_method = parts[2].split(".")[0]

                scaled = cv2.imread(path) 
                
                # ensure same size.... problem because we measure the quality of two scalings...
                original_height, original_width = original.shape[:2]
                resized_scaled_image = cv2.resize(scaled, (original_width, original_height), interpolation=name_to_cv2_scaling[interpolation_method])

                for metric in metrics:
                    metric_func = img_eval.metric_functions[metric]
                    out_value = float(metric_func(original, resized_scaled_image))
                    img_eval.logger.info(f"{metric.upper()} value is: {out_value}")
                    results.add_row([parts[1], metric, parts[2].split(".")[0], round(out_value,4)])
                    bar.next()

            results.add_row(["-","-","-","-"])
        
        print(results)

def main(args):     
# IF DEBUG
    if __debug__:
        #args = ["init"]
        args = ["ssim","psnr"]
# ENDIF DEBUG

    if len(args) > 0 and args[0] == "init":
        create_files()
    else:
        compare_files(args)
        
if __name__ == "__main__": 
    main(sys.argv) 