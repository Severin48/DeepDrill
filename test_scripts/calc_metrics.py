# Calculates some image quality metrics based of 
# https://pypi.org/project/image-similarity-measures/
# 
# Usage: 
# 1. call this script with the "init" argument (e.g. python calc_metrics.py init). Downscaled versions of the original images will be created.
# 4. call this script with metric names as arguments (e.g. python calc_metrics.py psnr rmse) 

import cv2 
import os
import sys
import image_similarity_measures.evaluate as img_eval
from prettytable import PrettyTable
from tqdm import tqdm

base_path = os.path.join(os.getcwd(),'test_img')

example_files = {
    # name : [original, original_downscaled]
   'seahorse1':['original/seahorse1_4k.jpg', 'original/seahorse1_1920x1080.png'],
   'seahorse2':['original/seahorse2_4k.jpg', 'original/seahorse2_1920x1080.png'],
   'seahorse3':['original/seahorse3_4k.jpg', 'original/seahorse3_1920x1080.png']
}

test_img_dir = "test_img"
scale_factor = 0.5
name_to_cv2_scaling = {"nn":cv2.INTER_NEAREST,
                       "linear":cv2.INTER_LINEAR,
                       "lanczos4":cv2.INTER_LANCZOS4,
                       "cubic":cv2.INTER_CUBIC}

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
    for name, paths in example_files.items():
        original = cv2.imread(os.path.join(base_path, paths[0])) 
        scaled_imgs = get_img_variants(original,scale_factor=scale_factor)
       
        for k,v in scaled_imgs.items():
            file_name = f'{name}_{k}.jpg'
            cv2.imwrite(os.path.join(test_img_dir, file_name), v)

def calc_metrics(metrics):
    """
    calculates the metrics for each image from 'paths' with the image from 'original_path'
    the images must have all the same size!
    """
    results = PrettyTable(["Image", "Metric", "Scaling", "Value"])

    with tqdm(total=len(example_files.items())*len(name_to_cv2_scaling.keys())*len(metrics)) as pbar:
        for name, paths in example_files.items():
            gt = cv2.imread(os.path.join(base_path, paths[1]))

            for metric in metrics:
                for method in name_to_cv2_scaling.keys():
                        pred = cv2.imread(os.path.join(base_path, f'{name}_{method}.jpg'))
                        metric_func = img_eval.metric_functions[metric]

                        metric_value = float(metric_func(gt, pred))
                        results.add_row([name, metric,method, round(metric_value,4)])

                        pbar.update(1)
                
            results.add_row(['','','',''])

    return results
    
def load_image_with_info(path):
    base_name = os.path.basename(path)
    parts = base_name.split('_')

    img_name = parts[0]
    scaling_method = parts[1].split(".")[0]
    img = cv2.imread(path)

    return (img_name,scaling_method,img)
  
def main(args):     
# IF DEBUG
    if __debug__:
        # args = ["init"]
        # rmse, psnr, ssim, fsim, issm
        args = ["psnr",'ssim']
        # args = ["hist"]
# ENDIF DEBUG

    if len(args) == 1 and args[0] == "init":
        create_files()
    elif len(args) == 1 and args[0] == "hist":
        pass
    else:
        metrics = calc_metrics(metrics=args)
        print(metrics)
        
if __name__ == "__main__": 
    main(sys.argv) 