"""
Purposes of this script:
 - To evaluate different downscaling algorithms and parameters in comparison with baseline reference images (Mac)
 - To find out which parameters/downscaling approaches result in images closest to the Mac downscaling
 - To introduce artificial defects in downscaled images in order to understand how this impacts the metrics
 - To scale the images down and up again in order to see how much information/quality was lost in the process
 - To visualize the results in plots/diagrams
 - To compare color/luminance histograms of original and down+up-scaled images to see the overall changes
"""

# TODO: 1. Function to read in all images and find image pairs
# TODO: 2. Compare Mac downscaled with cv2 downscaled images (metrics + pixel-by-pixel)
# TODO: 3. Scale down and up and evaluate the pixel-by-pixel difference, see histogram changes
# TODO: 4. Introduce random (with set seed) impurities in the images and see the effect on the metrics
# TODO: 5. Iteratively try out different parameters
# TODO: 6. Visualize all results as csv and plots
# TODO: 7. Evaluate many different constellations and apply regression to find optimal parameters?

import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import image_similarity_measures.evaluate as img_eval
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
original_dir = os.path.join(script_dir, '..', 'test_img', 'original')
output_dir = os.path.join(script_dir, 'output')

interpolation_methods = {
    'INTER_NEAREST': cv2.INTER_NEAREST,
    'INTER_LINEAR': cv2.INTER_LINEAR,
    'INTER_AREA': cv2.INTER_AREA,
    'INTER_CUBIC': cv2.INTER_CUBIC,
    'INTER_LANCZOS4': cv2.INTER_LANCZOS4
}


def get_image_names(folder):
    image_names = []
    for file in os.listdir(folder):
        if file.endswith('_4k.jpg'):
            name = file.replace('_4k.jpg', '')
            image_names.append(name)
    return image_names


def compare_pixel_wise(name, img1, img2):
    """
    Compare images pixel-by-pixel to identify differences.
    Returns pixel difference metrics.
    """
    # Ensure images are of the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions for pixel-wise comparison.")

    # Calculate absolute pixel differences and return relevant statistics
    diff = cv2.absdiff(img1, img2)
    diff_sum = np.sum(diff)
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)

    return {'pixel_diff_sum': diff_sum, 'pixel_diff_mean': diff_mean, 'pixel_diff_std': diff_std}


# TODO: Histogram difference plots (one plot per image containing different variations)


def compare_metrics(img_name, img1, img2, metrics=None):
    """
    Compare images based on a variety of image quality metrics.
    Returns a dictionary or DataFrame with metric results.
    """
    # If no specific metrics were chosen, apply all
    if metrics is None:
        metrics = ["rmse", "psnr", "ssim", "issm"]  # "fsim", ssim causes errors

    # Prepare dictionary to store metric values
    metric_results = {}

    # Convert images to grayscale for some metrics if needed
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate Root Mean Squared Error (RMSE)
    if "rmse" in metrics:
        rmse_value = np.sqrt(mean_squared_error(img1_gray.flatten(), img2_gray.flatten()))
        metric_results["RMSE"] = rmse_value

    # Calculate Peak Signal-to-Noise Ratio (PSNR)
    if "psnr" in metrics:
        psnr_value = psnr(img1_gray, img2_gray, data_range=img2_gray.max() - img2_gray.min())
        metric_results["PSNR"] = psnr_value

    # Calculate Structural Similarity Index (SSIM)
    if "ssim" in metrics:
        ssim_value, _ = ssim(img1_gray, img2_gray, full=True)
        metric_results["SSIM"] = ssim_value

    for metric in metrics:
        if metric in img_eval.metric_functions:
            metric_func = img_eval.metric_functions[metric]
            try:
                # Calculate metric and add to results
                out_value = float(metric_func(img1, img2))
                metric_results["IMG_EVAL_"+metric.upper()] = round(out_value, 4)
            except Exception as e:
                print(f"Error calculating {metric}: {e}")
                metric_results[metric.upper()] = np.nan

    return  pd.DataFrame([metric_results])


def compare_downscaled_to_baseline(img_names):
    """
    Compare each downscaled 4K image to its baseline Mac-downscaled image.
    Loops through image_names, downscaling algorithms, and impurity levels.
    Returns a DataFrame with all comparison results.
    """
    results = []
    original_settings = np.seterr()
    np.seterr(divide='ignore', invalid='ignore')
    for img_nr, name in enumerate(img_names):
        print(f"Evaluating image {img_nr+1}/{len(img_names)}")
        # Load original 4K and baseline Mac-downscaled image
        original_img_path = os.path.join(original_dir, f'{name}_4k.jpg')
        baseline_img_path = os.path.join(original_dir, f'{name}_1920x1080.png')

        original_img = cv2.imread(original_img_path)
        baseline_img = cv2.imread(baseline_img_path)

        baseline_hist = cv2.calcHist([cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])

        # Loop over each interpolation method
        for method_name, method in tqdm(interpolation_methods.items(), desc="Downscaling methods"):
            # Downscale original image to match baseline resolution
            baseline_height, baseline_width = baseline_img.shape[:2]
            downscaled_img = cv2.resize(original_img, (baseline_width, baseline_height), interpolation=method)

            downscaled_hist = cv2.calcHist([cv2.cvtColor(downscaled_img, cv2.COLOR_BGR2GRAY)], [0], None, [256],
                                           [0, 256])

            # Loop over different levels of artificial impurities
            for impurity_level in [0, 0.01, 0.05, 0.1]:
                if impurity_level > 0:
                    noisy_img = (downscaled_img +
                                 gaussian_filter(np.random.normal(0, 255 * impurity_level, downscaled_img.shape),
                                                 sigma=1).astype(np.uint8))
                else:
                    noisy_img = downscaled_img

                # Compare noisy or clean downscaled image to baseline
                pixel_comparison = compare_pixel_wise(name, noisy_img, baseline_img)
                metric_comparison = compare_metrics(name, noisy_img, baseline_img)

                # Store results for this combination
                result_entry = {
                    'image_name': name,
                    'method': method_name,
                    'impurity_level': impurity_level,
                    'histogram_baseline': baseline_hist.flatten(),  # Store baseline histogram
                    'histogram_downscaled': downscaled_hist.flatten(),  # Store downscaled histogram
                    **pixel_comparison,
                    **metric_comparison.to_dict(orient='records')[0]  # Convert DataFrame row to dictionary
                }
                results.append(result_entry)

    np.seterr(**original_settings)

    # Convert results to DataFrame and save as tab-separated CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'metrics_comparison_results.tsv'), sep='\t', index=False)

    return results_df


def compare_rescaled_to_original(img_names):
    pass # TODO: Implement


def create_summary_plots(df, img_names):
    """
    Create a summary plot for each image with overlapping histograms and metric-impurity correlation.
    """
    plot_paths = []
    for image_name in tqdm(img_names, desc="Plotting data"):
        # Filter dataframe for the specific image and exclude impurity effects for the histogram plot
        df_image = df[df['image_name'] == image_name]
        df_no_impurity = df_image[df_image['impurity_level'] == 0]

        # Set up a figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1]})
        fig.suptitle(f"Summary for {image_name}")

        # Plot 1: Overlapping Histogram of Mac vs. Downscaled with different methods (ignoring impurities)
        mac_baseline_hist = df_no_impurity['histogram_baseline'].values[0]
        sns.lineplot(data=mac_baseline_hist, label="Mac Baseline", color="black", ax=axes[0])

        # Plot downscaled histograms for each method, without impurities
        for method in df_no_impurity['method'].unique():
            method_data = df_no_impurity[df_no_impurity['method'] == method]
            downscaled_hist = method_data['histogram_downscaled'].values[0]
            sns.lineplot(data=downscaled_hist, label=method, ax=axes[0])

        axes[0].set_title("Overlapping Histograms")
        axes[0].set_xlabel("Pixel Intensity")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        # Plot 2: Correlation between Impurity and Metrics
        for method in df_image['method'].unique():
            df_method = df_image[df_image['method'] == method]
            for metric in ["SSIM", "PSNR", "RMSE"]:
                sns.lineplot(
                    x="impurity_level", y=metric, data=df_method, ax=axes[1], marker="o", label=f"{method} - {metric}"
                )

        axes[1].set_title("Metric vs. Impurity Level")
        axes[1].set_xlabel("Impurity Level")
        axes[1].set_ylabel("Metric Value")
        axes[1].legend(title="Method - Metric")

        # Plot 3: Metric Comparison at Zero Impurity (Bars slightly transparent)
        sns.barplot(
            x="method", y="SSIM", data=df_no_impurity, ax=axes[2], color="skyblue", alpha=0.6, label="SSIM"
        )
        sns.barplot(
            x="method", y="PSNR", data=df_no_impurity, ax=axes[2], color="salmon", alpha=0.6, label="PSNR"
        )
        sns.barplot(
            x="method", y="RMSE", data=df_no_impurity, ax=axes[2], color="lightgreen", alpha=0.6, label="RMSE"
        )
        axes[2].set_title("Metric Comparison (No Impurity)")
        axes[2].set_xlabel("Method")
        axes[2].set_ylabel("Metric Value")
        axes[2].legend(title="Metrics")

        # Adjust layout and save the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
        output_path = os.path.join(output_dir, f"{image_name}_summary.png")
        plt.savefig(output_path)
        plt.close(fig)

        plot_paths.append(output_path)

    for plot_path in plot_paths:
        print(f"Saved summary plot at {plot_path}")


def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of image names without extensions and resolutions
    image_names = get_image_names(original_dir)

    baseline_results_df = compare_downscaled_to_baseline(image_names)

    create_summary_plots(baseline_results_df, image_names)

    # rescaled_results_df = compare_rescaled_to_original(image_names)


if __name__ == '__main__':
    print("\n\n")
    main()
