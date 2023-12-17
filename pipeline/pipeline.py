import numpy as np
import pandas as pd
from math import floor

import os, shutil
from glob import glob
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage import io
from PIL import Image

import image_slicing
import torch_predict as pred
import postprocessing as pp
from process_curve import process_image
from approximation import fit_polyfit, format_equation

IMG_DIR = os.getcwd()+'/data'
IMG_SLICES_DIR = os.getcwd()+'/slices'
PATH_TO_MODEL = '/Users/danielageev/Work/AI BMSTU/ФОТОНИКА/pipeline/models/results_early_stop_15/i-0_art_depth-4_feats-32_BN-True_DO-True_LR-0.001_M-(0.9, 0.999)_Adam_UNET_4_32.pth'
PRED_MASK_DIR = os.getcwd()+'/predictions'
SMOOTHED_IMG_DIR = os.getcwd()+'/postprocessing'
COORDS_DIR = os.getcwd()+'/result_curve_coords'
COORDS_IMAGE_DIR = os.getcwd()+'/result_curve'
APPROXIMATION_RESULTS_DIR = os.getcwd()+'/approximate_result'
IMG_SLICE_SIZE = 512 # size of output snippet image

def step_1_segmentation():
    print(f"\n================================Prediction=================================")

    images_path = glob(os.path.join(IMG_DIR, "*"))
    
    for i in tqdm(range(len(images_path)), total=len(images_path)):
        im = io.imread(fname=images_path[i])
        original_height = im.shape[0]
        original_width = im.shape[1]
        sliced_imgs = image_slicing.image_slicing(im, IMG_SLICE_SIZE)

        img_name = '.'.join(os.path.basename(images_path[i]).split(".")[:-1])
        print(f"\n================================{img_name} done =================================")

        for i in range(len(sliced_imgs)):
            output_name = f"{img_name}_cut_{i + 1}.png"
            io.imsave(os.path.join(IMG_SLICES_DIR, output_name), sliced_imgs[i], check_contrast=False)
        pred_masks_slices = pred.predict(sliced_imgs, model_path=PATH_TO_MODEL)
        pp.assemble_image(pred_masks_slices, IMG_SLICE_SIZE, original_width, original_height,img_name)

def step_2_postprocessing(detect_mode='min', cs=50, prominence=8):
    print(f"\n================================Postprocessing=================================")

    images_path = glob(os.path.join(PRED_MASK_DIR, "*"))

    for i in tqdm(range(len(images_path)), total=len(images_path)):
        original = Image.open(images_path[i])
        smoothed, rectangled, result_curve = process_image(images_path[i],
                                                           detect_mode=detect_mode,
                                                           cs=cs,
                                                           prominence_=prominence)
        fig, axes = plt.subplots(3, 1, figsize=(30, 15))
        axes[0].imshow(np.asarray(original))
        axes[0].set_title('Step 1')
        axes[0].axis('off')

        axes[1].imshow(np.asarray(smoothed))
        axes[1].set_title('Step 2')
        axes[1].axis('off')

        axes[2].imshow(np.asarray(rectangled))
        axes[2].set_title('Step 3')
        axes[2].axis('off')
        image_filename = os.path.splitext(os.path.basename(images_path[i]))[0]
        print(f"\n================================{image_filename} done =================================")

        plt.savefig(os.path.join(SMOOTHED_IMG_DIR, f"{image_filename}_postprocessing.png"), bbox_inches='tight', dpi=100)
        plt.close(fig)

        max_y = max(result_curve)
        for i in range(len(result_curve)):
            result_curve[i] = max_y - result_curve[i]

        with open(os.path.join(COORDS_DIR, f"{image_filename}_curve_coords.txt"), "w") as file:
            for idx in range(len(result_curve)):
                file.write(f"{idx}\t{result_curve[idx]}\n")

        plt.figure()
        plt.plot(list(range(len(result_curve))), result_curve)
        plt.title("Result Curve")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")

        plt.savefig(os.path.join(COORDS_IMAGE_DIR, f"{image_filename}_curve.png"))
        plt.close(fig)

def step_3_approximation(degree=4, scale=370):
    print(f"\n================================Approximation=================================")

    coords_path = glob(os.path.join(COORDS_DIR, "*"))
    dataframe_rows = []

    for i in tqdm(range(len(coords_path)), total=len(coords_path)):
        image_filename = os.path.splitext(os.path.basename(coords_path[i]))[0]
        print(f"\n================================{image_filename} done =================================")

        coefficients, error, fig, depth, modification = fit_polyfit(coords_path[i], degree=degree)

        fig.savefig(os.path.join(APPROXIMATION_RESULTS_DIR, f"{image_filename}_approximation.png"))
        plt.close(fig)

        row = {
            "Curve Original": image_filename,
            "Function": format_equation(coefficients),
            "Error, %": f'{error:.2f} %',
            "Structure Depth, um": f'{(depth / scale):.2f}',
            "Modification diameter, um": f'{(modification / scale):.2f}'
        }
        for power, coeff in enumerate(coefficients[::-1]):
            row[f'x^{power}'] = f'{coeff:.6e}'
            #row[f'x^{power}'] = coeff
        dataframe_rows.append(row)

    df = pd.DataFrame(dataframe_rows)

    fixed_columns = ['Curve Original', 'Function', 'Error, %', "Structure Depth, um", "Modification diameter, um"]
    coeff_columns = [f'x^{i}' for i in range(degree + 1)]
    df = df[fixed_columns + coeff_columns[::-1]]
    df.to_csv('result_approximation.csv', index=False)



def clear_dirs():
    for folder in [PRED_MASK_DIR, SMOOTHED_IMG_DIR, COORDS_DIR, COORDS_IMAGE_DIR, IMG_DIR, IMG_SLICES_DIR]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == "__main__":
    #step_1_segmentation()
    #step_2_postprocessing()
    step_3_approximation()
    #clear_dirs()
