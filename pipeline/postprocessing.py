import os
from collections import defaultdict
from PIL import Image


def assemble_image(pred_image_slices,img_slice_size, original_width, original_height, filename):

    # Create a directory to save the combined images
    output_directory = "predictions"
    os.makedirs(output_directory, exist_ok=True)

    # Calculate the number of horizontal and vertical slices
    num_slices_horizontal = ceildiv(original_width, img_slice_size)
    num_slices_vertical = ceildiv(original_height, img_slice_size)

    # Iterate through the img_slices, combine them horizontally, and save as new images
    combined_image = Image.new("RGB", (original_width,original_height))
    row_ind = 0
    col_ind = 0
    step = 0
    for slice in pred_image_slices:

        if row_ind >= num_slices_horizontal:
            row_ind = 0
            col_ind += 1

        if col_ind >= num_slices_vertical - 1:
            step = original_height % num_slices_vertical

        x_offset = row_ind * slice.width
        y_offset = col_ind * slice.height - step
        combined_image.paste(slice, (x_offset, y_offset))

        row_ind += 1

        combined_image.save(os.path.join(output_directory, f"{filename}_assembled.png"))

    print(f"Image img_slices combined horizontally and saved to {output_directory}")


def assemble_image_from_dir(path,img_slice_size, original_width, original_height):
    # Create a dictionary to store img_slices by their filename beginning
    img_slices = defaultdict(list)

    # Iterate through files in the directory
    path_list = os.listdir(path)
    sorted_path_list = sorted(path_list, key=lambda item: (int(item.partition('cut_')[-1].partition('.png')[0])
                                   if item[0].isdigit() else float('inf'), item))
    for filename in sorted_path_list:
        if filename.endswith(".png"):  # Change the extension if needed
            # Extract the filename beginning (excluding the numerical part)
            filename_beginning = filename.split("_cut_")[0]  # Adjust the delimiter as needed

            # Open the image slice
            image = Image.open(os.path.join(path, filename))

            # Append the slice to the dictionary
            img_slices[filename_beginning].append(image)


    # Create a directory to save the combined images
    output_directory = "predicted_masks_assembled"
    os.makedirs(output_directory, exist_ok=True)

    # Calculate the number of horizontal and vertical slices
    num_slices_horizontal = ceildiv(original_width, img_slice_size)
    num_slices_vertical = ceildiv(original_height, img_slice_size)

    # Iterate through the img_slices, combine them horizontally, and save as new images
    for filename_beginning, slices_list in img_slices.items():
        combined_image = Image.new("RGB", (original_width,original_height))

        row_ind = 0
        col_ind = 0
        step = 0
        for slice in slices_list:

            if row_ind >= num_slices_horizontal:
                row_ind = 0
                col_ind += 1

            if col_ind >= num_slices_vertical - 1:
                step = original_height % num_slices_vertical

            x_offset = row_ind * slice.width
            y_offset = col_ind * slice.height - step
            combined_image.paste(slice, (x_offset, y_offset))

            row_ind += 1

        # Save the combined image with a new filename
        combined_image.save(os.path.join(output_directory, f"{filename_beginning}_combined.png"))

    print("Image img_slices combined horizontally and saved.")

def ceildiv(a,b):
    return -(a // -b)