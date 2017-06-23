import numpy as np
import nibabel as nib
import os.path


### variables ###

# validation list
val_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# source folder where the .nii.gz files are located 
source_folder = '../../data/Train Batch 1'

#################


# destination folder where the subfolders with npy files will go
destination_folder = 'data'

# returns the patient number from the filename
def get_patient(s): return int(s.split("-")[-1].split(".")[0])

# create destination folder and possible subfolders
subfolders = ["train", "val"]
if not os.path.isdir(destination_folder):
	os.makedirs(destination_folder)
for name in subfolders:
	if not os.path.isdir(os.path.join(destination_folder, name)):
        	os.makedirs(os.path.join(destination_folder, name))

for file_name in os.listdir(source_folder):

    print file_name

    # create new file name by stripping .nii.gz and adding .npy
    new_file_name = file_name[:-7]

    # decide whether it will go to the train or val folder
    sub = subfolders[1] if get_patient(file_name) in val_list else subfolders[0]

    # load file
    data = nib.load(os.path.join(source_folder, file_name))

    # convert to numpy
    data = data.get_data()

    # check if it is a volume file and clip and standardize if so
    if file_name[:3] == 'vol': 
        data = np.clip(data, -200, 200) / 400.0 + 0.5

    # check if it is a segmentation file and select only the tumor (2) as positive label
    if file_name[:3] == 'seg': data = (data==2).astype(np.uint8)

    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    # loop through the slices
    for i, z_slice in enumerate(data):

        # save at new location (train or val)
        np.save(os.path.join(destination_folder, sub, new_file_name + '_' + str(i)), z_slice)
