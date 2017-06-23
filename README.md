# Segmentation

This project is for the Liver Tumor Segmentation Challenge (LiTS). This challenge contains abdomen CT  of patients with liver tumors. In the training set both the liver and the liver tumors are labeled. The goal is to label the liver tumors only on the test set. This project ignores the liver labels and tries to segment the liver tumors directly. The project is written in PyTorch and contains a 2 dimensional adaption of VNet, using adjacent slices for more context, making it 2.5 dimensional. 

# Training

Make sure to run the data_create.py script once before training to convert the nii.gz files into npy files for every slice. Then run the train.py file to train the network.

# Inference

After training is done you can run the inference.py file to segment the test set.