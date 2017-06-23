import os
import networks
import numpy as np
import torch
import nibabel as nib
from torch.autograd import Variable

### variables ###

# name of the model saved
model_name = '25D'

# the number of context slices before and after as defined as in train.py before training
context = 2

# directory where to store nii.gz or numpy files
result_folder = 'results'
test_folder = '../../data/Test Batch 1'

#################

# create result folder if neccessary
if not os.path.isdir(result_folder):
	os.makedirs(result_folder)

# filter files that don't start with test
files = [file for file in os.listdir(test_folder) if file[:4]=="test"]

# load network
cuda = torch.cuda.is_available()
net = torch.load("model_"+model_name+".pht")
if cuda: net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).cuda()
net.eval() # inference mode

for file_name in files:

	# load file
	data = nib.load(os.path.join(test_folder, file_name))

	# save affine
	input_aff = data.affine

	# convert to numpy
	data = data.get_data()

	# normalize data
	data = np.clip(data, -200, 200) / 400.0 + 0.5
	
	# transpose so the z-axis (slices) are the first dimension
	data = np.transpose(data, (2, 0, 1))

	# save output here
	output = np.zeros((len(data), 512, 512))

	# loop through z-axis
	for i in range(len(data)):

		# append multiple slices in a row
		slices_input = []
		z = i - context

		# middle slice first, same as during training
		slices_input.append(np.expand_dims(data[i], 0))

		while z <= i + context:

			if z == i:
				# middle slice is already appended
				pass
			elif z < 0:
				# append first slice if z falls outside of data bounds
				slices_input.append(np.expand_dims(data[0], 0))
			elif z >= len(data):
				# append last slice if z falls outside of data bounds
				slices_input.append(np.expand_dims(data[len(data)-1], 0))
			else:
				# append slice z
				slices_input.append(np.expand_dims(data[z], 0))
			z += 1

		inputs = np.expand_dims(np.concatenate(slices_input, 0), 0)
		
		# run slices through the network and save the predictions
		inputs = Variable(torch.from_numpy(inputs).float(), volatile=True)
		if cuda: inputs = inputs.cuda()

		# inference
		outputs = net(inputs)
		outputs = outputs[0, 1, :, :].round()
		outputs = outputs.data.cpu().numpy()

		# save slices (* 2 because of liver tumor predictions, not liver predictions)
		output[i, :, :] = outputs * 2

	# transpose so z-axis is last axis again and transform into nifti file
	output = np.transpose(output, (1, 2, 0)).astype(np.uint8)
	output = nib.Nifti1Image(output, affine=input_aff)

	new_file_name = "test-segmentation-" + file_name.split("-")[-1]
	print new_file_name
	
	nib.save(output, os.path.join(result_folder, new_file_name))
