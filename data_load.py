import os
import torch
import numpy as np
import torch.utils.data as data_utils

# Liver Dataset - segmentation task
# when false selects both the liver and the tumor as positive labels
class LiverDataSet(torch.utils.data.Dataset):

    def __init__(self, directory, augment=False, context=0):

        self.augment = augment
        self.context = context
        self.directory = directory
        self.data_files = os.listdir(directory)

        def get_type(s): return s[:1]
        def get_item(s): return int(s.split("_")[1].split(".")[0])
        def get_patient(s): return int(s.split("-")[1].split("_")[0])

        self.data_files.sort(key = lambda x: (get_type(x), get_patient(x), get_item(x)))
        self.data_files = zip(self.data_files[len(self.data_files)/2:], self.data_files[:len(self.data_files)/2])
    
    def __getitem__(self, idx):

        if self.context > 0:
            return load_file_context(self.data_files, idx, self.context, self.directory, self.augment)
        else:
            return load_file(self.data_files[idx], self.directory, self.augment)

    def __len__(self):

        return len(self.data_files)

    def getWeights(self):

        weights = []
        pos = 0.0
        neg = 0.0

        for data_file in self.data_files:

            _, labels = data_file
            labels = np.load(os.path.join(self.directory, labels))

            if labels.sum() > 0:
                weights.append(-1)
                pos += 1
            else:
                weights.append(0)
                neg += 1

        weights = np.array(weights).astype(float)
        weights[weights==0] = 1.0 / neg * 0.1
        weights[weights==-1] = 1.0 / pos * 0.9

        print('%d samples with positive labels, %d samples with negative labels.' % (pos, neg))

        return weights

    def getPatients(self):

        patient_dictionary = {}

        for i, data_file in enumerate(self.data_files):

            _, labels = data_file
            patient = labels.split("_")[0].split("-")[1]

            if patient in patient_dictionary:
                patient_dictionary[patient].append(i)
            else:
                patient_dictionary[patient] = [i]

        return patient_dictionary


# load data_file in directory and possibly augment
def load_file(data_file, directory, augment):

    inputs, labels = data_file
    inputs, labels = np.load(os.path.join(directory, inputs)), np.load(os.path.join(directory, labels))
    inputs, labels = np.expand_dims(inputs, 0), np.expand_dims(labels, 0)

    # augment
    if augment and np.random.rand() > 0.5:
        inputs = np.fliplr(inputs).copy()
        labels = np.fliplr(labels).copy()

    features, targets = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
    return (features, targets)

# load data_file in directory and possibly augment including the slides above and below it
def load_file_context(data_files, idx, context, directory, augment):

    # check whether all inputs need to be augmented
    if augment and np.random.rand() > 0.5: augment = False

    # load middle slice
    inputs_b, labels_b = data_files[idx]
    inputs_b, labels_b = np.load(os.path.join(directory, inputs_b)), np.load(os.path.join(directory, labels_b))
    inputs_b, labels_b = np.expand_dims(inputs_b, 0), np.expand_dims(labels_b, 0)

    # augment
    if augment:
        inputs_b = np.fliplr(inputs_b).copy()
        labels_b = np.fliplr(labels_b).copy()

    # load slices before middle slice
    inputs_a = []
    for i in range(idx-context, idx):

        # if different patient or out of bounds, take middle slice, else load slide
        if i < 0 or data_files[idx][0][:-6] != data_files[i][0][:-6]:
            inputs = inputs_b
        else:
            inputs, _ = data_files[i]
            inputs = np.load(os.path.join(directory, inputs))
            inputs = np.expand_dims(inputs, 0)
            if augment: inputs = np.fliplr(inputs).copy()

        inputs_a.append(inputs)

    # load slices after middle slice
    inputs_c = []
    for i in range(idx+1, idx+context+1):

        # if different patient or out of bounds, take middle slice, else load slide
        if i >= len(data_files) or data_files[idx][0][:-6] != data_files[i][0][:-6]:
            inputs = inputs_b
        else:
            inputs, _ = data_files[i]
            inputs = np.load(os.path.join(directory, inputs))
            inputs = np.expand_dims(inputs, 0)
            if augment: inputs = np.fliplr(inputs).copy()

        inputs_c.append(inputs)

    # concatenate all slices for context
    # middle sice first, because the network that one for the residual connection
    inputs = [inputs_b] + inputs_a + inputs_c
    labels = labels_b

    inputs = np.concatenate(inputs, 0)

    features, targets = torch.from_numpy(inputs).float(), torch.from_numpy(labels).long()
    return (features, targets)
