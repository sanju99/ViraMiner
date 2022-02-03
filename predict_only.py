import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import binarize

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, concatenate, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, Nadam, SGD

from sklearn.metrics import confusion_matrix, roc_auc_score

from helper_with_N import *

import argparse
import subprocess


################################
##Read in the parameter values## 
################################
parser = argparse.ArgumentParser()
parser.add_argument("--input_file") # data file name
parser.add_argument("--model_path") # pretrained model
parser.add_argument("--output_path") # output path

args = parser.parse_args()

#################################################
# Automatically checking train and test set sizes!
#################################################

def wc(filename):
    return int(subprocess.run(["wc", "-l", filename], capture_output=True).stdout.split()[0])

test_set_size = wc(args.input_file)
print(f"test_set_size: {test_set_size}")
te_steps_per_ep = int(test_set_size/128)
print(f"##\n input data: \n {args.input_file} \n##")


#######################
# read in the test data, only true labels are needed for ROC
test_labels = []
counter = 0

f = open(args.input_file)
for line in f:
    line = line[:-1] #remove \n
    seq, lab = process_line(line)
    test_labels.append(lab)
    counter += 1
f.close()

test_labels = np.array(test_labels) # put to numpy format

###############################
####Defining the model#########
###############################

# load pretrained models
model = load_model(args.model_path)
model.summary()

model_name = (args.model_path).split(".hdf")[0]  # for saving predictions and labels
print(model_name)
###############################
####Testing the model##########
###############################

print("##########################")
pred_probas = model.predict(generate_batches_from_file(args.input_file, 128), steps=te_steps_per_ep+1,workers=1, use_multiprocessing=False)
print(f"original pred_probas size (divisible with batch size) {np.shape(pred_probas)}")
pred_probas = pred_probas[:test_set_size,:]
print(f"cropped the repetitions away, leaving {np.shape(pred_probas)}")

# convert the predicted probabilities to class labels
preds = binarize(pred_probas.reshape((-1, 1)), threshold=0.5)

print(f"TEST ROC area under the curve \n {roc_auc_score(test_labels, preds)}")
pd.DataFrame({"pred": preds.flatten(), "test": test_labels}).to_csv(args.output_path + "_test.txt", sep="\t")
                                                                 
# np.savetxt(args.output_path+"_TEST_predictions.txt", pred_probas, fmt="%.5f")
# np.savetxt(args.output_path+"_TEST_labels.txt", test_labels, fmt="%d")
