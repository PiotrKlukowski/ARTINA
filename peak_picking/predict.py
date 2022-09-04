"""

    The script:
    a) Downloads model data from nmrtist.org
    b) Loads the peak picking model and fragments cropped from NMR spectrum
    c) Makes prediction
    d) Visualizes prediction
       Central point of each panel (point evaluated by ResNet) is marked in red. The classifier output is presented in the upper part of each image.

    The use of model with external data:
    a) Normalize NMR spectrum, such that median signal amplitude is 100
    b) Crop spectrum fragment, such that the signal of interest is located in the central part of the image
    c) Normalize spectrum fragment amplitude: fragment = np.maximum(np.log(np.abs(fragment) + 1e-5) / np.log(1.4), 0)

    5 fold split:
    model_fold1 test proteins: 1SE9, 2JVD, 2K1G, 2K50, 2KFP, 2KJR, 2KKL, 2KL5, 2KOB, 2L05, 2L82, 2LA6, 2LEA, 2LFI, 2LML, 2LND, 2LRH, 2LTL, 2LX7, 2MDR
    model_fold2 test proteins: 2HEQ, 2JN8, 2JT1, 2K3A, 2K52, 2K5V, 2KD0, 2KHD, 2KIF, 2KKZ, 2KRS, 2L06, 2L3B, 2L8V, 2LAH, 2LK2, 2LN3, 2LTM, 2LXU, 2M7U, 2MB0, 1VDY, 6SVC
    model_fold3 test proteins: 2G0Q, 2JQN, 2K3D, 2K5D, 2KBN, 2KD1, 2KK8, 2KPN, 2KRT, 2KVO, 2L1P, 2L3G, 2L9R, 2LAK, 2LF2, 2LL8, 2LNA, 2LVB, 2MA6, 1VEE
    model_fold4 test proteins: 1PQX, 1T0Y, 1YEZ, 2B3W, 2JXP, 2K0M, 2K57, 2K75, 2KCD, 2KCT, 2KL6, 2KZV, 2L33, 2L7Q, 2LGH, 2M47, 2MK2, 1WQU
    model_fold5 test proteins: 2ERR, 2FB7, 2JRM, 2JVO, 2K1S, 2K53, 2KIW, 2LTA, 2M4F, 2M5O, 2MQL, 2N4B, 2RN7, 6FIP, 6GT7, 6SOW, 3JZK, KRAS4B, MH04

    Technical remarks:
    a) Model has been trained with Keras 2.3.1 and Tensorflow 2.0.0
    b) Zip archive with model data should be extracted in predict.py directory (nmrtist.org/static/public/publications/artina/models/ARTINA_peak_picking.zip)


"""

import keras
import pickle
import matplotlib.pyplot as plt
import shutil
import os

# Download data
if not os.path.exists("example_input") or not os.path.exists("model") or not os.path.exists("ARTINA_peak_picking.zip"):
    os.system("wget nmrtist.org/static/public/publications/artina/models/ARTINA_peak_picking.zip")
    shutil.unpack_archive("ARTINA_peak_picking.zip")

# Load the model
model = keras.models.load_model('model/fold_1.h5')

for batch_id in range(1000):

    # Load example input
    with open(f"example_input/{batch_id}.pickle", "rb") as f:
        batch = pickle.load(f)

    # Predict on batch
    output = model.predict_on_batch(batch)

    # Visualize results
    for i in range(0, 64, 8):
        plt.figure(figsize=(16, 16))
        for j in range(0, 8):
            plt.subplot(1, 8, j + 1)
            plt.contour(batch[i + j, :, :, 0], levels=30)
            plt.title(f"Prediction: {output[i + j][0]:.3f}")
            plt.plot(15.5, 127.5, 'rx', linewidth=4)
        plt.tight_layout()
        plt.show()
