"""

    The script:
    a) Downloads model data from nmrtist.org
    b) Loads the peak picking model and fragments cropped from NMR spectrum
    c) Makes prediction
    d) Visualizes prediction
       Central point of each panel (point evaluated by ResNet) is marked in red, signal components (after deconvolution) are marked in blue.

    The use of model with external data:
    a) Normalize NMR spectrum, such that median signal amplitude is 100
    b) Crop spectrum fragment, such that the signal of interest is located in the central part of the image
    c) Normalize spectrum fragment amplitude: fragment = np.maximum(np.log(np.abs(fragment) + 1e-5) / np.log(1.4), 0)

    Model trained using generated data.

    Technical remarks:
    a) Model has been trained with Keras 2.3.1 and Tensorflow 2.0.0
    b) Zip archive with model data should be extracted in predict.py directory (nmrtist.org/static/public/publications/artina/models/ARTINA_peak_deconvolution.zip)

"""

import keras
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import wminkowski, pdist, squareform
import os
import shutil

# Download data
if not os.path.exists("example_input"):
    os.system("wget nmrtist.org/static/public/publications/artina/models/ARTINA_peak_deconvolution.zip")
    shutil.unpack_archive("ARTINA_peak_deconvolution.zip")

# Settings
show_only_examples_with_multiple_components = False

# Load the model
model = keras.models.load_model('model.h5', compile=False)

from keras.utils import plot_model
plot_model(model, to_file='model_architecture.png', show_layer_names=False, show_shapes=True)


for batch_id in range(1, 6):

    # Load example input batch (scale-space pyramid) and make predictions
    annotations = []
    for w in [0.5, 0.75, 1.00, 1.25, 1.50]:
        for h in [0.75, 1.00, 1.25]:
            with open(f"example_input/{batch_id:d}_{w:.2f}_{h:.2f}.pickle", "rb") as f:
                batch = pickle.load(f)
                annotations.append(model.predict_on_batch(batch))

    # Postprocess results
    tolerance = 1.0

    for i in range(0, 16):

        # Select best prediction or switch to single component if prediction is unstable
        coordinates = np.asarray([e[i] for e in annotations])
        coordinates_consistency = np.sum(squareform(pdist(coordinates, wminkowski, 2, 1.)) < tolerance, axis=0)
        best_candidate_id = np.argmax(coordinates_consistency)
        final_prediction = coordinates[best_candidate_id] if coordinates_consistency[best_candidate_id] >= 5 else np.zeros((9, ))

        # Calculate distance between predicted components
        is_more_than_1_component_present = np.max(squareform(pdist(final_prediction.reshape(3, 3), wminkowski, 2, tolerance))[0]) > tolerance

        # Visualize results
        if is_more_than_1_component_present or not show_only_examples_with_multiple_components:

            plt.figure(figsize=(4, 8))
            plt.contour(batch[i, :, :, 2], levels=30)
            plt.title(f'Fragment {i}')
            plt.plot(15.5, 31.5, 'r.', linewidth=4)

            for k in [0, 3, 6]:
                plt.plot(final_prediction[k + 2] + 15.5, final_prediction[k + 1] + 31.5, 'b.')

            plt.tight_layout()
            plt.show()
