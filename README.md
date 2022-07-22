# Rapid protein assignments and structures from raw NMR spectra with the deep learning technique ARTINA

### Overview 
ARTINA is a deep learning-based method for NMR spectra analysis, resonance assignment, and protein structure determination by NMR spectroscopy. Using as input NMR spectra and the protein sequence, the method identifies automatically (strictly without any human intervention): cross-peak positions, chemical shift assignments, upper limit distance restraints, and the protein structure. ARTINA deep learning models have been trained with over 600 000 cross-peak examples from more than 1300 2D/3D/4D spectra. The method demonstrated its ability to solve structures with a median backbone RMSD of 1.44 Å to PDB reference, and identified correctly 91.36% of the chemical shift assignments.

### Availability 
* The method is available as an open web server (https://nmrtist.org), which can be accessed using popular web browsers (i.e. Chrome, Firefox, Safari, Edge). No software installation/configuration is required. The computation is executed on the server side, releasing the users from the necessity of maintaining any cluster computer infrastructure. 
* Machine learning models used by ARTINA are available in serialized form, together with: (a) exemplary input data, (b) specification of the model input/output, (c) model architecture schemes, (d) python implementation of the model prediction step with the example input.     

### Example input data
We provide 8 experimental datasets, which allow 8 protein structures to be automatically determined with ARTINA using the open web server. The below list contains the download links of the source data and example (demo) results:  
* Dataset 1: 109-residue β-barrel protein [[download data](https://nmrtist.org/media/examples/datasets/target_1_input.zip), [output example](https://nmrtist.org/static/public/examples/ARTINA/ARTINA_dataset1.html)]
* Dataset 2: 134-residue 3-layer (αβα) sandwich protein [[download data](https://nmrtist.org/media/examples/datasets/target_2_input.zip), [output example](https://nmrtist.org/static/public/examples/ARTINA/ARTINA_dataset2.html)]
* Dataset 3: 140-residue α horseshoe protein [[download data](https://nmrtist.org/media/examples/datasets/target_3_input.zip), [output example](https://nmrtist.org/static/public/examples/ARTINA/ARTINA_dataset3.html)]
* Dataset 4: 99-residue 2-layer (αβ) sandwich protein [[download data](https://nmrtist.org/media/examples/datasets/target_4_input.zip), [output example](https://nmrtist.org/static/public/examples/ARTINA/ARTINA_dataset4.html)]
* Dataset 5: 97-residue 2-layer (αβ) sandwich protein [[download data](https://nmrtist.org/media/examples/datasets/target_5_input.zip), [output example](https://nmrtist.org/static/public/examples/ARTINA/ARTINA_dataset5.html)]
* Dataset 6: 99-residue 3-layer (αβα) sandwich protein [[download data](https://nmrtist.org/media/examples/datasets/target_6_input.zip), [output example](https://nmrtist.org/static/public/examples/ARTINA/ARTINA_dataset6.html)]
* Dataset 7: 83-residue 2-layer (αβ) sandwich protein [[download data](https://nmrtist.org/media/examples/datasets/target_7_input.zip), [output example](https://nmrtist.org/static/public/examples/ARTINA/ARTINA_dataset7.html)]
* Dataset 8: 101-residue protein in complex with RNA [[download data](https://nmrtist.org/media/examples/datasets/target_8_input.zip), [output example](https://nmrtist.org/static/public/examples/ARTINA/ARTINA_dataset8.html)]

Each dataset contains the protein sequence and set of NMR spectra that is suitable for automated structure determination with ARTINA. 

Example input data for individual models is available for download: 
* pp-ResNet (https://nmrtist.org/static/public/publications/artina/models/ARTINA_peak_picking.zip)
* deconv-ResNet (https://nmrtist.org/static/public/publications/artina/models/ARTINA_peak_deconvolution.zip)
* shift refinement model (https://nmrtist.org/static/public/publications/artina/models/ARTINA_shift_prediction.zip)
* structure ranking model (https://nmrtist.org/static/public/publications/artina/models/ARTINA_structure_ranking.zip) 

Python scripts available in this git repository download automatically model binaries and the example input data using the above links.  

### Documentation and instructions to use

Documentation of the open web server is available online (https://nmrtist.org/home-blog). It contains step-by-step tutorials guiding users through the process of data upload and submission of an automated protein structure calculation job. 
Additionally, a video tutorial that presents the use of ARTINA can be downloaded from https://nmrtist.org/static/public/publications/artina/Movie_S5.mp4. 

Specifications of individual machine learning models are available in the `{peak_picking, peak_deconvolution, shift_prediction, structure_selection}/predict.py` files. They contain information about model inputs, and present example predictions with the model.
After configuration of the Python environment (see System requirements) each `predict.py` script can be executed independently.      


### System requirements

The online implementation of ARTINA can be accessed from any modern web browser (Chrome, Firefox, Safari, Edge). There are no hardware requirements, as the computational jobs are executed on the server side. 

To use individual models and scripts available in this repository, configuration of the Python environment is required. We recommend to use `Anaconda 5.3.0`. 
* The peak picking and deconvolution scripts require `tensorflow 2.1` and `keras 2.3.1` ([installation guide](https://www.tensorflow.org/install))
* The shift refinement script requires: `torch 1.6.0`, `torch-scatter 2.0.5`, `torch-sparse 0.6.7`, `torch-cluster 1.5.7`, `torch-spline-conv 1.2.0`, and `torch-geometric 1.6.1` ([installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html))
* The structure ranking model requires `catboost 0.23` ([installation guide](https://catboost.ai/en/docs/installation/python-installation-method-pip-install))

 
###  License
The code available in this repository is available under the **Apache 2.0 License**.


### Reference
* Klukowski, P., Riek, R., Güntert, P. (2022), Rapid protein assignments and structures from raw NMR spectra with the deep learning technique ARTINA, arxiv.org/abs/2201.12041
  



 