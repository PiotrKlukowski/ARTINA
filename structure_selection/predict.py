"""

    The script:
    a) Downloads model data from nmrtist.org
    b) Loads feature vector associated with 10 structure candidates
    c) Builds ranking
    d) Presents the predicted ranking

    Binary files in "example_input" directory contain serialized features vectors of protein structure models (10 structure proposals x 100 proteins), calculated in the last ARTINA automated structure determination cycle.
    This script reproduces Backbone RMSD to reference that has been reported in Table S2.

    Feature ID | Quantity
    =====================
    0*         | Always zero for pairwise comparison of the same protein
    1          | Ramachandran plot statistics: residues in most favoured regions
    2          | Ramachandran plot statistics: residues in additionally allowed regions
    3          | Ramachandran plot statistics: residues in generously allowed regions
    4          | Ramachandran plot statistics: residues in disallowed regions
    5          | CYANA final structure calculation cycle: number of violated distance restraints per residue
    6          | CYANA final structure calculation cycle: number of violated van der Waals restraints per residue
    7          | CYANA final structure calculation cycle: number of violated angle restraints per residue
    8          | Number of residues in secondary structure elements (ALL)
    9          | Number of residues in secondary structure elements (H)
    10         | Number of residues in secondary structure elements (B)
    11         | Number of residues in secondary structure elements (3)
    12         | Number of residues in secondary structure elements (5)
    13         | Number of secondary structure elements (ALL)
    14         | Number of secondary structure elements (H)
    15         | Number of secondary structure elements (B)
    16         | Number of secondary structure elements (3)
    17         | Number of secondary structure elements (5)
    18-25      | Average number of short distance restraints per residue (each CYANA structure calculation cycle)
    26-33      | Average number of short distance restraints per residue (each CYANA structure calculation cycle)
    34-41      | Average number of short distance restraints per residue (each CYANA structure calculation cycle)
    42-43*     | Always zero for pairwise comparison of the same protein
    44-50      | Average number of violations below 0.5 A (per residue) in each CYANA structure calculation cycle
    51-54*     | Always zero for pairwise comparison of the same protein
    55-61      | Average number of violations below 0.5 - 3.0 A (per residue) in each CYANA structure calculation cycle
    62-63*     | Always zero for pairwise comparison of the same protein
    64-70      | Average number of violations above 3.0 A (per residue) in each CYANA structure calculation cycle
    71-78      | Cyana target function (each structure calculation cycle)
    79-86      | Backbone RMSD to mean (each structure calculation cycle)
    86-94      | Heavy atom RMSD to mean (each structure calculation cycle)
    95         | Average number of assigned NOESY peaks per residue (last structure calculation cycle)

    Features marked with * should be set to 0 for pairwise comparison presented in ARTINA manuscript.

    model_fold1 test proteins: 1SE9, 2JVD, 2K1G, 2K50, 2KFP, 2KJR, 2KKL, 2KL5, 2KOB, 2L05, 2L82, 2LA6, 2LEA, 2LFI, 2LML, 2LND, 2LRH, 2LTL, 2LX7, 2MDR
    model_fold2 test proteins: 2HEQ, 2JN8, 2JT1, 2K3A, 2K52, 2K5V, 2KD0, 2KHD, 2KIF, 2KKZ, 2KRS, 2L06, 2L3B, 2L8V, 2LAH, 2LK2, 2LN3, 2LTM, 2LXU, 2M7U, 2MB0, 1VDY, 6SVC
    model_fold3 test proteins: 2G0Q, 2JQN, 2K3D, 2K5D, 2KBN, 2KD1, 2KK8, 2KPN, 2KRT, 2KVO, 2L1P, 2L3G, 2L9R, 2LAK, 2LF2, 2LL8, 2LNA, 2LVB, 2MA6, 1VEE
    model_fold4 test proteins: 1PQX, 1T0Y, 1YEZ, 2B3W, 2JXP, 2K0M, 2K57, 2K75, 2KCD, 2KCT, 2KL6, 2KZV, 2L33, 2L7Q, 2LGH, 2M47, 2MK2, 1WQU
    model_fold5 test proteins: 2ERR, 2FB7, 2JRM, 2JVO, 2K1S, 2K53, 2KIW, 2LTA, 2M4F, 2M5O, 2MQL, 2N4B, 2RN7, 6FIP, 6GT7, 6SOW, 3JZK, KRAS4B, MH04

"""

from catboost import CatBoostClassifier
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# Download data
if not os.path.exists("example_input"):
    os.system("wget nmrtist.org/static/public/publications/artina/models/ARTINA_structure_ranking.zip")
    shutil.unpack_archive("ARTINA_structure_ranking.zip")

# Load the model
model = CatBoostClassifier(n_estimators=30, learning_rate=0.3, depth=10)
model.load_model("model_fold4.cbm")

# Load feature vectors associated with evaluated protein
# Note: last element of the vector stored in the file is structure RMSD to PDB reference
#       it is not used in prediction (line 72)
vs = []
for i in range(10):
    with open(f"example_input/2B3W_{i}.pickle", "rb") as f:
        vs.append(pickle.load(f))
vs = np.asarray(vs)

# Make pairwise comparisons and calculate scores
output = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        if i != j:
            feature_vector = vs[i, :-1] - vs[j, :-1]
            output[i, j] = model.predict_proba(feature_vector)[0]
scores = np.sum(output, axis=1)

# Print results
print(f'{"Structure ID":>20s}\t{"Score":>8s}\t{"RMSD to reference [A]":>30s}')
for i in np.argsort(scores)[::-1]:
    print(f'{i:20d}\t{scores[i]:8.2f}\t{vs[i, -1]:30.2f}')

# Visualize results
plt.figure(figsize=(9, 4.5))
plt.plot(scores, vs[:, -1], 'rx')
plt.xlabel("Ranking score")
plt.ylabel("RMSD to PDB reference [$\\mathrm{\AA}$]")
plt.minorticks_on()
plt.tight_layout()
plt.show()
