"""


    The script:
    a) Downloads model data from nmrtist.org
    b) Loads graph that represents protein assignment obtained from FLYA run
    c) Loads GNN model and performs node regression to identify expected value of chemical shifts not assigned by FLYA
    d) Presents predicted shifts in table form


    Node feature ID | Quantity
    ==========================

    0-19            | Amino acid type (one-hot-encoding, 20 elements)

                      Example: MET
                      0    0    0    0    0    0    0    0    0    0    0    0    1    0    0    0    0    0    0    0
                      ALA, ARG, ASN, ASP, CYS, GLU, GLN, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL

    20-112          | Chemical shift type (one-hot-encoding, 93 elements)

                      Order of chemical shift types in the encoding:
                      O, OD, OG, OG1, OE, OH, OD1, OD2, OE1, OE2, SE, SG, SD, H, HA, HA2, HA3, HB, HB2, HB3, HG, HG1, HG12,
                      HG13, HG2, HG3, HD1, HD2, HD3, HE, HE1, HE2, HE3, HH, HH11, HH12, HH2, HH21, HH22, HD21, HD22, HE21,
                      HE22, HZ, HZ2, HZ3, C, CA, CB, CG, CG1, CG2, CD, CD1, CD2, CE, CE1, CE2, CE3, CZ, CZ1, CH2, CZ2, CZ3,
                      N, NE, NE2, NH1, NH2, ND2, NZ, NE1, ND1, QA, QB, QG, QQG, QG1, QG2, QD, QD1, QD2, QQD, QE, QH1, QH2,
                      QQH, QE2, QZ, QCG, QCD, QCE, QNH

    113             | Mean chemical shift value stored in BMRB database (calculated for amino acid type and chemical shift type encoded in features 0-112)
    114             | Standard deviation of chemical shift values stored in BMRB database (calculated for amino acid type and chemical shift type encoded in features 0-112)
    115-144         | Chemical shift values identified through BMRB fragment matching
                      To select fragments minimize mean absolute error between fragments extracted from BMRB database and shifts assigned by FLYA (feature 145 of neighbouring nodes).
                      Use sequence fragments of length:
                       - 3 (residue -1, residue 0, residue +1), features 115 - 124
                       - 2 (residue 0, residue +1), features 125 - 134
                       - 2 (residue -1, residue 0), features 135 - 144


    145             | Chemical shift assigned by FLYA (0 if shift hasn't been assigned)

    Model trained using BMRB records not included in the ARTINA benchmark dataset.
    Fragments for shift matching have been extracted from BMRB records not included in the ARTINA benchmark dataset.

    Technical remarks:
    a) Model has been trained with PyTorch 1.6.0 and PyTorch Geometric 1.6.1
    b) Zip archive with model data should be extracted in predict.py directory (nmrtist.org/static/public/publications/artina/models/ARTINA_shift_prediction.zip)


"""

import pickle
import torch
from torch_geometric.nn import  GENConv, DeepGCNLayer
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
import os
import shutil


# Model used to make prediction (DeeperGCN: All You Need to Train Deeper GCNs, https://arxiv.org/abs/2006.07739)
class GNNModel(torch.nn.Module):

    def __init__(self):
        super(GNNModel, self).__init__()
        num_features = 146
        num_res_blocks = 25
        repr_len = 256

        # Node encoder
        self.node_encoder = Linear(num_features, repr_len)

        # Graph residual connections
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_res_blocks + 1):
            conv = GENConv(repr_len, repr_len, aggr='softmax', t=0.01, learn_t=False, num_layers=2, norm='layer')
            norm = LayerNorm(repr_len, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1, ckpt_grad=False)
            self.layers.append(layer)

        # Output layer
        self.fc1 = Linear(repr_len, int(repr_len / 2))
        self.output_layer = Linear(int(repr_len / 2), 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x[:, 121:] = x[:, 121:] / 100. - 0.5

        x = self.node_encoder(x)
        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))

        x = self.fc1(x)
        x = F.relu(x)

        return self.output_layer(x)


# Download data
if not os.path.exists("example_input"):
    os.system("wget nmrtist.org/static/public/publications/artina/models/ARTINA_shift_prediction.zip")
    shutil.unpack_archive("ARTINA_shift_prediction.zip")

# Load model
model = GNNModel()
model.load_state_dict(torch.load("model.torch", map_location="cpu"))


# Load example input
with open("example_input/2B3W_graph.pickle", "rb") as f:
    graph = pickle.load(f)
unknown_shifts = graph.x[:, -1].detach().numpy() == 0

with open("example_input/2B3W_labels.pickle", "rb") as f:
    labels = pickle.load(f)

# Predict on batch
output = model(graph).detach().numpy().reshape(-1,)

# Print results (only unknown shifts, where feature 145 == 0)
output = output[unknown_shifts]
labels = labels[unknown_shifts]

print(f'{"Residue ID":>10s}\t{"Amino Acid":>10s}\t{"Shift label":>12s}\t{"Predicted shift value":>20s}')
for i in range(len(output)):
    if "C" not in labels[i][2] and "N" not in labels[i][2]:
        print(f'{labels[i][0]:>10s}\t{labels[i][1]:>10s}\t{labels[i][2]:>12s}\t{output[i]/10.0:20.4f}')
    else:
        print(f'{labels[i][0]:>10s}\t{labels[i][1]:>10s}\t{labels[i][2]:>12s}\t{output[i]:20.4f}')
