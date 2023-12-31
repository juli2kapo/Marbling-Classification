import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import onnx, onnxruntime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# From https://machinelearningmastery.com/building-a-multiclass-classification-model-in-pytorch/

#TODO: agregar el resto de las inicializaciones
def init_normal(m):
    # https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
    if type(m) == nn.Linear:
        # nn.init.uniform_(m.weight)

        # n = m.in_features
        # y = 1.0/np.sqrt(n)
        # m.weight.data.uniform_(-y, y)
        # m.bias.data.fill_(0)

        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
        # m.bias.data should be 0
        m.bias.data.fill_(0)

class Multiclass(nn.Module):
    def __init__(self, input_net, output_net):
        super().__init__()
        self.hidden1 = nn.Linear(input_net, 150)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(150, 150)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(150, 50)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(50, output_net)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.output(x)
        return x
    
# read data and apply one-hot encoding
# data = pd.read_csv("data/marbling_dataset_v2/marbling_features_shape_dataset.csv", header=None)
# data = pd.read_csv("data/marbling_dataset_v2/marbling_features_lbp_dataset.csv", header=None)

# read data and apply one-hot encoding
# datasets = ["data/marbling_dataset_v2/marbling_features_shape_dataset.csv",
#             "data/marbling_dataset_v2/marbling_features_lbp_dataset.csv",
#             "data/marbling_dataset_v2/marbling_features_sift_dataset.csv",
#             "data/marbling_dataset_v2/marbling_features_sift_rgb_dataset.csv",
#             "data/marbling_dataset_v2/marbling_features_orb_dataset.csv"
#            ]

datasets = ["shape",
            "lbp",
            "sift",
            "sift_rgb",
            "orb"
           ]

min_max_scale = False

# prepare model and training parameters
n_epochs = 500
batch_size = 10


for dataset in datasets:
    path_to_dataset = f"data/marbling_dataset_v2/marbling_features_{dataset}_dataset.csv"

    print(f"Loading dataset: {path_to_dataset}")

    data = pd.read_csv(path_to_dataset, header=None)

    len_features = len(data.columns)-1

    X = data.iloc[:, 1:len_features]
    y = data.iloc[:, len_features]


    # 012; 34; 567
    # 0123; 4567
    y.replace(to_replace=1, value=0, inplace=True)
    y.replace(to_replace=2, value=0, inplace=True)
    y.replace(to_replace=3, value=0, inplace=True)

    y.replace(to_replace=4, value=1, inplace=True)
    y.replace(to_replace=5, value=1, inplace=True)
    y.replace(to_replace=6, value=1, inplace=True)
    y.replace(to_replace=7, value=1, inplace=True)

    input_net = len_features-1
    output_net = len(y.unique())

    if min_max_scale:
        # n_X = (X-X.mean())/X.std()  # Rescale to mean = 0; sd ~ 1
        # mmn_X = (X-X.min())/(X.max()-X.min())  # Normalized (rescale  0 to 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))  # Rescale -1 to 1
        mmn_sk_X = pd.DataFrame(scaler.fit_transform(X))
        X = mmn_sk_X

    # One hot encoder for "label" numerical column
    df_dummy = pd.get_dummies(y, columns=['label_column', ])

    # ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
    # y = ohe.fit_transform(y.values.reshape(-1,1))

    # convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(df_dummy.values, dtype=torch.float32)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=1)

    torch.manual_seed(1)
    # loss metric and optimizer
    model = Multiclass(input_net, output_net)
    # use the modules apply function to recursively apply the initialization
    # model.apply(init_normal)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    batches_per_epoch = len(X_train) // batch_size

    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []

    # training loop
    with tqdm.trange(n_epochs, unit="Epoch", mininterval=0) as bar:
        bar.set_description(f"Epochs")
    # for epoch in range(n_epochs):
        for i in bar:
            epoch_loss = []
            epoch_acc = []
            # set model in training mode and run through each batch
            model.train()
            # with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            #     bar.set_description(f"Epoch {epoch}")
            for i in range(batches_per_epoch):
                # take a batch
                start = i * batch_size
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # compute and store metrics
                acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )
            # set model in evaluation mode and run through the test set
            model.eval()
            y_pred = model(X_test)
            ce = loss_fn(y_pred, y_test)
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
            ce = float(ce)
            acc = float(acc)
            train_loss_hist.append(np.mean(epoch_loss))
            train_acc_hist.append(np.mean(epoch_acc))
            test_loss_hist.append(ce)
            test_acc_hist.append(acc)
            if acc >= best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
        # print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")

    # Restore best model and save it
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), f"outputs/mlp_{dataset}_{output_net}classes.pt")

    # Export the model from PyTorch to ONNX
    torch_input = torch.randn(1, input_net)
    onnx_program = torch.onnx.dynamo_export(model, torch_input)
    onnx_program.save(f"outputs/mlp_{dataset}_{output_net}classes.onnx")
    # Check model
    onnx_model = onnx.load(f"outputs/mlp_{dataset}_{output_net}classes.onnx")
    onnx.checker.check_model(onnx_model)

    # Eval PyTorch model
    model.eval()
    y_pred = model(X_test)

    y_pred_np = torch.argmax(y_pred, 1).detach().numpy()
    y_test_np = torch.argmax(y_test, 1).detach().numpy()

    cm = confusion_matrix(y_test_np, y_pred_np)
    acc = accuracy_score(y_test_np, y_pred_np)
    prec = precision_score(y_test_np, y_pred_np, average='macro')
    rec = recall_score(y_test_np, y_pred_np, average='macro')
    f1 = f1_score(y_test_np, y_pred_np, average='macro')

    print("\n######################################################################")
    print("MLP PYTORCH MODEL FOR", dataset)
    print("  Confusion Matrix:\n",cm)
    print("  Accuracy: ", acc)
    print("  Precision:", prec)
    print("  Recall:   ", rec)
    print("  F-1 score ", f1)
    print("######################################################################\n\n")


    # Eval ONNX model
    ort_session = onnxruntime.InferenceSession(f"outputs/mlp_{dataset}_{output_net}classes.onnx", providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    y_pred_np = []
    for x in X_test:
        ort_input = {ort_session.get_inputs()[0].name: np.expand_dims(to_numpy(x),0)}
        ort_output = ort_session.run(None, ort_input)
        y_pred_np.append(np.argmax(ort_output[0], axis=1))
    
    # y_pred_np = torch.argmax(y_pred, 1).detach().numpy()
    # y_test_np = torch.argmax(y_test, 1).detach().numpy()

    cm = confusion_matrix(y_test_np, y_pred_np)
    acc = accuracy_score(y_test_np, y_pred_np)
    prec = precision_score(y_test_np, y_pred_np, average='macro')
    rec = recall_score(y_test_np, y_pred_np, average='macro')
    f1 = f1_score(y_test_np, y_pred_np, average='macro')

    print("\n######################################################################")
    print("MLP ONNX MODEL FOR", dataset)
    print("  Confusion Matrix:\n",cm)
    print("  Accuracy: ", acc)
    print("  Precision:", prec)
    print("  Recall:   ", rec)
    print("  F-1 score ", f1)
    print("######################################################################\n\n")
