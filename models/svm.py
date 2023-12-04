import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import svm


# read data and apply one-hot encoding
datasets = ["data/marbling_dataset_v2/marbling_features_shape_dataset.csv",
            "data/marbling_dataset_v2/marbling_features_lbp_dataset.csv",
            "data/marbling_dataset_v2/marbling_features_sift_dataset.csv",
            "data/marbling_dataset_v2/marbling_features_sift_rgb_dataset.csv",
            "data/marbling_dataset_v2/marbling_features_orb_dataset.csv"]

for dataset in datasets:
    data = pd.read_csv(dataset, header=None)

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


    # n_X = (X-X.mean())/X.std()  # Rescale to mean = 0; sd ~ 1
    # X = n_X
    # # n_X.to_csv("data/marbling_dataset/marbling_dataset_n_X.csv")

    # mmn_X = (X-X.min())/(X.max()-X.min())  # Normalized (rescale  0 to 1)
    # X = mmn_X
    # # mmn_X.to_csv("data/marbling_dataset/marbling_dataset_mmn_X.csv")

    scaler = MinMaxScaler(feature_range=(-1, 1))  # Rescale -1 to 1
    mmn_sk_X = pd.DataFrame(scaler.fit_transform(X))
    X = mmn_sk_X
    # mmn_sk_X.to_csv("data/marbling_dataset/marbling_dataset_mmn_sk_X.csv")

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=1)

    # From https://www.baeldung.com/cs/svm-multiclass-classification
    kernel = 'rbf'
    gamma = 5.0
    degree = 10
    C = 10
    svm_class = svm.SVC(kernel='rbf', gamma=5.0, degree=10, C=10).fit(X_train, y_train)

    # kernel = 'rbf'
    # gamma = 5.0
    # degree = 10
    # C = 10
    # svm_class = svm.SVC(kernel='poly', degree=10, C=100).fit(X_train, y_train)

    y_pred = svm_class.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("\n######################################################################")
    print("SVM Model for", dataset)

    print("  Kernel: ", kernel)
    print("  gamma:  ", gamma, "(used only for 'rbf' kernel)")
    print("  degree: ", degree, "(used only for 'poly' kernel)")
    print("  C:      ", C)

    print("\n  RESULTS")
    print("    Confusion Matrix:\n",cm)
    print("    Accuracy: ", acc)
    print("    Precision:", prec)
    print("    Recall:   ", rec)
    print("    F-1 score ", f1)
    print("######################################################################\n")