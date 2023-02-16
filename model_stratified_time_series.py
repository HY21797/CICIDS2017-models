import pandas as pd
import numpy as np
import os
import math
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, fbeta_score
import matplotlib.pyplot as plt

print("scikit-learn version: {}".format(sklearn.__version__))
print("Pandas version: {}".format(pd.__version__))
print("NumPy version: {}".format(np.__version__))


def sort_files(f):
    if (f.split("-")[0] == "Monday"):
        return 0
    if (f.split("-")[0] == "Tuesday"):
        return 1
    if (f.split("-")[0] == "Wednesday"):
        return 2
    if (f.split("-")[0] == "Thursday" and f.split("-")[2] == "Morning"):
        return 3
    if (f.split("-")[0] == "Thursday" and f.split("-")[2] == "Afternoon"):
        return 4
    if (f.split("-")[0] == "Friday" and f.split(".")[0].split("-")[2] == "Morning"):
        return 5
    if (f.split("-")[0] == "Friday" and f.split(".")[0].split("-")[3] == "PortScan"):
        return 6
    return 7


# Reading CSV files, and merging all of them into a single DataFrame
file_number = 0
root_folder = os.path.dirname(
    os.path.abspath(__file__)) + "/MachineLearningCVE/"
df = pd.DataFrame()
dfs = [None] * 8
for f in os.listdir(root_folder):
    file_number = file_number + 1
    print("Reading: ", f)
    dfs[sort_files(f)] = pd.read_csv(root_folder + f)
for x in range(file_number):
    df = pd.concat([df, dfs[x]])

# QUICK PREPROCESSING.
# Some classifiers do not like "infinite" (inf) or "null" (NaN) values.
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Label all Patator attacks as 'Patator', all Brute Force attacks as 'Brute Force'
# df = df[df[' Label'].str.contains("BENIGN|PortScan|Patator|Brute")==True]
df[' Label'] = df[' Label'].replace(['FTP-Patator', 'SSH-Patator'], 'Patator')
df.loc[df[' Label'].str.contains('Brute'), ' Label'] = 'Brute Force'
df = df[df[' Label'].isin(['BENIGN', 'PortScan', 'Patator', 'Brute Force'])]

# print(df[' Label'].unique())
# print(len(df)) #2445463
# print(len(df[df[' Label'] == 'BENIGN']))  # 2271320
# print(len(df[df[' Label'] == 'PortScan']))  # 158804
# print(len(df[df[' Label'] == 'Patator']))  # 13832
# print(len(df[df[' Label'] == 'Brute Force']))  # 1507

# Define the features used by the classifier
features = pd.Index([' Destination Port', ' Flow Duration', ' Total Fwd Packets',
                     ' Total Backward Packets', 'Total Length of Fwd Packets',
                     ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
                     ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
                     ' Fwd Packet Length Std', 'Bwd Packet Length Max',
                     ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
                     ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s',
                     ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
                     'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
                     ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
                     ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
                     ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
                     ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
                     ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
                     ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
                     ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
                     ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
                     ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
                     ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
                     ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
                     ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
                     'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
                     ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                     ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
                     ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
                     ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'])

b1 = df[df[' Label'] == 'BENIGN']
range1 = math.trunc(len(b1)/file_number)
b2 = df[df[' Label'] == 'PortScan']
range2 = math.trunc(len(b2)/file_number)
b3 = df[df[' Label'] == 'Patator']
range3 = math.trunc(len(b3)/file_number)
b4 = df[df[' Label'] == 'Brute Force']
range4 = math.trunc(len(b4)/file_number)
print("\nStratifing data\n")
new_df = pd.DataFrame()
for i in range(file_number):
    new_df = pd.concat([new_df, b1[i * range1: i * range1 + range1]])
    new_df = pd.concat([new_df, b2[i * range2: i * range2 + range2]])
    new_df = pd.concat([new_df, b3[i * range3: i * range3 + range3]])
    new_df = pd.concat([new_df, b4[i * range4: i * range4 + range4]])


class ConfusionMatrix(object):
    def getMultiPR(self, pred, target_data):
        tp = 0  # true positives
        tn = 0  # true negatives
        fp = 0  # false positive
        fn = 0  # false negatives
        for i in range(len(pred)):
            if pred[i] != 'BENIGN' and pred[i] == target_data.iloc[i]:  # True positive
                tp += 1
            if pred[i] != target_data.iloc[i] and target_data.iloc[i] == 'BENIGN':  # False positive
                fp += 1
            if pred[i] == 'BENIGN' and target_data.iloc[i] != pred[i]:  # False negative
                fn += 1
            if pred[i] == 'BENIGN' and target_data.iloc[i] == 'BENIGN':  # True negative
                tn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return [precision, recall]

    def getBinaryPR(self, pred, target_data, malicious, benign):
        tp = 0  # true positives
        tn = 0  # true negatives
        fp = 0  # false positive
        fn = 0  # false negatives
        for i in range(len(pred)):
            if pred[i] == malicious and target_data.iloc[i] == malicious:  # True positive
                tp += 1
            if pred[i] == malicious and target_data.iloc[i] == benign:  # False positive
                fp += 1
            if pred[i] == benign and target_data.iloc[i] == malicious:  # False negative
                fn += 1
            if pred[i] == benign and target_data.iloc[i] == benign:  # True negative
                tn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return [precision, recall]


cM = ConfusionMatrix()
X = new_df[features]
y = new_df[' Label']
tscv = TimeSeriesSplit(n_splits=file_number-1,
                       test_size=round(len(new_df)/file_number))

# for i, (train_index, test_index) in enumerate(tscv.split(X)):
#     print(f"Fold {i}:")
#     print(f"  Train: index={train_index}")
#     print(f"  Test:  index={test_index}")

accuracy_scores1 = []
f1_scores1 = []
p1 = []
r1 = []
p_port = []
r_port = []
p_patator = []
r_patator = []
p_brute = []
r_brute = []
count_1 = []
count1 = 0
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    multi_model = RandomForestClassifier(n_jobs=-2)
    multi_model.fit(X_train, y_train)
    y_pred = multi_model.predict(X_test)
    count1 = count1 + 1
    pr1 = cM.getMultiPR(y_pred, y_test)
    p1.append(pr1[0])
    r1.append(pr1[1])
    pr2 = cM.getBinaryPR(y_pred, y_test, 'PortScan', 'BENIGN')
    p_port.append(pr2[0])
    r_port.append(pr2[1])
    pr3 = cM.getBinaryPR(y_pred, y_test, 'Patator', 'BENIGN')
    p_patator.append(pr3[0])
    r_patator.append(pr3[1])
    pr4 = cM.getBinaryPR(y_pred, y_test, 'Brute Force', 'BENIGN')
    p_brute.append(pr4[0])
    r_brute.append(pr4[1])
    count_1.append(count1)
    accuracy_scores1.append(accuracy_score(y_test, y_pred))
    f1_scores1.append(fbeta_score(y_test, y_pred, average='macro', beta=1.0))

plt.subplot(2, 2, 1)
plt.plot(count_1, accuracy_scores1, label="accuracy")
plt.plot(count_1, f1_scores1, label="f1")
plt.xlabel("Test Fold")
plt.legend()
plt.title('Multiclass Classifier')
plt.subplot(2, 2, 2)
plt.plot(r1, p1, label="Overall")
plt.plot(r_port, p_port, label="PortScan")
plt.plot(r_patator, p_patator, label="Patator")
plt.plot(r_brute, p_brute, label="Brute Force")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title('Multiclass Classifier')
print("After time series cross validation for the multiclass model:\n")
print(" Accuracy: {:3f}".format(sum(accuracy_scores1)/len(accuracy_scores1)))
print(" F1-score: {:3f}".format(sum(f1_scores1)/len(f1_scores1)))


new_df['GT'] = np.where(new_df[' Label'] == 'BENIGN', 'Benign', 'Malicious')
y_binary = new_df['GT']
tscv2 = TimeSeriesSplit(n_splits=file_number-1,
                        test_size=round(len(new_df)/file_number))
accuracy_scores2 = []
f1_scores2 = []
p2 = []
r2 = []
count_2 = []
count2 = 0
for train_index, test_index in tscv2.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_binary.iloc[train_index], y_binary.iloc[test_index]
    binary_model = RandomForestClassifier(n_jobs=-2)
    binary_model.fit(X_train, y_train)
    y_pred = binary_model.predict(X_test)
    pr = cM.getBinaryPR(y_pred, y_test, 'Malicious', 'Benign')
    p2.append(pr[0])
    r2.append(pr[1])
    count2 = count2 + 1
    count_2.append(count2)
    accuracy_scores2.append(accuracy_score(y_test, y_pred))
    f1_scores2.append(fbeta_score(y_test, y_pred, average='macro', beta=1.0))
plt.subplot(2, 2, 3)
plt.plot(count_2, accuracy_scores2, label="accuracy")
plt.plot(count_2, f1_scores2, label="f1")
plt.xlabel("Test Fold")
plt.legend()
plt.title('Binary Classifier')
plt.subplot(2, 2, 4)
plt.plot(r2, p2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title('Binary Classifier')
print("\nAfter time series cross validation for the binary model:")
print(" Accuracy: {:3f}".format(sum(accuracy_scores2)/len(accuracy_scores2)))
print(" F1-score: {:3f}".format(sum(f1_scores2)/len(f1_scores2)))
plt.show()


# Accuracy: 0.998966
# F1-score: 0.984150
# Accuracy: 0.998908
# F1-score: 0.995886
