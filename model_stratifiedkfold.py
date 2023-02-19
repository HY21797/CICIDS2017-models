import pandas as pd
import numpy as np
import os
import math
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, fbeta_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

print("scikit-learn version: {}".format(sklearn.__version__))
print("Pandas version: {}".format(pd.__version__))
print("NumPy version: {}".format(np.__version__))


# Reading CSV files, and merging all of them into a single DataFrame
file_number = 0
root_folder = os.path.dirname(
    os.path.abspath(__file__)) + "/MachineLearningCVE/"
df = pd.DataFrame()
for f in os.listdir(root_folder):
    file_number = file_number + 1
    print("Reading: ", f)
    df = pd.concat([df, pd.read_csv(root_folder + f)])

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


class ConfusionMatrix(object):
    # micro average
    def getMultiPR(self, pred, target_data):
        tp = 0  # true positives
        fp = 0  # false positive
        fn = 0  # false negatives
        for i in range(len(pred)):
            if pred[i] != 'BENIGN' and pred[i] == target_data.iloc[i]:  # True positive
                tp += 1
            if pred[i] != target_data.iloc[i] and target_data.iloc[i] == 'BENIGN':  # False positive
                fp += 1
            if pred[i] == 'BENIGN' and target_data.iloc[i] != pred[i]:  # False negative
                fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return [precision, recall]

    def getBinaryPR(self, pred, target_data, malicious):
        tp = 0  # true positives
        fp = 0  # false positive
        fn = 0  # false negatives
        for i in range(len(pred)):
            if pred[i] == malicious and target_data.iloc[i] == malicious:  # True positive
                tp += 1
            if pred[i] == malicious and target_data.iloc[i] != malicious:  # False positive
                fp += 1
            if pred[i] != malicious and target_data.iloc[i] == malicious:  # False negative
                fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return [precision, recall]


cM = ConfusionMatrix()
X = df[features]
y = df[' Label']
skf = StratifiedKFold(n_splits=file_number-1, random_state=1, shuffle=True)

accuracy_scores1 = []
f1_scores1 = []
p1 = []
r1 = []
labels = ['PortScan', 'Patator', 'Brute Force']
p2 = [[] for i in range(len(labels))]
r2 = [[] for i in range(len(labels))]
count_1 = []
count1 = 0
multi_model = RandomForestClassifier(n_jobs=-2)

print("Stratified k-fold cross validation for the multiclass model:")
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    multi_model.fit(X_train, y_train)
    y_pred = multi_model.predict(X_test)
    p = 0
    r = 0
    for i in range(len(labels)):
        pr = cM.getBinaryPR(y_pred, y_test, labels[i])
        p2[i].append(pr[0])
        r2[i].append(pr[1])
        p = p + pr[0]
        r = r + pr[1]
    p1.append(p/3)
    r1.append(r/3)
    count1 = count1 + 1
    count_1.append(count1)
    accuracy_scores1.append(accuracy_score(y_test, y_pred))
    f1_scores1.append(fbeta_score(y_test, y_pred, average='macro', beta=1.0))

plt.subplot(2, 2, 1)
r1, p1 = zip(*sorted(zip(r1, p1)))
plt.plot(r1, p1, label="Overall")
for i in range(len(labels)):
    r2[i], p2[i] = zip(*sorted(zip(r2[i], p2[i])))
    plt.plot(r2[i], p2[i], label=labels[i])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title('Multiclass Classifier')
plt.subplot(2, 2, 2)
plt.plot(count_1, accuracy_scores1, label="accuracy")
plt.plot(count_1, f1_scores1, label="f1")
plt.xlabel("Test Fold")
plt.legend()
plt.title('Multiclass Classifier')
print(" Accuracy: {:3f}".format(sum(accuracy_scores1)/len(accuracy_scores1)))
print(" F1-score: {:3f}".format(sum(f1_scores1)/len(f1_scores1)))


df['GT'] = np.where(df[' Label'] == 'BENIGN', 'Benign', 'Malicious')
y_binary = df['GT']
skf2 = StratifiedKFold(n_splits=file_number-1, random_state=1, shuffle=True)
accuracy_scores2 = []
f1_scores2 = []
y_real2 = []
y_proba2 = []
count_2 = []
count2 = 0
binary_model = RandomForestClassifier(n_jobs=-2)

plt.subplot(2, 2, 3)
print("Stratified k-fold cross validation for the binary model:")
for train_index, test_index in skf2.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_binary.iloc[train_index], y_binary.iloc[test_index]
    binary_model.fit(X_train, y_train)
    y_pred = binary_model.predict(X_test)
    pred_proba = binary_model.predict_proba(X_test)
    precision, recall, thresholds = precision_recall_curve(
        y_test, pred_proba[:, 1], pos_label='Malicious')
    plt.plot(recall, precision, lw=1, alpha=0.3,
             label='PR fold %d (AUC = %0.2f)' % (count2, average_precision_score(y_test, pred_proba[:, 1], pos_label='Malicious')))
    y_real2.append(y_test)
    y_proba2.append(pred_proba[:, 1])
    count2 = count2 + 1
    count_2.append(count2)
    accuracy_scores2.append(accuracy_score(y_test, y_pred))
    f1_scores2.append(fbeta_score(y_test, y_pred, average='macro', beta=1.0))

y_real2 = np.concatenate(y_real2)
y_proba2 = np.concatenate(y_proba2)
precision2, recall2, _ = precision_recall_curve(
    y_real2, y_proba2, pos_label='Malicious')
plt.plot(recall2, precision2, color='b',
         label=r'Precision-Recall (AUC = %0.2f)' % (
             average_precision_score(y_real2, y_proba2, pos_label='Malicious')),
         lw=2, alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Binary Classifier')
plt.subplot(2, 2, 4)
plt.plot(count_2, accuracy_scores2, label="accuracy")
plt.plot(count_2, f1_scores2, label="f1")
plt.xlabel("Test Fold")
plt.legend()
plt.title('Binary Classifier')
print(" Accuracy: {:3f}".format(sum(accuracy_scores2)/len(accuracy_scores2)))
print(" F1-score: {:3f}".format(sum(f1_scores2)/len(f1_scores2)))
plt.suptitle("Stratified K-fold Cross Validation")
plt.show()


# Accuracy: 0.999322
# F1-score: 0.995763
# Accuracy: 0.999323
# F1-score: 0.997442
