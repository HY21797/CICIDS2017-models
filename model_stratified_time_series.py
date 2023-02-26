import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from functions import print_versions, sort_files, process_data
from functions import generate_new_df, multiclass_cross_validation
from functions import plot_multiclass_results, binary_cross_validation
from functions import plot_cv, plot_all_feature_importances
from functions import binary_time_cross_validation, get_features_without_repeat
from functions import multiclass_hp_tuning, binary_hp_tuning

print_versions()

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
df = process_data(df)


# print(len(df)) #2445463
# print(len(df[df[' Label'] == 'BENIGN']))  # 2271320
# print(len(df[df[' Label'] == 'PortScan']))  # 158804
# print(len(df[df[' Label'] == 'Patator']))  # 13832
# print(len(df[df[' Label'] == 'Brute Force']))  # 1507


# Stratified the data
new_df = generate_new_df(df, file_number)


# threshold: 0.01
# 32 features
features = pd.Index(['PSH Flag Count', 'Total Length of Fwd Packets', 'Packet Length Mean',
                     'Init_Win_bytes_forward', 'Packet Length Variance', 'Packet Length Std',
                     'Avg Bwd Segment Size', 'Average Packet Size', 'Bwd Packets/s',
                     'min_seg_size_forward', 'Flow Bytes/s', 'Subflow Fwd Bytes',
                     'Bwd Packet Length Min', 'Total Length of Bwd Packets',
                     'Fwd Packet Length Max', 'Destination Port', 'Bwd Packet Length Mean',
                     'Init_Win_bytes_backward', 'Total Fwd Packets', 'Flow IAT Max',
                     'ACK Flag Count', 'Fwd Header Length', 'Flow Duration',
                     'Subflow Fwd Packets', 'Subflow Bwd Bytes', 'Avg Fwd Segment Size',
                     'Max Packet Length', 'Fwd Packet Length Mean', 'Bwd Header Length',
                     'Fwd IAT Min', 'Fwd IAT Max', 'Bwd Packet Length Max'],
                    dtype='object')

X = new_df[features]
y = new_df['Label']
labels = ['PortScan', 'Patator', 'Brute Force']
tscv = TimeSeriesSplit(n_splits=file_number-1,
                       test_size=round(len(new_df)/file_number))

print("Visualizing cross-validation behavior for the multiclass model")
plot_cv(tscv, X, y, file_number-1,
        'Stratified Time Series Split - Multiclass Classifier')

# multi_model = RandomForestClassifier(n_jobs=-1, random_state=1)
# print('Hyperparameter tuning for the multiclass classifier')
# multiclass_hp_tuning(tscv, X, y)

multi_model = RandomForestClassifier(n_estimators=59, criterion='gini', max_depth=22,
                                     min_samples_split=13, min_samples_leaf=1,
                                     max_features=5, n_jobs=-1, random_state=1)

print("Stratified time series cross validation for the multiclass model")
results = multiclass_cross_validation(
    multi_model, tscv.split(X), X, y, labels)
# [r1, p1, r2, p2, count_1, accuracy_scores, f1_scores]
suptitle = 'Stratified Time Series Cross Validation'
plot_multiclass_results(results, labels, suptitle)

print("Training the multiclass classifier")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=round(len(new_df)/file_number), shuffle=False)
multi_model.fit(X_train, y_train)

print("Plotting the feature importances for the multiclass classifier")
plot_all_feature_importances(
    X_test, y_test, multi_model, features, 'Multiclass Classifier')


# threshold: 0.015
# 27 features
features2 = pd.Index(['PSH Flag Count', 'Packet Length Variance',
                      'Total Length of Fwd Packets', 'Packet Length Mean',
                      'Packet Length Std', 'min_seg_size_forward', 'Init_Win_bytes_forward',
                      'Bwd Packets/s', 'Avg Bwd Segment Size', 'Bwd Packet Length Mean',
                      'Flow Bytes/s', 'Fwd Packet Length Max', 'Bwd Packet Length Min',
                      'Destination Port', 'Total Length of Bwd Packets', 'Subflow Fwd Bytes',
                      'Average Packet Size', 'ACK Flag Count', 'Subflow Bwd Bytes',
                      'Total Fwd Packets', 'Init_Win_bytes_backward', 'Flow IAT Max',
                      'Bwd Header Length', 'Avg Fwd Segment Size', 'Flow Duration',
                      'Fwd IAT Mean', 'Max Packet Length'],
                     dtype='object')

new_df['GT'] = np.where(new_df['Label'] == 'BENIGN', 'Benign', 'Malicious')
X_binary = new_df[features2]
y_binary = new_df['GT']
tscv2 = TimeSeriesSplit(n_splits=file_number-1,
                        test_size=round(len(new_df)/file_number))

print("Visualizing cross-validation behavior for the binary model")
plot_cv(tscv2, X_binary, y_binary, file_number-1,
        'Stratified Time Series Split - Binary Classifier')

# binary_model = RandomForestClassifier(n_jobs=-1, random_state=1)
# print('Hyperparameter tuning for the binary classifier')
# binary_hp_tuning(tscv2, X_binary, y_binary)

binary_model = RandomForestClassifier(n_estimators=161, criterion='entropy', max_depth=81,
                                      min_samples_split=18, n_jobs=-1, random_state=1)

print("Stratified time series cross validation for the binary model")
suptitle2 = "Stratified Time Series Cross Validation"
binary_cross_validation(binary_model, tscv2.split(
    X_binary), X_binary, y_binary, suptitle2)

print("Training the binary classifier")
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_binary, y_binary, test_size=round(len(new_df)/file_number), shuffle=False)
binary_model.fit(X_train2, y_train2)

print("Plotting the feature importances for the binary classifier")
plot_all_feature_importances(
    X_test2, y_test2, binary_model, features2, 'Binary Classifier')


# # Remove 'Fwd Header Length.1'(Repeat)
# features = get_features_without_repeat()

# # threshold: 0.005
# # 39 features
# features = pd.Index(['PSH Flag Count', 'Total Length of Fwd Packets', 'Packet Length Mean',
#                      'Init_Win_bytes_forward', 'Packet Length Variance', 'Packet Length Std',
#                      'Avg Bwd Segment Size', 'Average Packet Size', 'Bwd Packets/s',
#                      'min_seg_size_forward', 'Flow Bytes/s', 'Subflow Fwd Bytes',
#                      'Bwd Packet Length Min', 'Total Length of Bwd Packets',
#                      'Fwd Packet Length Max', 'Destination Port', 'Bwd Packet Length Mean',
#                      'Init_Win_bytes_backward', 'Total Fwd Packets', 'Flow IAT Max',
#                      'ACK Flag Count', 'Fwd Header Length', 'Flow Duration',
#                      'Subflow Fwd Packets', 'Subflow Bwd Bytes', 'Avg Fwd Segment Size',
#                      'Max Packet Length', 'Fwd Packet Length Mean', 'Bwd Header Length',
#                      'Fwd IAT Min', 'Fwd IAT Max', 'Bwd Packet Length Max', 'Fwd IAT Mean',
#                      'Fwd Packets/s', 'Min Packet Length', 'Fwd IAT Total',
#                      'Total Backward Packets', 'Flow Packets/s', 'Flow IAT Std'],
#                     dtype='object')

# # threshold: 0.015
# # 24 features
# features = pd.Index(['PSH Flag Count', 'Total Length of Fwd Packets', 'Packet Length Mean',
#                      'Init_Win_bytes_forward', 'Packet Length Variance', 'Packet Length Std',
#                      'Avg Bwd Segment Size', 'Average Packet Size', 'Bwd Packets/s',
#                      'min_seg_size_forward', 'Flow Bytes/s', 'Subflow Fwd Bytes',
#                      'Bwd Packet Length Min', 'Total Length of Bwd Packets',
#                      'Fwd Packet Length Max', 'Destination Port', 'Bwd Packet Length Mean',
#                      'Init_Win_bytes_backward', 'Total Fwd Packets', 'Flow IAT Max',
#                      'ACK Flag Count', 'Fwd Header Length', 'Flow Duration',
#                      'Subflow Fwd Packets'],
#                     dtype='object')


# # Remove 'Fwd Header Length.1'(Repeat)
# features2 = get_features_without_repeat()

# # threshold: 0.005
# # 39 features
# features2 = pd.Index(['PSH Flag Count', 'Packet Length Variance',
#                       'Total Length of Fwd Packets', 'Packet Length Mean',
#                       'Packet Length Std', 'min_seg_size_forward', 'Init_Win_bytes_forward',
#                       'Bwd Packets/s', 'Avg Bwd Segment Size', 'Bwd Packet Length Mean',
#                       'Flow Bytes/s', 'Fwd Packet Length Max', 'Bwd Packet Length Min',
#                       'Destination Port', 'Total Length of Bwd Packets', 'Subflow Fwd Bytes',
#                       'Average Packet Size', 'ACK Flag Count', 'Subflow Bwd Bytes',
#                       'Total Fwd Packets', 'Init_Win_bytes_backward', 'Flow IAT Max',
#                       'Bwd Header Length', 'Avg Fwd Segment Size', 'Flow Duration',
#                       'Fwd IAT Mean', 'Max Packet Length', 'Subflow Fwd Packets',
#                       'Bwd Packet Length Max', 'Fwd Packet Length Mean', 'Fwd IAT Min',
#                       'Fwd IAT Max', 'Subflow Bwd Packets', 'Fwd IAT Total', 'Flow IAT Std',
#                       'Min Packet Length', 'Fwd Packets/s', 'act_data_pkt_fwd',
#                       'Fwd Header Length'],
#                      dtype='object')

# # threshold: 0.01
# # 33 features
# features2 = pd.Index(['PSH Flag Count', 'Packet Length Variance',
#                       'Total Length of Fwd Packets', 'Packet Length Mean',
#                       'Packet Length Std', 'min_seg_size_forward', 'Init_Win_bytes_forward',
#                       'Bwd Packets/s', 'Avg Bwd Segment Size', 'Bwd Packet Length Mean',
#                       'Flow Bytes/s', 'Fwd Packet Length Max', 'Bwd Packet Length Min',
#                       'Destination Port', 'Total Length of Bwd Packets', 'Subflow Fwd Bytes',
#                       'Average Packet Size', 'ACK Flag Count', 'Subflow Bwd Bytes',
#                       'Total Fwd Packets', 'Init_Win_bytes_backward', 'Flow IAT Max',
#                       'Bwd Header Length', 'Avg Fwd Segment Size', 'Flow Duration',
#                       'Fwd IAT Mean', 'Max Packet Length', 'Subflow Fwd Packets',
#                       'Bwd Packet Length Max', 'Fwd Packet Length Mean', 'Fwd IAT Min',
#                       'Fwd IAT Max', 'Subflow Bwd Packets'],
#                      dtype='object')

# # threshold: 0.02
# # 20 features
# features2 = pd.Index(['PSH Flag Count', 'Packet Length Variance',
#                       'Total Length of Fwd Packets', 'Packet Length Mean',
#                       'Packet Length Std', 'min_seg_size_forward', 'Init_Win_bytes_forward',
#                       'Bwd Packets/s', 'Avg Bwd Segment Size', 'Bwd Packet Length Mean',
#                       'Flow Bytes/s', 'Fwd Packet Length Max', 'Bwd Packet Length Min',
#                       'Destination Port', 'Total Length of Bwd Packets', 'Subflow Fwd Bytes',
#                       'Average Packet Size', 'ACK Flag Count', 'Subflow Bwd Bytes',
#                       'Total Fwd Packets'],
#                      dtype='object')
