import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from functions import print_versions, process_data, generate_new_df
from functions import reading_files, plot_cv, cross_validation
from functions import multiclass_hp_tuning, binary_hp_tuning

print_versions()

# Reading CSV files, and merging all of them into a single DataFrame
df, file_number = reading_files()
df = process_data(df)

# Stratified the data
new_df = generate_new_df(df, file_number)

# Multiclass classifier
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

# print("Visualizing cross-validation behavior for the multiclass model")
# tscv = TimeSeriesSplit(n_splits=file_number-1,
#                        test_size=round(len(new_df)/file_number))
# plot_cv(tscv, X, y, file_number-1,
#         'Stratified Time Series Split - Multiclass Classifier')

multi_model = RandomForestClassifier(n_jobs=-1, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=round(len(new_df)/file_number), shuffle=False)
tscv = TimeSeriesSplit(n_splits=file_number-2,
                       test_size=round(len(new_df)/file_number))

# Stratified time series cross validation
print("Stratified time series cross validation for the multiclass model")
cross_validation(multi_model, tscv, X_train, y_train, 'Multiclass Classifier')

# Hyperparameter tuning
print('Hyperparameter tuning for the multiclass classifier')
multiclass_hp_tuning(tscv, X_train, X_train)

multi_model = RandomForestClassifier(
    n_estimators=76, criterion='gini', max_depth=44, min_samples_split=11, n_jobs=-1, random_state=1)


# Binary classifier
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

# print("Visualizing cross-validation behavior for the binary model")
# tscv2 = TimeSeriesSplit(n_splits=file_number-1,
#                         test_size=round(len(new_df)/file_number))
# plot_cv(tscv2, X_binary, y_binary, file_number-1,
#         'Stratified Time Series Split - Binary Classifier')

binary_model = RandomForestClassifier(n_jobs=-1, random_state=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_binary, y_binary, test_size=round(len(new_df)/file_number), shuffle=False)
tscv2 = TimeSeriesSplit(n_splits=file_number-2,
                        test_size=round(len(new_df)/file_number))

# Stratified time series cross validation
print("Stratified time series cross validation for the binary model")
cross_validation(binary_model, tscv2, X_train2, y_train2, 'Binary Classifier')

# Hyperparameter tuning
print('Hyperparameter tuning for the binary classifier')
binary_hp_tuning(tscv2, X_train2, y_train2)

binary_model = RandomForestClassifier(n_estimators=159, criterion='entropy', max_depth=67,
                                      min_samples_split=18, n_jobs=-1, random_state=1)


# Multiclass classifier
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

# # threshold: 0.02
# # 18 features
# features = pd.Index(['PSH Flag Count', 'Total Length of Fwd Packets', 'Packet Length Mean',
#                      'Init_Win_bytes_forward', 'Packet Length Variance', 'Packet Length Std',
#                      'Avg Bwd Segment Size', 'Average Packet Size', 'Bwd Packets/s',
#                      'min_seg_size_forward', 'Flow Bytes/s', 'Subflow Fwd Bytes',
#                      'Bwd Packet Length Min', 'Total Length of Bwd Packets',
#                      'Fwd Packet Length Max', 'Destination Port', 'Bwd Packet Length Mean',
#                      'Init_Win_bytes_backward'],
#                     dtype='object')


# Binary classifier
# # threshold: 0.005
# # 39 features
# features2 = pd.Index(['PSH Flag Count', 'Packet Length Variance',
#                       'Total Length of Fwd Packets', 'Packet Length Mean',
#                      'Packet Length Std', 'min_seg_size_forward', 'Init_Win_bytes_forward',
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
#                      'Packet Length Std', 'min_seg_size_forward', 'Init_Win_bytes_forward',
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

# threshold: 0.02
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
