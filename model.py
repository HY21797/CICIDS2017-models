import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from functions import print_versions, reading_files, process_data, generate_new_df
from functions import evaluation, generate_new_df
from functions import get_features_without_repeat, plot_all_feature_importances

print_versions()

# Reading CSV files, and merging all of them into a single DataFrame
df, file_number = reading_files()
df = process_data(df)

# Stratified the data
new_df = generate_new_df(df, file_number)

# Define the features used by the classifier
features = get_features_without_repeat()

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

# multi_model = RandomForestClassifier(n_jobs=-1, random_state=1)
multi_model = RandomForestClassifier(
    n_estimators=76, criterion='gini', max_depth=44, min_samples_split=11, n_jobs=-1, random_state=1)

# Train the model
print("Training the multiclass classifier")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=round(len(new_df)/file_number), shuffle=False)
multi_model.fit(X_train, y_train)
y_pred = multi_model.predict(X_test)

# Evaluate the model
print("Evaluating the multiclass classifier")
labels = ['PortScan', 'Patator', 'Brute Force', 'BENIGN']
evaluation(labels, y_test, y_pred, 'Multiclass Classifier')

# # Feature importances
# print("Plotting the feature importances for the multiclass classifier")
# plot_all_feature_importances(
#     X_test, y_test, multi_model, features, 'Multiclass Classifier')


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

# binary_model = RandomForestClassifier(n_jobs=-1, random_state=1)
binary_model = RandomForestClassifier(
    n_estimators=159, criterion='entropy', max_depth=67, min_samples_split=18, n_jobs=-1, random_state=1)

# Train the model
print("Training the binary classifier")
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_binary, y_binary, test_size=round(len(new_df)/file_number), shuffle=False)
binary_model.fit(X_train2, y_train2)
y_pred2 = binary_model.predict(X_test2)

# Evaluate the model
print("Evaluating the multiclass classifier")
labels2 = ['Malicious', 'Benign']
evaluation(labels2, y_test2, y_pred2, 'Binary Classifier')

# # Feature importances
# print("Plotting the feature importances for the binary classifier")
# plot_all_feature_importances(
#     X_test2, y_test2, binary_model, features2, 'Binary Classifier')
