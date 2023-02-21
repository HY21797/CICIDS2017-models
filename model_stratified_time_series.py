import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from functions import print_versions, sort_files, process_data, generate_new_df, multiclass_cross_validation, plot_multiclass_results, binary_cross_validation, plot_cv, plot_all_feature_importances

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

# Define the features used by the classifier
features = df.columns[:-1]
# Stratified the data
new_df = generate_new_df(df, file_number)

X = new_df[features]
y = new_df['Label']
labels = ['PortScan', 'Patator', 'Brute Force']
multi_model = RandomForestClassifier(n_jobs=-2)

# tscv = TimeSeriesSplit(n_splits=file_number-1,
#                        test_size=round(len(new_df)/file_number))

# print("Visualizing cross-validation behavior for the multiclass model")
# plot_cv(tscv, X, y, file_number-1,
#         'Stratified Time Series Split - Multiclass Classifier')

# print("Stratified time series cross validation for the multiclass model")
# results = multiclass_cross_validation(
#     multi_model, tscv.split(X), X, y, labels)
# # [r1, p1, r2, p2, count_1, accuracy_scores, f1_scores]
# suptitle = 'Stratified Time Series Cross Validation'
# plot_multiclass_results(results, labels, suptitle)

print("Training the multiclass classifier")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=round(len(new_df)/file_number), shuffle=False)
multi_model.fit(X_train, y_train)

print("Plotting the feature importances for the multiclass classifier")
plot_all_feature_importances(
    X_test, y_test, multi_model, features, 'Multiclass Classifier')


new_df['GT'] = np.where(new_df['Label'] == 'BENIGN', 'Benign', 'Malicious')
y_binary = new_df['GT']
binary_model = RandomForestClassifier(n_jobs=-2)

# tscv2 = TimeSeriesSplit(n_splits=file_number-1,
#                         test_size=round(len(new_df)/file_number))

# print("Visualizing cross-validation behavior for the binary model")
# plot_cv(tscv2, X, y_binary, file_number-1,
#         'Stratified Time Series Split - Multiclass Classifier')

# print("Stratified time series cross validation for the binary model")
# suptitle2 = "Stratified Time Series Cross Validation"
# binary_cross_validation(binary_model, tscv2.split(X), X, y_binary, suptitle2)

print("Training the binary classifier")
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_binary, test_size=round(len(new_df)/file_number), shuffle=False)
binary_model.fit(X_train2, y_train2)

print("Plotting the feature importances for the binary classifier")
plot_all_feature_importances(
    X_test2, y_test2, binary_model, features, 'Binary Classifier')
