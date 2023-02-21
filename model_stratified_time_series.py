import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
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

print("Stratified time series cross validation for the multiclass model")
tscv = TimeSeriesSplit(n_splits=file_number-1,
                       test_size=round(len(new_df)/file_number))
# plot_cv(tscv, X, y, file_number-1,
#         'Stratified Time Series Split - Multiclass Classifier')

labels = ['PortScan', 'Patator', 'Brute Force']
multi_model = RandomForestClassifier(n_jobs=-2)
plot_all_feature_importances(
    tscv, X, y, multi_model, features, 'Multiclass Classifier')

# results = multiclass_cross_validation(
#     multi_model, tscv.split(X), X, y, labels)
# # [r1, p1, r2, p2, count_1, accuracy_scores, f1_scores]
# suptitle = 'Stratified Time Series Cross Validation'
# plot_multiclass_results(results, labels, suptitle)

new_df['GT'] = np.where(new_df['Label'] == 'BENIGN', 'Benign', 'Malicious')
y_binary = new_df['GT']

print("Stratified time series cross validation for the binary model")
tscv2 = TimeSeriesSplit(n_splits=file_number-1,
                        test_size=round(len(new_df)/file_number))
# plot_cv(tscv2, X, y_binary, file_number-1,
#         'Stratified Time Series Split - Multiclass Classifier')

binary_model = RandomForestClassifier(n_jobs=-2)
plot_all_feature_importances(
    tscv2, X, y_binary, binary_model, features, 'Binary Classifier')

# suptitle2 = "Stratified Time Series Cross Validation"
# binary_cross_validation(binary_model, tscv2.split(X), X, y_binary, suptitle2)
