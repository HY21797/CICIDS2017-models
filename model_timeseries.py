import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from functions import print_versions, sort_files
from functions import process_data, multiclass_cross_validation
from functions import plot_multiclass_results, binary_time_cross_validation
from functions import plot_binary_results, plot_cv, get_features_without_repeat

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
features = get_features_without_repeat()

X = df[features]
y = df['Label']
labels = ['PortScan', 'Patator', 'Brute Force']
tscv = TimeSeriesSplit(n_splits=file_number-1,
                       test_size=round(len(df)/file_number))

print("Visualizing cross-validation behavior for the multiclass model")
plot_cv(tscv, X, y, file_number-1, 'Time Series Split - Multiclass Classifier')

print("Time series cross validation for the multiclass model")
multi_model = RandomForestClassifier(n_jobs=-1, random_state=1)
results1 = multiclass_cross_validation(
    multi_model, tscv.split(X), X, y, labels)
# [r1, p1, r2, p2, counts, accuracy_scores, f1_scores]
suptitle = 'Time Series Cross Validation'
plot_multiclass_results(results1, labels, suptitle)


df['GT'] = np.where(df['Label'] == 'BENIGN', 'Benign', 'Malicious')
y_binary = df['GT']
tscv2 = TimeSeriesSplit(n_splits=file_number-1,
                        test_size=round(len(df)/file_number))

print("Visualizing cross-validation behavior for the binary model")
plot_cv(tscv2, X, y_binary, file_number-1,
        'Time Series Split - Binary Classifier')

print("Time series cross validation for the binary model")
binary_model = RandomForestClassifier(n_jobs=-1, random_state=1)
results2 = binary_time_cross_validation(
    binary_model, tscv2.split(X), X, y_binary)
# [r, p, counts, accuracy_scores, f1_scores]
plot_binary_results(results2)
