import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from functions import print_versions, process_data, multiclass_cross_validation, plot_multiclass_results, binary_cross_validation, plot_cv

print_versions()

# Reading CSV files, and merging all of them into a single DataFrame
file_number = 0
root_folder = os.path.dirname(
    os.path.abspath(__file__)) + "/MachineLearningCVE/"
df = pd.DataFrame()
for f in os.listdir(root_folder):
    file_number = file_number + 1
    print("Reading: ", f)
    df = pd.concat([df, pd.read_csv(root_folder + f)])
df = process_data(df)

# Define the features used by the classifier
features = df.columns[:-1]
X = df[features]
y = df['Label']

print("Stratified k-fold cross validation for the multiclass model")
skf = StratifiedKFold(n_splits=file_number-1, random_state=1, shuffle=True)
# plot_cv(skf, X, y, file_number-1,
#         'Stratified K-fold Split - Multiclass Classifier')

labels = ['PortScan', 'Patator', 'Brute Force']
multi_model = RandomForestClassifier(n_jobs=-2)
results = multiclass_cross_validation(
    multi_model, skf.split(X, y), X, y, labels)
# [r1, p1, r2, p2, count_1, accuracy_scores, f1_scores]
suptitle = 'Stratified K-fold Cross Validation'
plot_multiclass_results(results, labels, suptitle)

df['GT'] = np.where(df['Label'] == 'BENIGN', 'Benign', 'Malicious')
y_binary = df['GT']

print("Stratified k-fold cross validation for the binary model")
skf2 = StratifiedKFold(n_splits=file_number-1, random_state=1, shuffle=True)
# plot_cv(skf2, X, y_binary, file_number-1,
#         'Stratified K-fold Split - Binary Classifier')

binary_model = RandomForestClassifier(n_jobs=-2)
suptitle2 = "Stratified K-fold Cross Validation"
binary_cross_validation(binary_model, skf2.split(X, y), X, y_binary, suptitle2)
