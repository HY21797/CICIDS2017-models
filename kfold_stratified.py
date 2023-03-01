import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from functions import print_versions, process_data, plot_cv
from functions import get_features_without_repeat, reading_files

print_versions()

# Reading CSV files, and merging all of them into a single DataFrame
df, file_number = reading_files()
df = process_data(df)

# Multiclass classifier
features = get_features_without_repeat()

X = df[features]
y = df['Label']
skf = StratifiedKFold(n_splits=file_number-1, random_state=1, shuffle=True)

print("Visualizing cross-validation behavior for the multiclass model")
plot_cv(skf, X, y, file_number-1,
        'Stratified K-Fold Split - Multiclass Classifier')


# Binary classifier
features2 = get_features_without_repeat()

df['GT'] = np.where(df['Label'] == 'BENIGN', 'Benign', 'Malicious')
X_binary = df[features2]
y_binary = df['GT']

skf2 = StratifiedKFold(n_splits=file_number-1, random_state=1, shuffle=True)

print("Visualizing cross-validation behavior for the binary model")
plot_cv(skf2, X_binary, y_binary, file_number-1,
        'Stratified K-Fold Split - Binary Classifier')
