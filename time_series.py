import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from functions import print_versions, process_data, generate_new_df
from functions import get_features_without_repeat, plot_cv, reading_files
from functions import plot_all_feature_importances
from functions import print_versions, reading_files, process_data, evaluation, generate_new_df

print_versions()

# Reading CSV files, and merging all of them into a single DataFrame
df, file_number = reading_files()
df = process_data(df)

# Multiclass classifier
features = get_features_without_repeat()

X = df[features]
y = df['Label']
tscv = TimeSeriesSplit(n_splits=file_number-1,
                       test_size=round(len(df)/file_number))

print("Visualizing cross-validation behavior for the multiclass model")
plot_cv(tscv, X, y, file_number-1,
        'Time Series Split - Multiclass Classifier')


# Binary classifier
features2 = get_features_without_repeat()

df['GT'] = np.where(df['Label'] == 'BENIGN', 'Benign', 'Malicious')
X_binary = df[features2]
y_binary = df['GT']

tscv2 = TimeSeriesSplit(n_splits=file_number-1,
                        test_size=round(len(df)/file_number))

print("Visualizing cross-validation behavior for the binary model")
plot_cv(tscv2, X_binary, y_binary, file_number-1,
        'Time Series Split - Binary Classifier')
