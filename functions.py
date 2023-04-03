import pandas as pd
import numpy as np
import seaborn as sns
import os
import math
import sklearn
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score


def print_versions():
    print("scikit-learn version: {}".format(sklearn.__version__))
    print("Pandas version: {}".format(pd.__version__))
    print("NumPy version: {}".format(np.__version__))


# Append files in time order
def sort_files(f):
    if (f.split("-")[0] == "Monday"):
        return 0
    if (f.split("-")[0] == "Tuesday"):
        return 1
    if (f.split("-")[0] == "Wednesday"):
        return 2
    if (f.split("-")[0] == "Thursday" and f.split("-")[2] == "Morning"):
        return 3
    if (f.split("-")[0] == "Thursday" and f.split("-")[2] == "Afternoon"):
        return 4
    if (f.split("-")[0] == "Friday" and f.split(".")[0].split("-")[2] == "Morning"):
        return 5
    if (f.split("-")[0] == "Friday" and f.split(".")[0].split("-")[3] == "PortScan"):
        return 6
    return 7


def reading_files():
    # Reading CSV files, and merging all of them into a single
    # Downloading the dataset from https://www.unb.ca/cic/datasets/ids-2017.html
    file_number = 0
    root_folder = os.path.dirname(
        os.path.abspath(__file__)) + "/MachineLearningCVE/"
    df = pd.DataFrame()
    dfs = [None] * 8
    for f in os.listdir(root_folder):
        file_number = file_number + 1
        dfs[sort_files(f)] = f
    for x in range(file_number):
        print("Reading: ", dfs[x])
        df = pd.concat([df, pd.read_csv(root_folder + dfs[x])])
    return df, file_number


def process_data(df):
    # print(len(df))  # 2830743

    # Label all Patator attacks as 'Patator', all Brute Force attacks as 'Brute Force'
    # df = df[df['Label'].str.contains("BENIGN|PortScan|Patator|Brute")==True]
    df.columns = df.columns.str.lstrip()
    df['Label'] = df['Label'].replace(
        ['FTP-Patator', 'SSH-Patator'], 'Patator')
    df.loc[df['Label'].str.contains('Brute'), 'Label'] = 'Brute Force'
    df = df[df['Label'].isin(['BENIGN', 'PortScan', 'Patator', 'Brute Force'])]

    # print(len(df))  # 2447369
    # print(len(df[df['Label'] == 'BENIGN']))  # 2273097
    # print(len(df[df['Label'] == 'PortScan']))  # 158930
    # print(len(df[df['Label'] == 'Patator']))  # 13835
    # print(len(df[df['Label'] == 'Brute Force']))  # 1507

    # QUICK PREPROCESSING.
    # Some classifiers do not like "infinite" (inf) or "null" (NaN) values.
    # https://github.com/hihey54/dummy-ML_NIDS/blob/main/code/CICIDS17_example.ipynb
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # print(len(df))  # 2445463
    # print(len(df[df['Label'] == 'BENIGN']))  # 2271320
    # print(len(df[df['Label'] == 'PortScan']))  # 158804
    # print(len(df[df['Label'] == 'Patator']))  # 13832
    # print(len(df[df['Label'] == 'Brute Force']))  # 1507
    return df


# Stratifing data: split the data into 'file_number' folds,
#   each file contains same amount of different data with the same label
def generate_new_df(df, file_number):
    print("Stratifing data")
    b1 = df[df['Label'] == 'BENIGN']
    range1 = math.trunc(len(b1)/file_number)
    b2 = df[df['Label'] == 'PortScan']
    range2 = math.trunc(len(b2)/file_number)
    b3 = df[df['Label'] == 'Patator']
    range3 = math.trunc(len(b3)/file_number)
    b4 = df[df['Label'] == 'Brute Force']
    range4 = math.trunc(len(b4)/file_number)
    new_df = pd.DataFrame()
    for i in range(file_number):
        new_df = pd.concat([new_df, b1[i * range1: i * range1 + range1]])
        new_df = pd.concat([new_df, b2[i * range2: i * range2 + range2]])
        new_df = pd.concat([new_df, b3[i * range3: i * range3 + range3]])
        new_df = pd.concat([new_df, b4[i * range4: i * range4 + range4]])
    print("Finish stratifing data")
    return new_df


def get_features_without_repeat():
    # Remove 'Fwd Header Length.1'(Repeat)
    features = pd.Index(['Destination Port', 'Flow Duration', 'Total Fwd Packets',
                         'Total Backward Packets', 'Total Length of Fwd Packets',
                         'Total Length of Bwd Packets', 'Fwd Packet Length Max',
                         'Fwd Packet Length Min', 'Fwd Packet Length Mean',
                         'Fwd Packet Length Std', 'Bwd Packet Length Max',
                         'Bwd Packet Length Min', 'Bwd Packet Length Mean',
                         'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
                         'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
                         'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
                         'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
                         'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
                         'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
                         'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
                         'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
                         'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
                         'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                         'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
                         'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
                         'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
                         'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
                         'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
                         'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                         'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
                         'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
                         'Idle Std', 'Idle Max', 'Idle Min'],
                        dtype='object')
    return features


def evaluation(labels, y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt='d')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.suptitle(title)
    plt.title('Accuracy = {:3f}     Precision = {:3f}     Recall = {:3f}     F1-score = {:3f}'.format(
        accuracy_score(y_test, y_pred), precision_score(
            y_test, y_pred, average='macro'),
        recall_score(y_test, y_pred, average='macro'),
        fbeta_score(y_test, y_pred, average='macro', beta=1.0)))
    plt.show()


# def pr_curve(model, X_test, y_test):
#     pred_proba = model.predict_proba(X_test)
#     precision, recall, thresholds = precision_recall_curve(
#         y_test, pred_proba[:, 1], pos_label='Malicious')
#     fig, ax = plt.subplots()
#     plt.plot(recall, precision, lw=1, alpha=0.3,
#              label='AUC = %0.2f' % (average_precision_score(y_test, pred_proba[:, 1], pos_label='Malicious')))
#     ax.set_title('Binary Classifier')
#     ax.set_ylabel('Precision')
#     ax.set_xlabel('Recall')
#     plt.show()


# Plot feature importances to select features
def plot_feature_importance(type, model, X_test, y_test, label, title, features):
    # feature importances
    if (type == 'f'):
        importances = model.feature_importances_
        sorted_idx = importances.argsort()
        print(features[np.argsort(importances)[::-1]])
        bars = plt.barh(range(len(sorted_idx)), importances[sorted_idx],
                        align='edge', color='lightskyblue')
    # permutation importances
    if (type == 'p'):
        perm_importances = permutation_importance(model, X_test, y_test)
        sorted_idx = perm_importances.importances_mean.argsort()
        print(features[np.argsort(perm_importances)[::-1]])
        bars = plt.barh(range(len(sorted_idx)), perm_importances.importances_mean[sorted_idx],
                        align='edge', color='lightskyblue')
    plt.bar_label(bars, padding=10, color='blue', fontsize=7)
    plt.yticks(range(len(sorted_idx)), np.array(
        X_test.columns)[sorted_idx], fontsize=7)
    plt.xlabel(label)
    plt.title(title)
    plt.show()


# Generate all feature importances
def plot_all_feature_importances(X_test, y_test, model, features, title):
    # feature importances
    plot_feature_importance('f', model, X_test, y_test,
                            'Feature Importance', title, features)

    # # permutation importances
    # plot_feature_importance('p', model, X_test, y_test,
    #                         'Permutation Importance', title, features)

    # shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # shap.force_plot(
    #     explainer.expected_value[0], shap_values[0], X_test.iloc[0:])
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=len(
        features), class_names=model.classes_)


# Visualizing cross-validation
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
def plot_cv_indices(cv, X, y, ax, n_splits, title):
    lw = 10
    length = 0
    cy = {0: 'blue', 1: 'salmon', 2: 'red'}
    # Generate the training/validating visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X, y)):
        # Fill in indices with the training/validating groups
        length = len(tr)+len(tt)
        indices = np.array([np.nan] * length)
        indices[tr] = 0
        indices[tt] = 1
        if ii == n_splits-1:
            indices[tt] = 2
        cmap = np.vectorize(cy.get)(indices)
        # Visualize the results
        ax.scatter(range(len(indices)), [ii + 0.5] * len(indices),
                   c=cmap, marker="_", lw=lw)

    # Plot the data classes at the end
    l = sorted(y.unique())
    colors = ['lightskyblue', 'orange', 'purple', 'lime']
    cy = y.map({l[i]: colors[i] for i in range(len(l))})
    ax.scatter(range(length), [ii + 1.5] * length,
               c=cy, marker="_", lw=lw)

    # Formatting
    yticklabels = list(i + 1 for i in range(n_splits)) + ["class"]
    xticklabels = list(math.trunc((i+1)*(length/(n_splits+1)))
                       for i in range(n_splits+1))
    ax.xaxis.set_major_locator(ticker.FixedLocator(xticklabels))
    ax.set_xticklabels(xticklabels, fontsize=7)
    ax.set(yticks=np.arange(n_splits + 1) + 0.5, yticklabels=yticklabels, xlabel="Data index",
           ylabel="CV iteration", ylim=[n_splits + 1.2, -0.2], xlim=(0, length))
    ax.set_title(title, fontsize=10)
    cs = [Patch(color='blue'), Patch(color='salmon'), Patch(color='red')]
    ls = ["Training set", "Validation set", "Testing set"]
    for i in range(len(l)):
        cs.append(Patch(color=colors[i]))
        ls.append(l[i])
    return ax, cs, ls


# Visualizing cross-validation - plot class label
def plot_cv(cv, X, y, n_splits, title):
    fig, ax = plt.subplots()
    ax, cs, ls = plot_cv_indices(cv, X, y, ax, n_splits, title)
    ax.legend(cs, ls, loc=(1.02, 0.7), fontsize=8)
    plt.tight_layout()
    plt.show()


def cross_validation(model, tscv, X, y, suptitle):
    count = 0
    counts = []
    accuracy_scores = []
    f1_scores = []
    precisions = []
    recalls = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        count = count + 1
        counts.append(count)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(fbeta_score(
            y_test, y_pred, average='macro', beta=1.0))
        precisions.append(precision_score(y_test, y_pred, average='macro'))
        recalls.append(recall_score(y_test, y_pred, average='macro'))
    plt.subplot(2, 1, 1)
    r1, p1 = zip(*sorted(zip(recalls, precisions)))
    plt.plot(r1, p1)
    plt.xlabel('Recall (AVG = {:3f})'.format(
        sum(recalls)/len(recalls)))
    plt.ylabel('Precision (AVG = {:3f})'.format(
        sum(precisions)/len(precisions)))
    plt.title(suptitle)
    plt.subplot(2, 1, 2)
    plt.plot(counts, accuracy_scores, label='Accuracy (AVG = {:3f})'.format(
        sum(accuracy_scores)/len(accuracy_scores)))
    plt.plot(counts, f1_scores, label='F1 (AVG = {:3f})'.format(
        sum(f1_scores)/len(f1_scores)))
    plt.xlabel("Test Fold")
    plt.legend()
    plt.title(suptitle)
    plt.show()


# Stratified time series cross validation for hyperparameter tuning
def hp_cross_validation(model, tscv, X, y):
    f1_scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1_scores.append(fbeta_score(
            y_test, y_pred, average='macro', beta=1.0))
    return sum(f1_scores)/len(f1_scores)


# Hyperparameter tuning for criterion
def hp_criterion(tscv, X, y, n):
    for i in ['gini', 'entropy']:
        model = RandomForestClassifier(
            n_estimators=n, criterion=i, n_jobs=-1, random_state=1)
        score = hp_cross_validation(model, tscv, X, y)
        print('%s: %.5f' % (i, score))


# Plot the hyperparameter tuning graph
def plot_hp(scores, param, suptitle):
    scores = np.array(scores)
    max_score = np.where(scores == np.max(scores[:, 1]))[0][0]
    plt.plot(scores[:, 0], scores[:, 1])
    plt.title('%s=%s      max f1-score=%6f' %
              (param, scores[max_score][0], scores[max_score][1]))
    plt.suptitle(suptitle)
    plt.show()


# Hyperparameter tuning for the multiclass model
def multiclass_hp_tuning(tscv, X, y):
    # n_estimators: (10, 201, 10) -> 80
    # n_estimators: (51, 70) -> 76
    # # criterion: ['gini': 0.98356, 'entropy': 0.98308] -> gini
    # hp_criterion(tscv, X, y, 76)
    # max_depth: (10, 101, 10) -> 50
    # max_depth: (41, 51) -> 44
    # min_samples_split (1, 21) -> 11
    f1_scores = []
    for i in range(1, 21):
        model = RandomForestClassifier(n_estimators=76, criterion='gini', max_depth=44,
                                       min_samples_split=i, n_jobs=-1, random_state=1)
        f1_score = hp_cross_validation(model, tscv, X, y)
        f1_scores.append([i, f1_score])
        print('Finish ' + str(i) + " " + str(f1_score))
    plot_hp(f1_scores, 'min_samples_split', 'Multiclass Classifier')


# Hyperparameter tuning for the binary model
def binary_hp_tuning(tscv, X, y):
    # n_estimators: (10, 201, 10) -> 160
    # n_estimators: (151, 170) -> 159
    # # criterion: ['gini': 0.99553, 'entropy': 0.99557] -> entropy
    # hp_criterion(tscv, X, y, 159)
    # max_depth: (50, 101, 10) -> 70
    # max_depth: (61, 71) -> 67
    # min_samples_split (11, 31) -> 18
    f1_scores = []
    for i in range(11, 31):
        model = RandomForestClassifier(n_estimators=159, criterion='entropy', max_depth=67,
                                       min_samples_split=i, n_jobs=-1, random_state=1)
        f1_score = hp_cross_validation(model, tscv, X, y)
        f1_scores.append([i, f1_score])
        print('Finish ' + str(i) + " " + str(f1_score))
    plot_hp(f1_scores, 'min_samples_split', 'Binary Classifier')


# Append files in time order
def sort_files_2018(f):
    if (f.split("-")[1] == "14"):
        return 0
    if (f.split("-")[1] == "15"):
        return 1
    if (f.split("-")[1] == "16"):
        return 2
    if (f.split("-")[1] == "20"):
        return 3
    if (f.split("-")[1] == "21"):
        return 4
    if (f.split("-")[1] == "22"):
        return 5
    if (f.split("-")[1] == "23"):
        return 6
    # Remove Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv
    if (f.split("-")[1] == "28"):
        return 8
    # Remove Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
    if (f.split("-")[1] == "01"):
        return 9
    return 7


def reading_processing_2018():
    # Reading CSV files, and merging all of them into a single DataFrame
    # Downloading the dataset from https://www.unb.ca/cic/datasets/ids-2018.html
    # and https://registry.opendata.aws/cse-cic-ids2018/
    df = pd.DataFrame()
    dfs = [None] * 10
    root_folder = os.path.dirname(
        os.path.abspath(__file__)) + "/CSECICIDS2018/"
    for f in os.listdir(root_folder):
        dfs[sort_files_2018(f)] = f
    for x in range(len(dfs)-2):
        print("Reading: ", dfs[x])
        df2 = change_feature_name(pd.read_csv(
            root_folder + dfs[x], low_memory=False))
        # Drop "infinite" (inf) or "null" (NaN) values.
        df2 = process_data_2018(df2)
        df = pd.concat([df, df2], ignore_index=True)
    return df


# Change the features name to fit the 2017 dataset.
def change_feature_name(df2):
    df2['Label'] = df2['Label'].replace('Benign', 'BENIGN')
    df2.loc[df2['Label'].str.contains('-Brute'), 'Label'] = 'Patator'
    df2.loc[df2['Label'].str.contains('Brute Force'), 'Label'] = 'Brute Force'
    df2 = df2[df2['Label'].isin(['BENIGN', 'Patator', 'Brute Force'])]
    df2 = df2.drop(['Timestamp'], axis=1)
    df2.columns = df2.columns.str.replace("Len", "Length")
    df2.columns = df2.columns.str.replace("Dst", "Destination")
    df2.columns = df2.columns.str.replace("Tot", "Total")
    df2.columns = df2.columns.str.replace("Pkts", "Packets")
    df2.columns = df2.columns.str.replace("Pkt", "Packet")
    df2.columns = df2.columns.str.replace("Cnt", "Count")
    df2.columns = df2.columns.str.replace("Var", "Variance")
    df2.columns = df2.columns.str.replace("Byts", "Bytes")
    df2 = df2.rename({'Total Bwd Packets': 'Total Backward Packets',
                      'TotalLength Fwd Packets': 'Total Length of Fwd Packets',
                      'TotalLength Bwd Packets': 'Total Length of Bwd Packets',
                      'Packet Length Min': 'Min Packet Length',
                      'Packet Length Max': 'Max Packet Length',
                      'Packet Size Avg': 'Average Packet Size',
                      'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
                      'Bwd Seg Size Avg': 'Avg Bwd Segment Size',
                      'Fwd Bytes/b Avg': 'Fwd Avg Bytes/Bulk',
                      'Fwd Packets/b Avg': 'Fwd Avg Packets/Bulk',
                      'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate',
                      'Bwd Bytes/b Avg': 'Bwd Avg Bytes/Bulk',
                      'Bwd Packets/b Avg': 'Bwd Avg Packets/Bulk',
                      'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
                      'Init Fwd Win Bytes': 'Init_Win_bytes_forward',
                      'Init Bwd Win Bytes': 'Init_Win_bytes_backward',
                      'Fwd Act Data Packets': 'act_data_pkt_fwd',
                      'Fwd Seg Size Min': 'min_seg_size_forward'
                      }, axis=1)
    return df2


# Processing the CSECICIDS2018 dataset
def process_data_2018(df2):
    df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    df2.dropna(inplace=True)
    df2.drop_duplicates(inplace=True)
    return df2


def evaluation_2018(cm, labels, title):
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt='d')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.title(title)
    plt.show()
