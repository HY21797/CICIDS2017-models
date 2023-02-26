import pandas as pd
import numpy as np
import math
import sklearn
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, fbeta_score
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


def process_data(df):
    # QUICK PREPROCESSING.
    # Some classifiers do not like "infinite" (inf) or "null" (NaN) values.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.columns = df.columns.str.lstrip()

    # Label all Patator attacks as 'Patator', all Brute Force attacks as 'Brute Force'
    # df = df[df['Label'].str.contains("BENIGN|PortScan|Patator|Brute")==True]
    df['Label'] = df['Label'].replace(
        ['FTP-Patator', 'SSH-Patator'], 'Patator')
    df.loc[df['Label'].str.contains('Brute'), 'Label'] = 'Brute Force'
    df = df[df['Label'].isin(['BENIGN', 'PortScan', 'Patator', 'Brute Force'])]
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


# Calculate the precision and recall - micro average
def getMultiPR(pred, target_data):
    tp = 0  # true positives
    fp = 0  # false positive
    fn = 0  # false negatives
    for i in range(len(pred)):
        if pred[i] != 'BENIGN' and pred[i] == target_data.iloc[i]:  # True positive
            tp += 1
        if pred[i] != target_data.iloc[i] and target_data.iloc[i] == 'BENIGN':  # False positive
            fp += 1
        if pred[i] == 'BENIGN' and target_data.iloc[i] != pred[i]:  # False negative
            fn += 1
    precision = 1
    recall = 1
    if (tp + fp != 0):
        precision = tp / (tp + fp)
    if (tp + fn != 0):
        recall = tp / (tp + fn)
    return [precision, recall]


# Calculate the precision and recall - macro average
def getBinaryPR(pred, target_data, malicious):
    tp = 0  # true positives
    fp = 0  # false positive
    fn = 0  # false negatives
    for i in range(len(pred)):
        if pred[i] == malicious and target_data.iloc[i] == malicious:  # True positive
            tp += 1
        if pred[i] == malicious and target_data.iloc[i] != malicious:  # False positive
            fp += 1
        if pred[i] != malicious and target_data.iloc[i] == malicious:  # False negative
            fn += 1
    precision = 1
    recall = 1
    if (tp + fp != 0):
        precision = tp / (tp + fp)
    if (tp + fn != 0):
        recall = tp / (tp + fn)
    return [precision, recall]


def multiclass_cross_validation(multi_model, split, X, y, labels):
    accuracy_scores = []
    f1_scores = []
    p1 = []
    r1 = []
    p2 = [[] for i in range(len(labels))]
    r2 = [[] for i in range(len(labels))]
    counts = []
    count = 0
    # for i, (train_index, test_index) in enumerate(tscv.split(X)):
    # print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    for train_index, test_index in split:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        multi_model.fit(X_train, y_train)
        y_pred = multi_model.predict(X_test)
        p = 0
        r = 0
        for i in range(len(labels)):
            pr = getBinaryPR(y_pred, y_test, labels[i])
            p2[i].append(pr[0])
            r2[i].append(pr[1])
            p = p + pr[0]
            r = r + pr[1]
        p1.append(p/3)
        r1.append(r/3)
        count = count + 1
        counts.append(count)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(fbeta_score(
            y_test, y_pred, average='macro', beta=1.0))
    return [r1, p1, r2, p2, counts, accuracy_scores, f1_scores]


def plot_multiclass_results(results, labels, suptitle):
    # results = [r1, p1, r2, p2, counts, accuracy_scores, f1_scores]

    # Plot the average precision-recall curve
    r1, p1 = zip(*sorted(zip(results[0], results[1])))
    plt.plot(r1, p1, label="Overall")
    for i in range(len(labels)):
        results[2][i], results[3][i] = zip(
            *sorted(zip(results[2][i], results[3][i])))
        plt.plot(results[2][i], results[3][i], label=labels[i])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title(suptitle + ' - Multiclass Classifier')
    plt.show()

    # Plot the accuracy and f1 graph
    plt.plot(results[4], results[5], label='Accuracy (AVG = {:3f})'.format(
        sum(results[5])/len(results[5])))
    plt.plot(results[4], results[6], label='F1 (AVG = {:3f})'.format(
        sum(results[6])/len(results[6])))
    plt.xlabel("Test Fold")
    plt.legend()
    plt.title(suptitle + ' - Multiclass Classifier')
    plt.show()


# For time series cross validation only
def binary_time_cross_validation(binary_model, split, X_binary, y_binary):
    accuracy_scores = []
    f1_scores = []
    p = []
    r = []
    counts = []
    count = 0
    for train_index, test_index in split:
        X_train, X_test = X_binary.iloc[train_index], X_binary.iloc[test_index]
        y_train, y_test = y_binary.iloc[train_index], y_binary.iloc[test_index]
        binary_model.fit(X_train, y_train)
        y_pred = binary_model.predict(X_test)
        pr = getBinaryPR(y_pred, y_test, 'Malicious')
        p.append(pr[0])
        r.append(pr[1])
        count = count + 1
        counts.append(count)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(fbeta_score(
            y_test, y_pred, average='macro', beta=1.0))
    return [r, p, counts, accuracy_scores, f1_scores]


def plot_binary_results(results):
    # results = [r, p, counts, accuracy_scores, f1_scores]

    # Plot the average precision-recall curve
    r, p = zip(*sorted(zip(results[0], results[1])))
    plt.plot(r, p, label="Overall")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Time Series Cross Validation - Binary Classifier')
    plt.show()

    # Plot the accuracy and f1 graph
    plt.plot(results[2], results[3], label='Accuracy (AVG = {:3f})'.format(
        sum(results[3])/len(results[3])))
    plt.plot(results[2], results[4], label='F1 (AVG = {:3f})'.format(
        sum(results[4])/len(results[4])))
    plt.xlabel("Test Fold")
    plt.legend()
    plt.title('Time Series Cross Validation - Binary Classifier')
    plt.show()


def binary_cross_validation(binary_model, split, X_binary, y_binary, suptitle):
    accuracy_scores = []
    f1_scores = []
    y_real = []
    y_proba = []
    counts = []
    count = 0
    # Stratified cross validation
    for train_index, test_index in split:
        X_train, X_test = X_binary.iloc[train_index], X_binary.iloc[test_index]
        y_train, y_test = y_binary.iloc[train_index], y_binary.iloc[test_index]
        binary_model.fit(X_train, y_train)
        y_pred = binary_model.predict(X_test)
        pred_proba = binary_model.predict_proba(X_test)
        precision, recall, thresholds = precision_recall_curve(
            y_test, pred_proba[:, 1], pos_label='Malicious')
        plt.plot(recall, precision, lw=1, alpha=0.3,
                 label='PR fold %d (AUC = %0.2f)' % (count, average_precision_score(y_test, pred_proba[:, 1], pos_label='Malicious')))
        y_real.append(y_test)
        y_proba.append(pred_proba[:, 1])
        count = count + 1
        counts.append(count)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(fbeta_score(
            y_test, y_pred, average='macro', beta=1.0))

    # Plot the average precision-recall curve
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(
        y_real, y_proba, pos_label='Malicious')
    plt.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.2f)' % (
                 average_precision_score(y_real, y_proba, pos_label='Malicious')),
             lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title(suptitle + ' - Binary Classifier')
    plt.show()

    # Plot the accuracy and f1 graph
    plt.plot(counts, accuracy_scores, label='Accuracy (AVG = {:3f})'.format(
        sum(accuracy_scores)/len(accuracy_scores)))
    plt.plot(counts, f1_scores, label='F1 (AVG = {:3f})'.format(
        sum(f1_scores)/len(f1_scores)))
    plt.xlabel("Test Fold")
    plt.legend()
    plt.title(suptitle + ' - Binary Classifier')
    plt.show()


# Visualizing cross-validation
def plot_cv_indices(cv, X, y, ax, n_splits, title, cmap_cv):
    lw = 20
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X, y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + 0.5] * len(indices),
                   c=indices, marker="_", lw=lw, cmap=cmap_cv, vmin=-0.2, vmax=1.2)

    # Plot the data classes at the end
    l = sorted(y.unique())
    colors = ['lightskyblue', 'orange', 'purple', 'lime']
    cy = y.map({l[i]: colors[i] for i in range(len(l))})
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=cy, marker="_", lw=lw)

    # Formatting
    yticklabels = list(i + 1 for i in range(n_splits)) + ["class"]
    xticklabels = list(math.trunc((i+1)*(len(X)/(n_splits+1)))
                       for i in range(n_splits+1))
    ax.xaxis.set_major_locator(ticker.FixedLocator(xticklabels))
    ax.set_xticklabels(xticklabels, fontsize=7)
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Data index",
        ylabel="CV iteration",
        ylim=[n_splits + 1.2, -0.2],
        xlim=(0, len(X))
    )
    ax.set_title(title, fontsize=10)
    cs = [Patch(color=cmap_cv(0.02)), Patch(color=cmap_cv(0.8))]
    ls = ["Training set", "Testing set"]
    for i in range(len(l)):
        cs.append(Patch(color=colors[i]))
        ls.append(l[i])
    return ax, cs, ls


# Visualizing cross-validation - plot class label
def plot_cv(cv, X, y, n_splits, title):
    fig, ax = plt.subplots()
    cmap_cv = plt.cm.coolwarm
    ax, cs, ls = plot_cv_indices(cv, X, y, ax, n_splits, title, cmap_cv)
    ax.legend(cs, ls, loc=(1.02, 0.7), fontsize=8)
    plt.tight_layout()
    plt.show()


# Plot feature importances to select features
def plot_feature_importance(type, model, X_test, y_test, label, title, features):
    # feature importances
    if (type == 'f'):
        importances = model.feature_importances_
        sorted_idx = importances.argsort()
        print(features[sorted_idx])
        bars = plt.barh(range(len(sorted_idx)), importances[sorted_idx],
                        align='edge', color='lightskyblue')
    # permutation importances
    if (type == 'p'):
        perm_importances = permutation_importance(model, X_test, y_test)
        sorted_idx = perm_importances.importances_mean.argsort()
        print(features[sorted_idx])
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
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=len(
        features), class_names=model.classes_)


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
    plt.title('max score=%6f      %s=%s' %
              (scores[max_score][1], param, scores[max_score][0]))
    plt.suptitle(suptitle)
    plt.show()


# Hyperparameter tuning for the multiclass model
def multiclass_hp_tuning(tscv, X, y):
    # n_estimators: (10, 201, 10) -> 60
    # n_estimators: (51, 70) -> 59
    # criterion: ['gini': 0.9847, 'entropy': 0.9846] -> gini
    # hp_criterion(tscv, X, y, 59)
    # max_depth: (10, 29, 3) -> 22
    # max_depth: (20, 25) -> 22
    # min_samples_split (10, 21) -> 13
    f1_scores = []
    for i in range(10, 21):
        model = RandomForestClassifier(n_estimators=59, criterion='gini', max_depth=22,
                                       min_samples_split=13, n_jobs=-1, random_state=1)
        f1_score = hp_cross_validation(model, tscv, X, y)
        f1_scores.append([i, f1_score])
    plot_hp(f1_scores, 'min_samples_split', 'Multiclass Classifier')
    # min_samples_leaf (1, 11) -> 1
    # max_features (1, 33, 8) -> 1
    # max_features (1, 9) -> 5


# Hyperparameter tuning for the binary model
def binary_hp_tuning(tscv, X, y):
    # n_estimators: (10, 201, 10) -> 160
    # n_estimators: (151, 170) -> 161
    # criterion: ['gini': 0.99560, 'entropy': 0.99563] -> entropy
    # hp_criterion(tscv, X, y, 161)
    # max_depth: (50, 151, 10) -> 90
    # max_depth: (81, 100) -> 81
    # min_samples_split (10, 101, 10) -> 20
    # min_samples_split (11, 30) -> 18
    f1_scores = []
    for i in range(11, 30):
        model = RandomForestClassifier(n_estimators=161, criterion='entropy', max_depth=81,
                                       min_samples_split=18, n_jobs=-1, random_state=1)
        f1_score = hp_cross_validation(model, tscv, X, y)
        f1_scores.append([i, f1_score])
        print('Finish ' + str(i))
    plot_hp(f1_scores, 'min_samples_split', 'Binary Classifier')
    # min_samples_leaf (1, 11) -> 1
    # max_features (1, 28, 5) -> 16
    # max_features (12, 21) -> 12
