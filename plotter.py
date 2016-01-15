import matplotlib.colors as colors
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from bokeh.plotting import output_notebook, figure, show
from bokeh.charts import Bar,BoxPlot, output_file, show
import sklearn
import scipy


def make_cross_validated_roc(X,y,cv,classifier):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 500)
    all_tpr = []
    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y.iloc[test],y_score = probas_[:, 1])
        mean_tpr += scipy.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_tpr = mean_tpr/len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return classifier,plt


def plot_feature_importance(classifier,feature_names_nice):
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    feature_names_nice_ordered = [feature_names_nice[i] for i in indices]

    # Print the feature ranking
    flist = "Feature ranking:\n"

    for f,(imp,name) in enumerate(zip(importances[indices],feature_names_nice_ordered)):
        flist += ("%d. %s (%f)\n"%(f + 1, name, imp))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(importances)), importances[indices], color="r", yerr=std[indices], align="center")
    #plt.xticks(range(len(importances)), indices+1)
    plt.xlim([-1,len(importances)])
    return flist,plt
