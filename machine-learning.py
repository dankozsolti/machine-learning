# -*- coding: utf-8 -*-
"""
Created on Thu May 20 19:40:52 2021

@author: Zsolt Dankó
"""

import numpy as np;  # Numerical Python library
import matplotlib.pyplot as plt;  # Matlab-like Python module
import pandas as pd; 
from sklearn.naive_bayes import GaussianNB; #  importing naive Bayes classifier
from sklearn.cluster import KMeans; # Class for K-means clustering
from sklearn.metrics import davies_bouldin_score;  # function for Davies-Bouldin goodness-of-fit 
from sklearn.decomposition import PCA; #  Class for Principal Component analysis
from sklearn.model_selection import train_test_split;
from sklearn.feature_selection import SelectKBest;
import matplotlib.colors as col;
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc; #  importing performance metrics


#Loading the dataset
data = pd.read_csv('https://raw.githubusercontent.com/dankozsolti/machine-learning/main/data/pulsar_data.csv');

# filling NaN values with mean of their columns
for column in data.columns:
    data[column] = data[column].fillna(data[column].mean());

#Defining X, Y values and names
X = data.drop(['target_class'], axis = 1)
X2 = X.to_numpy();
y = data['target_class'];
X_names = X.columns;
target_names = ["NORMAL", "PULSAR"]

#Fitting PCA
pca = PCA();
pca.fit(data);

#Plotting the variance of principal components
fig = plt.figure();
plt.title('Explained variance ratio plot');
var_ratio = pca.explained_variance_ratio_;
x_pos = np.arange(len(var_ratio));
plt.xticks(x_pos,x_pos+1);
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos,var_ratio, align='center');
plt.show(); 

#Scatterplot using the first two principal components
colors = ['blue', 'red'];
plt.figure();
p = X.shape[1];
feature_selection = SelectKBest(k=2);
feature_selection.fit(X,y);
scores = feature_selection.scores_;
features1 = feature_selection.transform(X);
mask = feature_selection.get_support();
feature_indices = [];
for i in range(p):
    if mask[i] == True : feature_indices.append(i);
x_axis, y_axis = feature_indices;
plt.title('Scatterplot for 2 Kbest attributes');
plt.xlabel(X_names[x_axis]);
plt.ylabel(X_names[y_axis]);
plt.scatter(X2[:,0],X2[:,1], s=5, c=y, alpha=0.5, cmap=col.ListedColormap(colors));
plt.show();

#Split the data into train and test data, 70-30%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle = True)

# Fitting logistic regression
logreg_classifier = LogisticRegression(solver='liblinear');
logreg_classifier.fit(X_train,y_train);
ypred_logreg = logreg_classifier.predict(X_train);   # spam prediction for train
accuracy_logreg_train = logreg_classifier.score(X_train,y_train);
cm_logreg_train = confusion_matrix(y_train, ypred_logreg); # train confusion matrix
ypred_logreg = logreg_classifier.predict(X_test);   # spam prediction for test
cm_logreg_test = confusion_matrix(y_test, ypred_logreg); # test confusion matrix
yprobab_logreg = logreg_classifier.predict_proba(X_test);  #  prediction probabilities
accuracy_logreg_test = logreg_classifier.score(X_test,y_test);

accuracy = logreg_classifier.score(X_test, y_test);
print("Test Accuracy of Logistic Regression: {:.2f}%".format(accuracy * 100));

# Plotting confusion matrix of Logistic regression
# On Train data
plot_confusion_matrix(logreg_classifier, X_train, y_train, display_labels = target_names);
plt.title('Confusion matrix for Train data (Logistic regression)')
plt.xlabel('Predicted');
plt.ylabel('Actual');

# On Test data
plot_confusion_matrix(logreg_classifier, X_test, y_test, display_labels = target_names);
plt.title('Confusion matrix for Test data (Logistic regression)')
plt.xlabel('Predicted');
plt.ylabel('Actual');

#Fitting naive Bayes classifier
naive_bayes_classifier = GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
ypred_naive_bayes = naive_bayes_classifier.predict(X_train);  # spam prediction for train
cm_naive_bayes_train = confusion_matrix(y_train, ypred_naive_bayes); # train confusion matrix
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);  # spam prediction
cm_naive_bayes_test = confusion_matrix(y_test, ypred_naive_bayes); # test confusion matrix 
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);  #  prediction probabilities

accuracy = naive_bayes_classifier.score(X_test, y_test);
print("Test Accuracy of Naive Bayes: {:.2f}%".format(accuracy * 100));

# Plotting confusion matrix of Bayes classifier
# On Train data
plot_confusion_matrix(naive_bayes_classifier, X_train, y_train, display_labels = target_names);
plt.title('Confusion matrix for Train data (Naive Bayes classifier)')
plt.xlabel('Predicted');
plt.ylabel('Actual');

# On Test data
plot_confusion_matrix(naive_bayes_classifier, X_test, y_test, display_labels = target_names); 
plt.title('Confusion matrix for Test data (Naive Bayes classifier)')
plt.xlabel('Predicted');
plt.ylabel('Actual');

# Plotting ROC curve
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, yprobab_logreg[:,1]);
roc_auc_logreg = auc(fpr_logreg, tpr_logreg);

fpr_naive_bayes, tpr_naive_bayes, _ = roc_curve(y_test, yprobab_naive_bayes[:,1]);
roc_auc_naive_bayes = auc(fpr_naive_bayes, tpr_naive_bayes);

plt.figure();
lw = 2;
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=lw, label='Logistic regression (AUC = %0.2f)' % roc_auc_logreg);
plt.plot(fpr_naive_bayes, tpr_naive_bayes, color='blue',
         lw=lw, label='Naive Bayes (AUC = %0.2f)' % roc_auc_naive_bayes);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve');
plt.legend(loc="lower right");
plt.show();

#Finding the optimal cluster number
Max_K = 9
SSE =np.zeros((Max_K-2))
DB = np.zeros((Max_K-2))
for i in range(Max_K-2):
    n_c = i+2
    kmeans = KMeans(n_clusters=n_c)
    kmeans.fit(X)
    labels=kmeans.labels_
    SSE[i] = kmeans.inertia_
    DB[i] = davies_bouldin_score(X, labels)
    
#Visualization of DB scores
plt.figure()
plt.title('Davies-Bouldin score curve')
plt.xlabel('Number of clusters')
plt.ylabel("DB index")
plt.plot(np.arange(2,Max_K),DB,color='blue')
plt.show();

#From the graph, we can see that the optimal cluster number is 2

# Kmeans clustering with K=2
kmeans = KMeans(n_clusters=2);  # instance of KMeans class
kmeans.fit(data);   #  fitting the model to data
labels = kmeans.labels_;  # cluster labels
centers = kmeans.cluster_centers_;  # centroid of clusters
distX = kmeans.transform(data);
dist_center = kmeans.transform(centers);

# Visualizing of clustering in the distance space
fig = plt.figure();
plt.title('Pulsars data in the distance space');
plt.xlabel('Cluster 1');
plt.ylabel('Cluster 2');
plt.scatter(distX[:,0],distX[:,1],s=50,c=labels);  # data
plt.scatter(dist_center[:,0],dist_center[:,1],s=200,c='red',marker='X');  # centroids
plt.show();