import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import decomposition

from time import time
from PIL import Image

plt.ion()


def image_drawer(dirr, df, row_nber):
    selected_row = df.ix[row_nber, 1:].values
    im = Image.new('L', (28, 28))
    im.putdata(selected_row)
    im.save(dirr+'test.png')


def data_reader(file):
    print("Extracting data from csv file")
    df = pd.read_csv(file)
    print("End of extraction step")
    return df


def splitting_dataset(df):
    print("Splitting data into train and test sets")
    X = (df.ix[:,1:]).as_matrix()
    y = df.ix[:,0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=45)

    print("End of split step")
    return X_train, X_test, y_train, y_test


def principal_components(dirr, X_train, X_test, percentage):
    pca = decomposition.PCA()
    pca.fit(X_train)
    variance = 100. * pca.explained_variance_ratio_

    var_df = pd.DataFrame(np.cumsum(variance), columns=['Cumul. var.'])
    var_df['Components'] = np.arange(0, X_train.shape[1])
    var_df['Variance'] = variance
    var_df.to_csv(dirr+'variance.csv', index=False)
    # The first 152 components explain 95% of the variance
    # The first 259 components explain 98% of the variance
    # The first 329 components explain 99% of the variance

    # plt.figure(1000)
    # plt.plot(variance)
    # plt.xlabel('Principal component')
    # plt.ylabel('Proportion of variance explained (%)')
    # plt.savefig(dirr+'Scree_plot.png')
    # plt.close()
    #
    # plt.figure(2000)
    # plt.plot(np.cumsum(variance))
    # plt.xlabel('Principal component')
    # plt.ylabel('Cumulative proportion of variance explained (%)')
    # plt.savefig(dirr+'Scree_plot_cumulative.png')
    # plt.close()

    nbr_components = (var_df[var_df['Cumul. var.'] < percentage]).shape[0]
    print("%s components" %nbr_components)

    # Selection of the nbr_components first components
    pca.n_components = nbr_components
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.fit_transform(X_test)

    return X_train_reduced, X_test_reduced


def classification(X_train, X_test, y_train, y_test, X_actual_test, classif_list, classif_names, dirr, study):

    print("Classifying data")
    # Iteration over classifiers
    # Help from: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    master_accuracy = []
    master_time = []
    for name, model in zip(classif_names, classif_list):
        t0 = time()

        print("========================================")
        print("Classification by %s" %name)
        print("========================================")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        t1 = time()
        master_time.append(t1-t0)

        cm = confusion_matrix(y_test, y_pred)
        print(">> Confusion matrix:")
        print(cm)

        accuracy = 100. * np.sum(cm.diagonal()) / np.sum(cm)
        print(">> Accuracy = %s %%" %round(accuracy,2))
        master_accuracy.append(accuracy)

        pred_df = pd.DataFrame(np.arange(1, len(y_pred)+1), columns=['ImageId'])
        pred_df['Label'] = y_pred
        pred_df.to_csv(dirr+name+'_predicted_class.csv', index=False)

        if study == 'final':
            y_actual_pred = model.predict(X_actual_test)
            actual_pred_df = pd.DataFrame(np.arange(1, len(y_actual_pred) + 1), columns=['ImageId'])
            actual_pred_df['Label'] = y_actual_pred
            actual_pred_df.to_csv(dirr + name + '_actual_predicted_class.csv', index=False)

    print("End of classification step")

    accuracy_df = pd.DataFrame(classif_names, columns=['Classifier'])
    accuracy_df['Accuracy (%)'] = master_accuracy
    accuracy_df['Run time (sec)'] = master_time

    # plt.figure(3000)
    # plt.plot(accuracy_df['Accuracy (%)'].values, marker='o')
    # plt.xticks(list(np.arange(accuracy_df.shape[0])), accuracy_df['Classifier'].values)
    # plt.xlabel('Algorithms')
    # plt.ylabel('Accuracy (%)')
    # plt.savefig(dirr + study + '_Algorithms_Ranking.png')
    # plt.close()

    accuracy_df = accuracy_df.sort_values(by=['Accuracy (%)', 'Classifier'], ascending=[False, True])

    print("Ranking of classifiers:")
    print(accuracy_df)


def main():
    """

    :return: Classifies hand-written digits images
    """
    dirr = '/Users/alexandra/ProgCode/Coursera_DataScienceAtScale/datasci_course_materials/Kaggle_DigitRecognizer/'
    file = dirr + 'train.csv'
    file2 = dirr + 'test.csv'

    # classifiers = [
    #     RandomForestClassifier(),
    #     SVC(kernel="linear"),
    #     SVC(),
    #     SVC(C=3),
    #     SVC(C=30),
    #     SVC(C=300),
    #     LogisticRegression()]
    # classif_names = ['Random Forest', 'Linear SVM', 'RBF SVM',
    #                  'RBF SVM C=3', 'RBF SVM C=30', 'RBF SVM C=300',
    #                  'Logistic Regression']

    # classifiers = [
    #     RandomForestClassifier(),
    #     AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=100,learning_rate=1)]
    # classif_names = ['Random Forest', 'Adaboost']

    # classifiers = [RandomForestClassifier()]
    # classif_names = ['Random Forest']

    # Extraction of training data
    df = data_reader(file)
    df2 = data_reader(file2)

    # Splitting of training data into train and test sets
    # No normalization needed as all values are between 0 and 255
    X_train, X_test, y_train, y_test = splitting_dataset(df)
    X_actual_test = df2.as_matrix()


    # classification(X_train, X_test, y_train, y_test, classifiers, classif_names, dirr, 'Default')
    #  For simple case, i.e. no PCA, no grid search

    # Principal components
    # percentage = 99.0
    # for pc in ['without', 'with']:  # 'without' first not to have an impact on 'with'
    #     print("%s principal components" %pc.upper())
    #     if pc == 'with':
    #         # Principal components
    #         X_train, X_test = principal_components(dirr, X_train, X_test, percentage)
    #
    #     # Initial assumption: all features are important and used in the model
    #     # In a second time, we will use PCA to reduce dimensionality + grid search (gradient descent)
    #     # Testing of 4 different methods:
    #     # - Random Forest
    #     # - SVM
    #     # - Logistic Regression
    #     classification(X_train, X_test, y_train, y_test, classifiers, classif_names, dirr, 'Princ_compo')


    # Grid search
    # classifiers = []
    # classif_names = []
    # # for nbr_estim in range(5, 100, 5):
    # for nbr_estim in range(65, 86, 1):
    #     classifiers.append(RandomForestClassifier(n_estimators=nbr_estim))
    #     classif_names.append(str(nbr_estim))
    # classification(X_train, X_test, y_train, y_test, classifiers, classif_names, dirr, 'Grid_search_refined_2_')


    # Final predictor
    classifiers = [RandomForestClassifier(), RandomForestClassifier(n_estimators=83)]
    classif_names = ['Random_Forest_Default', 'Random_Forest_83_estimators']

    # classification(X_train, X_test, y_train, y_test, X_actual_test, classifiers, classif_names, dirr, 'RF_10_and_83_estimators_')
    classification(X_train, X_test, y_train, y_test, X_actual_test, classifiers, classif_names, dirr,'final')

if __name__ == '__main__':
    main()




