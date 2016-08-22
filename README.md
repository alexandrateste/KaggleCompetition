"Digit Recognizer" Kaggle competition
-- Assignment for the Coursera's Data Science At Scale Specialization track) --
(August 2016)

We are provided with a set of 42000 grey-scale images of hand-written digits. Each of these images is composed of 28x28 pixels and displays a number between 0 and 9. Each pixel takes a value between 0 and 255. The training set contains 28x28=784 features and a label, which corresponds to the number drawn on the image.
The goal is to train a model to be able to classify the 28000 images provided in the test set.
The evaluation criterion is accuracy, i.e. the percentage of well classified images in the test set.


This competition is a classification one, so I tried a few methods to determine which one would be the most accurate and the fastest. I compared Random Forest, Linear Support Vector Machine, Radial Basis Function Support Vector Machine, and Logistic Regression.

As a first step, I considered all features. I did not normalize the data since all features are comprised between 0 and 255. I randomly split (with a seed) the provided training set into a training (60%) and a test (40%) sets. I used the same training and test sets for the 4 methods listed above.

Below are the results I obtained on the test set I had created:
           Classifier  Accuracy (%)  Run time (sec)
        Random Forest     93.053571        4.287671
           Linear SVM     90.851190      497.499010
  Logistic Regression     89.226190     1723.002861
              RBF SVM     11.261905     2087.622686

Random forest and linear SVM are the best performing algorithms here. The former is the fastest, taking only 4 seconds to run. So, I used random forest to determine the classes of the images in the competition's test set.


I used python's scikit-learn to conduct the whole analysis:
* train_test_split() to split the training set into a training and a test sets -- I used a seed (random_state) of 45
* RandomForestClassifier(), SVC(kernel="linear"), SVC(), LogisticRegression() to build the 4 models
* confusion_matrix() to compute the confusion matrix and see the number of images well and mis-classified, since the test set I used also contained labels
* time() to measure the time it took each model to be built and to predict classes


The results I obtained with the test set provided in the competition were 93.66% accurate. This is quite decent given that I used the algorithm with its default parameters.


To improve on this result, I tried several optimizations:
* principal component analysis: when keeping only principal components that explain 95% or 99% of the variance observed in the data, I obtained an accuracy of ~ 33% on the test set I had built
* adaboost on decision trees: when using the default parameters, the algorithm reached an accuracy of ~ 85% on the test set I had created
* grid search on the random forest (best performing algorithm here): by default, random forest uses 10 estimators (i.e. decision trees), so I ran the same algorithm, but with different numbers of estimators:
	> first, every 5 from 5 to 95 estimators
	> second, every 1 from 65 to 85 estimators (range on which the accuracy obtained in the first grid search was the highest)
	> third, with 83 estimators (I obtained an accuracy of ~ 95.8% on the test set I had built)
I submitted the classes I obtained with the latter solution. My accuracy reached 95.81%, i.e. 2.15 percentage points higher than with the default parameters. This accuracy is consistent with the one I obtained on the test set I had created.

