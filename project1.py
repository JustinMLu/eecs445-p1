"""EECS 445 - Winter 2023.

Project 1
"""

import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import metrics
from matplotlib import pyplot as plt


from helper import *

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)



def extract_word_original(input_string): # THIS WAS THE ORIGINAL ONE, I RENAMED IT FOR Q6 SO AS NOT TO DISTURB ALL OTHER DEPENDENCIES
    """Preprocess review into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    word_list = input_string.lower()
    for p in string.punctuation:
        word_list = word_list.replace(p, " ")
    
    result = word_list.split()
    return result

def extract_word(input_string):
    stopwords = "a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves"
    stopword_list = extract_word_original(stopwords)
    # below is just a slightly modified version of extract_words
    word_list = input_string.lower()
    for p in string.punctuation:
        word_list = word_list.replace(p, " ")
    
    result = word_list.split()
    return [w for w in result if w not in stopword_list]


def extract_dictionary(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | text                          | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    # TODO: Implement this function
    idx = 0
    for index, row in df.iterrows():
        reddit_comment = extract_word(row['text'])
        for word in reddit_comment:
            if word not in word_dict:
                word_dict[word] = idx
                idx += 1
    return word_dict


def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review.  Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    
    # TODO: Implement this function
    word_idx = 0
    for index, row in df.iterrows():
        reddit_comment = extract_word(row['text'])
        for w in reddit_comment:
            if w in word_dict:
                feature_matrix[word_idx][word_dict[w]] = 1 
        word_idx += 1
        
    return feature_matrix


def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    
    elif metric == "f1-score":
        return metrics.f1_score(y_true, y_pred)
    
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)
    
    elif metric == "precision":
        return metrics.precision_score(y_true, y_pred)
    
    elif metric == "sensitivity":
        return metrics.recall_score(y_true, y_pred)
    
    elif metric == "specificity":
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
            
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.



def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful

    scores = []
    kfolds = StratifiedKFold(n_splits = k)

    for train, test in kfolds.split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        clf.fit(X_train, y_train) 

        y_pred = 0
        if metric == "auroc": # Special auroc case
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        
        scores.append(performance(y_test, y_pred, metric))

    mean_score = np.mean(scores) # DEBUG?
    return mean_score


def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True
):
    """Search for hyperparameters from the given candidates of linear SVM with 
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1"ß)
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM
    perf_scores = []
    for c in C_range:
        clf = LinearSVC(penalty=penalty, loss=loss, dual=dual, C=c, random_state=445)
        cv_perf = cv_performance(clf, X, y, k, metric)
        perf_scores.append(cv_perf)
        
    max_perf = max(perf_scores)
    idx_of_max_perf = perf_scores.index(max_perf) # used in C_range

    print("max_performance: ", max_perf) 
    return C_range[idx_of_max_perf]


def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    for c in C_range:
        clf = LinearSVC(penalty = penalty, loss = loss, dual = dual, C = c, random_state = 445)
        clf = clf.fit(X, y)
        
        l0_norm = 0 # THE L0, or 0-NORM of a vector, *IS THE SUM OF ALL NONZERO ELEMENTS*
        theta_vector = clf.coef_[0]
        
        for v in theta_vector: # For every word "feature" in the theta vector (1 if it's there, 0 if it's not)
            if v != 0:
                l0_norm += 1 # Add 1 

        norm0.append(sigma_l0_norm)
    
    plt.plot(C_range, norm0)
    plt.xlabel("C")
    plt.ylabel("norm0")
    plt.legend(["L0 norm"])
    plt.xscale("log")
    plt.title(penalty.upper() + " penalty " + "chart.png")
    plt.savefig(penalty.upper() + " penalty " + "chart.png")
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """Search for hyperparameters from the given candidates of quadratic SVM 
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    perf_scores = []

    for i in param_range: #i[0] = c, i[1] = r
        clf = SVC(kernel='poly', degree=2, C=i[0], coef0=i[1], gamma='auto')
        perf = cv_performance(clf, X, y, k, metric)
        perf_scores.append(perf)

    max_perf = max(perf_scores)
    idx_of_best = perf_scores.index(max_perf)
    best_C_val, best_r_val = param_range[idx_of_best][0], param_range[idx_of_best][1]
    print("Best parameters (C,r): ", best_C_val, best_r_val)
    return best_C_val, best_r_val



# TODO: ATTENTION: IF YOU'RE A GRADER AND ARE TRYING TO RUN MY CODE, PLEASE ONLY UNCOMMENT **SECTIONS** OF CODE 
# (ex. 3.a ONLY, 4.1.c ONLY, etc.), OTHERWISE THE FACT THAT SOME VARIABLES ARE REASSIGNED/REUSED **MAY** RESULT 
# IN UNDEFINED BEHAVIOR. ALSO EXTRACT_WORDS() WILL GIVE A DIFFERENT INPUT UNLESS YOU REVERT THE CHANGES I MADE
# REGARDING THE TRIMMING OF STOPWORDS

def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )


    # TODO: Questions 3, 4, 5
    # # ------------------------QUESTION 3--------------------------
    # TODO: 3.a
    # print(extract_word("It's a test sentence! Does it look CORRECT?"))
    # TODO: 3.b
    # print("Number of unique words (d): ", len(dictionary_binary))
    # TODO: 3.c
    # print("Avg. # of nonzero features: ", np.sum(X_train) / len(X_train)) # Since a feature is 0 or 1, avg. num. nonzero features  = sum of feature vector / its length
    # keys = list(dictionary_binary.keys())
    # values = list(dictionary_binary.values())
    # word_counts = np.sum(X_train, axis=0) # should be a 1 x d vector where d is length of feature vectors
    # max_count_idx = np.argmax(word_counts) # index of word with max. counts (index of most frequent word)
    # print("Most frequently-appearing word: ", keys[values.index(max_count_idx)]) # most frequent word
    # print("Number of appearances: ", np.max(word_counts))
    
    
    # ------------------------QUESTION 4---------------------------
    # # TODO: 4.1.b
    # C_range = [10**n for n in range(-3,4)]
    # metrics = ["precision", "accuracy", "f1-score", "auroc", "sensitivity", "specificity"]
    # for m in metrics:
    #     print("testing metric: ", m)
    #     c = select_param_linear(X_train, Y_train, 5, m, C_range)
    #     print("c value: ", c, "\n")

    # # TODO: 4.1.c
    # metrics = ["precision", "accuracy", "f1-score", "auroc", "sensitivity", "specificity"]
    
    # clf = LinearSVC(loss="hinge", C=0.01, random_state=445)
    # clf.fit(X_train, Y_train)

    # for m in metrics:
    #     print("testing metric: ", m)
    #     if (m == "auroc"):
    #         print(performance(Y_test, clf.decision_function(X_test), "auroc"))
    #     else:
    #         print(performance(Y_test, clf.predict(X_test), m))        

    # TODO: 4.1.d
    # C_range = [10**n for n in range(-3,4)]
    # print(C_range)
    # plot_weight(X_train, Y_train, 'l2', C_range, 'hinge', True)

    # TODO: 4.2.a,b
    # c = select_param_linear(X_train, Y_train, 5, "auroc", C_range, "squared_hinge", "l1", False)
    # clf = LinearSVC(penalty = 'l1', loss = 'squared_hinge', C = c, dual = False, random_state = 445)
    # clf.fit(X_train, Y_train)
    
    # C_range = [10**n for n in range(-3,1)]
    # print("C: ", c)
    # print("AUROC score on test set", performance(Y_test, clf.decision_function(X_test), 'auroc'))

    # plot_weight(X_train, Y_train, 'l1', C_range, "squared_hinge", False)

    # # TODO: 4.3.a.i - NON-CV PERFORMANCE
    # Cr_range = [10**n for n in range(-2,4)] # BOTH C & R RANGE
    # param_range = []
    # for i in Cr_range:
    #     for j in Cr_range:
    #         param_range.append([i, j])
   
    
    # best_params = select_param_quadratic(X_train, Y_train, 5, 'auroc', param_range)
    # # print("Best parameters (C,r): ", best_params[0], best_params[1])
    
    # clf = SVC(kernel='poly', degree=2, C=best_params[0], coef0=best_params[1], gamma='auto')
    # clf.fit(X_train, Y_train)
    # y_pred = clf.decision_function(X_test)

    # print("Performance (AUROC): ", performance(Y_test, y_pred, 'auroc'))

    # # TODO: 4.3.a.ii - NON-CV PERFORMANCE
    # param_range = []
    # C_range = np.random.uniform(.001, 1000, 5)
    # r_range = np.random.uniform(.001, 1000, 5)
    
    # for c in C_range:
    #     for r in r_range:
    #         param_range.append([c, r])
    
    # best_params = select_param_quadratic(X_train, Y_train, 5, 'auroc', param_range)
    # # print("Best parameters (C,r): ", best_params[0], best_params[1])

    # clf = SVC(kernel = 'poly', degree=2, C=best_params[0], coef0=best_params[1], gamma='auto')
    # clf.fit(X_train, Y_train)
    # y_pred = clf.decision_function(X_test)
    # print("Performance (AUROC): ", performance(Y_test, y_pred, 'auroc'))
    
    # TODO: 4.3.b - CROSS-VALIDATED PERFORMANCE
    # clf_grid = SVC(kernel='poly', degree=2, C=1, coef0=100, gamma='auto')
    # clf_grid.fit(X_train, Y_train)
    # print(cv_performance(clf_grid, X_train, Y_train, 5, 'auroc'))

    # clf_random = SVC(kernel='poly', degree=2, C=192.8677950549348, coef0=84.3491104221373, gamma='auto')
    # clf_random.fit(X_train, Y_train)
    # print(cv_performance(clf_random, X_train, Y_train, 5, 'auroc'))


    # ------------------------QUESTION 5---------------------------
    # TODO: 5.1c
    # metrics = ["accuracy", "f1-score", "precision", "sensitivity", "specificity", "auroc"]
    # clf = LinearSVC(loss = 'hinge', C = 0.01, class_weight = {-1:1, 1:10}, random_state = 445)
    # clf.fit(X_train, Y_train)
    # for m in metrics:
    #     if m == "auroc":
    #         print(m, ": ", performance(Y_test, clf.decision_function(X_test), m))
    #     else: 
    #         print(m, ": ", performance(Y_test, clf.predict(X_test), m))
    

    # TODO: 5.2a
    # metrics = ["accuracy", "f1-score", "precision", "sensitivity", "specificity", "auroc"]
    # clf = LinearSVC(loss = 'hinge', C = 0.01, class_weight = {-1:1, 1:1}, random_state = 445)
    # clf.fit(IMB_features, IMB_labels)
    # for m in metrics:
    #     if m == "auroc":
    #         print(m, ": ", performance(IMB_test_labels, clf.decision_function(IMB_test_features), m))
    #     else: 
    #         print(m, ": ", performance(IMB_test_labels, clf.predict(IMB_test_features), m))

    # TODO: 5.3a
    # metrics = ["sensitivity", "specificity", "accuracy", "auroc", "f1-score"] 
    # Wp_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # Wn_multiplier = [1, 2, 3, 4, 5, 6, 7, 8, 9] # greater emphasis on negative weight

    # # Run beeg test
    # for m in metrics:
    #     best_perf = 0.0
    #     best_wp, best_wn = 0.0, 0.0
    #     for wp in Wp_range:
    #         for r in Wn_multiplier:
    #             wn = wp * r
    #             clf = LinearSVC(loss='hinge', C=0.01, class_weight={-1:wn, 1:wp}, random_state = 445)
    #             cv_perf = cv_performance(clf, IMB_features, IMB_labels, 5, m)
    #             # print("Wp=", wp, " | Wn=", wn, " | CV_perf=", cv_perf)
    #             if cv_perf > best_perf:
    #                 best_perf = cv_perf
    #                 best_wp, best_wn = wp, wn
    #     # Prints best Wp and Wn for every metric
    #     print("Best weights for given metric: ", (m, best_wp, best_wn))
    #     print("Performance: ", best_perf)

    # TODO: 5.3b
    # WHO THOUGHT THAT "BIG MAIN METHOD FILLED WITH COMMENTS" WAS A GOOD WAY OF TESTING SECTIONS OF THIS PROJECT????
    # metrics = ["accuracy", "f1-score", "precision", "sensitivity", "specificity", "auroc"]
    # wp_star = 10
    # wn_star = 20
    # clf = LinearSVC(loss = 'hinge', C = 0.01, class_weight = {-1:wn_star, 1:wp_star}, random_state = 445)
    # clf.fit(IMB_features, IMB_labels)
    # for m in metrics:
    #     if m == "auroc":
    #         print(m, ": ", performance(IMB_test_labels, clf.decision_function(IMB_test_features), m))
    #     else: 
    #         print(m, ": ", performance(IMB_test_labels, clf.predict(IMB_test_features), m))
        

    # TODO: 5.4
    # clf_default = LinearSVC(loss = 'hinge', C = .01, class_weight = {-1:1, 1:1}, random_state = 445)
    # clf_custom = LinearSVC(loss = 'hinge', C = .01, class_weight = {-1:20, 1:10}, random_state = 445)
    # clf_default.fit(IMB_features, IMB_labels)
    # clf_custom.fit(IMB_features, IMB_labels)
    
    # y_pred_default = clf_default.decision_function(IMB_test_features)
    # y_pred_custom = clf_custom.decision_function(IMB_test_features)
    # fpri, tpri, _ = metrics.roc_curve(IMB_test_labels, y_pred_default)
    # fprl, tprl, _ = metrics.roc_curve(IMB_test_labels, y_pred_custom)

    # plt.plot(fpri, tpri, label="W_n = 1, W_p = 1")
    # plt.plot(fprl, tprl, label='W_n = 20, W_p = 10')
    # plt.xlabel("False Positives")
    # plt.ylabel("True Positives")
    # plt.legend()
    # plt.title("The ROC Curve")
    # plt.savefig("roc_curve.png")
    # plt.close()
    
    # ------------------------QUESTION 6---------------------------
    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    
    (multiclass_features,
    multiclass_labels,
    multiclass_dictionary) = get_multiclass_training_data()
    heldout_features = get_heldout_reviews(multiclass_dictionary)

    # First, select an optimal C value
    unique_nums = range(1,10)
    power_range = range(-3,3)
    C_range = []
    for p in power_range:
        for n in unique_nums:
            C_range.append(n ** p)
    c_star = select_param_linear(multiclass_features, multiclass_labels, 5, 'accuracy', C_range, 'squared_hinge', 'l2', True)
    

    # Maps the unique/different parameters being tested to their indices in 'classifiers'
    param_dict = {0 : "l1 penalty, squared_hinge", 1 : "l2 penalty, hinge", 2 : "l2 penalty, squared_hinge"}
    
    # Test optimal c-value on a bunch of differing OvO classifiers
    classifiers = []
    classifiers.append(OneVsOneClassifier(LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=c_star, random_state=445)))
    classifiers.append(OneVsOneClassifier(LinearSVC(penalty='l2', loss='hinge', C=c_star, random_state=445)))
    classifiers.append(OneVsOneClassifier(LinearSVC(penalty='l2', loss='squared_hinge', C=c_star, random_state=445)))
    
    performances = [] # SAME LENGTH AS CLASSIFIERS!
    for clf in classifiers:
        cv = cv_performance(clf, multiclass_features, multiclass_labels)
        performances.append(cv)
    
    best_clf_idx = performances.index(max(performances))
    print("Selected c*: ", c_star)
    print("Selected clf params: ", param_dict[best_clf_idx])
    print("cv_performance score: ", max(performances))

    clf_star = OneVsOneClassifier(LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=c_star, random_state=445))
    clf_star.fit(multiclass_features, multiclass_labels)
    y_pred_star = clf_star.predict(heldout_features)
    generate_challenge_labels(y_pred_star, "lujust")


if __name__ == "__main__":
    main()
