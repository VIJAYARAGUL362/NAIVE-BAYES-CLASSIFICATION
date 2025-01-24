# Naive Bayes Classifier for Social Network Ads

This Colab notebook demonstrates the implementation of a Naive Bayes classifier for predicting customer purchase behavior based on social network ads. The notebook provides a step-by-step guide to building, evaluating, and visualizing the model's performance.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Workflow](#workflow)
4. [Libraries Used](#libraries-used)
5. [How to Use](#how-to-use)
6. [Results and Evaluation](#results-and-evaluation)
7. [Visualization](#visualization)
8. [Conclusion](#conclusion)

## Introduction

This project aims to predict whether a customer will purchase a product based on their demographics and social network ad interactions. We will use a Naive Bayes classifier, a probabilistic algorithm based on Bayes' theorem, for this classification task. Naive Bayes classifiers are known for their simplicity, efficiency, and effectiveness in various classification problems, especially with high-dimensional data. They are based on the assumption of feature independence, which simplifies the calculation of probabilities.

## Dataset

The notebook uses the "Social Network Ads" dataset, which contains the following features:

- **User ID:** Unique identifier for each user.
- **Gender:** Male or Female.
- **Age:** Age of the user in years.
- **EstimatedSalary:** Estimated salary of the user in dollars.
- **Purchased:** Whether the user purchased the product (0 for No, 1 for Yes).

The dataset is loaded from a CSV file named "Social_Network_Ads.csv". Make sure this file is accessible in your Colab environment or mounted Google Drive.

## Workflow

The notebook follows these steps:

1. **Data Loading and Preprocessing:** Import necessary libraries (pandas, numpy, matplotlib, scikit-learn), load the dataset from the CSV file, and split the dataset into features (X) and the target variable (y). The features are 'Age' and 'EstimatedSalary', while the target variable is 'Purchased'.
2. **Data Splitting:** Divide the dataset into training and testing sets using `train_test_split` from scikit-learn with a `test_size` of 0.25 and `random_state` of 0. This means 75% of the data is used for training and 25% for testing.
3. **Feature Scaling:** Apply feature scaling using `StandardScaler` to standardize the features. This ensures that features with different scales do not disproportionately influence the model's learning process. StandardScaler transforms the data by subtracting the mean and dividing by the standard deviation.
4. **Model Training:** Train a Gaussian Naive Bayes classifier using the training data. This involves fitting the model to the training data to learn the relationship between features and the target variable. The Gaussian Naive Bayes classifier assumes that the features follow a Gaussian distribution.
5. **Prediction and Evaluation:** Make predictions on the training and testing sets using the trained model. Evaluate the model's performance using metrics like accuracy score and confusion matrix. These metrics provide insights into the model's ability to correctly classify instances.
6. **Cross-Validation:** Perform 10-fold cross-validation using `cross_val_score` from scikit-learn. This technique assesses the model's generalization ability and provides a more robust estimate of its performance on unseen data. It involves splitting the training data into 10 folds, training the model on 9 folds, and testing on the remaining fold. This process is repeated 10 times, and the average accuracy is calculated.
7. **Visualization:** Visualize the results using a scatter plot, showing the decision boundary learned by the classifier and the classification of data points in the training and testing sets. This helps to understand how the model separates the different classes. The visualization uses `matplotlib.pyplot` to create a contour plot and scatter plot.

## Libraries Used

- **pandas:** For data manipulation and analysis.
- **numpy:** For numerical computations.
- **matplotlib:** For data visualization.
- **scikit-learn:** For machine learning algorithms, model selection, and evaluation.

## How to Use

1. Open the Colab notebook.
2. Ensure the dataset file ("Social_Network_Ads.csv") is accessible in your Colab environment or mounted Google Drive.
3. Run the code cells sequentially.
4. Observe the results, evaluation metrics, and visualizations.

## Results and Evaluation

The notebook presents the accuracy score, confusion matrix, and cross-validation results to evaluate the model's performance. These metrics provide insights into the model's ability to correctly classify instances and its generalization ability. The accuracy score represents the percentage of correctly classified instances. The confusion matrix shows the number of true positives, true negatives, false positives, and false negatives. Cross-validation provides an average accuracy score and standard deviation across the 10 folds.

## Visualization

The notebook includes a visualization of the decision boundary learned by the classifier. This helps to understand how the model separates the different classes and the distribution of data points in the feature space. The visualization uses a scatter plot to show the data points and a contour plot to represent the decision boundary.

## Conclusion

This notebook provides a practical example of using a Naive Bayes classifier for a binary classification problem. It showcases the steps involved in building, evaluating, and visualizing the model's performance. The results and visualizations help to understand the model's effectiveness and its potential for real-world applications. The Naive Bayes classifier achieved a good accuracy score and demonstrated its ability to classify customer purchase behavior based on social network ads.
