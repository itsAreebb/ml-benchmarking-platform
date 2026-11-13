Hello everyone, this is Areeb Ali!

After completing my machine learning learning journey, I decided to use my weekend to build something fun and meaningful. That’s how this project was born.

I created an AutoML Benchmarking System that compares and analyzes the performance of various classical machine learning models across multiple datasets. The idea was not just to train models, but to build a system that can experiment, evaluate, rank, and visualize how different algorithms behave in real scenarios.

The datasets used are mainly “toy” and standard datasets provided by the scikit-learn library (along with a few from OpenML), but the focus of the project is on system design and experimentation, not on the complexity of the data itself.

The core goal of this project is to:

1. Analyze model performance
2. Compare training time vs accuracy
3. Understand which models are more stable and powerful across different datasets
4. Observe how model families behave (linear, probabilistic, tree-based, ensemble, boosting)

In simple terms, this project answers:

“Which machine learning model works best, where, and why?”

* Summary of Observations

--> One clear pattern that emerged from the experiments is:
--> Ensemble learning methods consistently outperform simpler models.

Boosting-based models such as:

1. XGBoost
2. Gradient Boosting
3. AdaBoost 
dominated the rankings across most datasets.

They were followed by:
4. Random Forests
5. Probabilistic models (Naive Bayes)
6. Linear and distance-based models

This strongly validates the idea that ensemble methods combine multiple weak learners to create powerful and stable predictors.

Objective of the Project:

The objective was not to build “the best model”, but to build a framework that can:
Automatically detect problem type (classification / regression),Train multiple models
Evaluate their performance,Rank them per dataset,Rank them globally,Visualize performance trends , Handle broken datasets without crashing.

This makes the project closer to a mini AutoML research platform rather than a single ML script.

Datasets:

The datasets are a mixture of Classification and Regression problems.

Mainly sourced from:
1. scikit-learn.datasets
2. OpenML

Models Used

Some of the main models included:

--> XGBoostRegressor / XGBoostClassifier
--> GradientBoostingRegressor / GradientBoostingClassifier
--> AdaBoostRegressor / AdaBoostClassifier
--> Random Forest
--> Decision Trees
--> Logistic Regression / Linear Regression
--> KNN
--> Naive Bayes

