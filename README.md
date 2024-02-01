# Sentiment-Analysis
Sentiment analysis is a popular task in natural language processing. The goal of sentiment analysis is to classify the text based on the mood or mentality expressed in the text, which can be positive negative, or neutral.




sentiment analysis is a methodology for analyzing a piece of text to discover the sentiment hidden within it. It accomplishes this by combining machine learning and natural language processing (NLP). Sentiment analysis allows you to examine the feelings expressed in a piece of text.


1.DATA EXPLORATION:

The dataset contains more than 14000 tweets data samples classified into 3 types: positive, negative, neutral.

![types of sentiment ](https://github.com/Rajendradegala/Sentiment-Analysis/assets/140039152/c80e6cc0-73dc-4c58-ae28-834a3790fa93)


Tools and Libraries used
* Python – 3.x
* Pandas – 1.2.4
* Matplotlib – 3.3.4
* TensorFlow – 2.4.1


To install the above modules into your local machine, run the following command in your command line.

  --> (pip install pandas matplotlib tensorflow)

  STEPS TO BUILD  SENTIMENT ANALYSIS TEXT CLASSIFIER IN DATA SCIENCE:

  2. DATA PREPROCESSING:

 As we are dealing with the text data, we need to preprocess it using word embeddings.

Let’s see what our data looks like.

      import pandas as pd
  --> (df = pd.read_csv("./DesktopDataFlair/Sentiment-Analysis/Tweets.csv") )


We only need the text and sentiment column:

   -->  [review_df = df[['text','airline_sentiment']]
      print(review_df.shape)
      review_df.head(5)]

There are more than 14000 data samples in the sentiment analysis dataset.
  
   Let's check the column names.

--->  df.columns

[](![df columns](https://github.com/Rajendradegala/Sentiment-Analysis/assets/140039152/1346d955-13a6-4aed-8684-0b4e85260740)


We don’t really need neutral reviews in our dataset for this binary classification problem. So, drop those rows from the dataset

![dataset ahead](https://github.com/Rajendradegala/Sentiment-Analysis/assets/140039152/473fa705-09ab-457b-8ce1-4a95362aaf6f)

Check the values of the airline sentiment column.

  review_df["airline_sentiment"].value_counts()

  ![sentiment label count](https://github.com/Rajendradegala/Sentiment-Analysis/assets/140039152/f1fa6c8c-d3fb-4bcf-aa29-192fa398a26c)

The labels for this dataset are categorical. Machines understand only numeric data. So, convert the categorical values to numeric using the factorize() method. This returns an array of numeric values and an Index of categories.

sentiment_label = review_df.airline_sentiment.factorize()
sentiment_label


![sentiment label](https://github.com/Rajendradegala/Sentiment-Analysis/assets/140039152/60644b73-6b8d-4bb6-a257-4c918b39a140)


3.Exploratory Data Analysis(EDA):

You can use the Matplotlib and Seaborn libraries to create visualizations like histograms, scatter plots, and box plots to explore the relationships and distributions within your dataset.

python
import seaborn as sns
import matplotlib.pyplot as plt

# Create a histogram
sns.histplot(data=df, x='your_column', kde=True)
plt.show()


Sure! Here's an example of how you can create a histogram using the Seaborn library in Python:

python
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'sentiment_data' is your dataset containing the sentiment classes
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment_class', data=sentiment_data)
plt.title('Distribution of Sentiment Classes')
plt.show()

4.TEXT VECTORIZATION:


python
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming 'corpus' is your collection of text data


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.shape)


5.MODEL SELECTION:

you can perform model selection using Python's scikit-learn library:

python

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, y_test are your training and testing data

# Initialize the classifiers

nb_classifier = MultinomialNB()
svm_classifier = SVC()
rf_classifier = RandomForestClassifier()

# Train the classifiers

nb_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)

# Make predictions

nb_pred = nb_classifier.predict(X_test)
svm_pred = svm_classifier.predict(X_test)
rf_pred = rf_classifier.predict(X_test)

# Evaluate the models

nb_accuracy = accuracy_score(y_test, nb_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Naive Bayes Accuracy:", nb_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

6.HYPERPARAMETER TUNING:

 You can use techniques like grid search or random search to find the best combination of hyperparameters for your chosen model.  If you're using a Random Forest classifier, you can tune parameters like the number of trees, maximum depth, and minimum samples per leaf.



python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the hyperparameters grid

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the classifier

rf_classifier = RandomForestClassifier()

# Perform grid search

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)


7. CROSS-VALIDATION:

Cross-validation is a crucial technique for assessing the performance of your model and ensuring that it generalizes well to new data. One common method is k-fold cross-validation, where the training set is split into k smaller sets. The model is trained on k-1 of the folds and validated on the remaining fold. This process is repeated k times, with each fold used as the validation set exactly once.


python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# Initialize the classifieR

log_reg_classifier = LogisticRegression()


# Perform 5-fold cross-validation

cv_scores = cross_val_score(log_reg_classifier, X, y, cv=5)


# Print the cross-validation scores

print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())


# MODEL INTERPRETABILITY:


LIME (Local Interpretable Model-agnostic Explanations) is a fantastic technique for model interpretability.It's used to explain the predictions of any machine learning model. LIME works by approximating the predictions of the model in the vicinity of a particular instance.

python
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from lime.lime_text import LimeTextExplainer


# Initialize the classifier

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Initialize the LIME explainer

explainer = LimeTextExplainer(class_names=['class_0', 'class_1'])

# Choose a specific instance for explanation

text_instance = "Your text instance here"

# Generate an explanation for the instance

explanation = explainer.explain_instance(text_instance, rf_classifier.predict_proba, num_features=10)

# Display the explanation

explanation.show_in_notebook()


# Evaluation Metrics:

Evaluation metrics are crucial for assessing the performance of machine learning models. Common metrics include accuracy, precision, recall, F1 score, and area under the ROC curve (AUC-ROC). 

Each metric provides unique insights into how well the model is performing.


For instance, accuracy measures the proportion of correctly classified instances, while precision and recall focus on the model's ability to correctly identify positive instances. 

The F1 score balances precision and recall, making it useful when there is an uneven class distribution.












