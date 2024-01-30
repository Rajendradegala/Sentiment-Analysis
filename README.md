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

