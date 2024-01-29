# Sentiment-Analysis
Sentiment analysis is a popular task in natural language processing. The goal of sentiment analysis is to classify the text based on the mood or mentality expressed in the text, which can be positive negative, or neutral.




sentiment analysis is a methodology for analyzing a piece of text to discover the sentiment hidden within it. It accomplishes this by combining machine learning and natural language processing (NLP). Sentiment analysis allows you to examine the feelings expressed in a piece of text.


1.DATA EXPLORATION:

The dataset contains more than 14000 tweets data samples classified into 3 types: positive, negative, neutral.

Tools and Libraries used
* Python – 3.x
* Pandas – 1.2.4
* Matplotlib – 3.3.4
* TensorFlow – 2.4.1


To install the above modules into your local machine, run the following command in your command line.

  --> (pip install pandas matplotlib tensorflow)

  STEPS TO BUILD  SENTIMENT ANALYSIS TEXT CLASSIFIER IN DATA SCIENCE:

  2. DATA PREPROCESSING:

  3. As we are dealing with the text data, we need to preprocess it using word embeddings.

Let’s see what our data looks like.

import pandas as pd
  --> (df = pd.read_csv("./DesktopDataFlair/Sentiment-Analysis/Tweets.csv") )
