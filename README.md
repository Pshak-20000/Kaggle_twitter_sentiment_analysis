Project Summary: Sentiment Analysis on Twitter Data Objective:

The goal was to perform sentiment analysis on a Twitter dataset, classifying tweets into categories such as Positive, Negative, Neutral, and Irrelevant. Data Preparation:

Dataset: You started with a dataset containing tweets, which included text and sentiment labels. Initial Exploration: You explored the dataset to understand its structure, including the number of entries and types of sentiments present. Data Cleaning:

Cleaned the text data by removing unwanted characters, stop words, and applying techniques like lowercasing and lemmatization to make the text suitable for analysis. Sentiment Distribution:

You analyzed the distribution of sentiments in the dataset, noting counts for each class, which is crucial for understanding class balance. Feature Extraction:

Used TF-IDF Vectorization to convert the cleaned text data into numerical format, which allows the model to process the text effectively. Train-Test Split:

Split the dataset into training and validation sets to evaluate the model's performance accurately. python Copy code from sklearn.model_selection import train_test_split X = train_df['cleaned_text'] y = train_df['Sentiment'] X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) Model Selection:

Chose Logistic Regression as the machine learning model for sentiment classification. Handling Convergence Warning:

Initially faced a convergence warning, which was resolved by: Increasing the number of iterations (max_iter). Switching the solver to liblinear. Implementing feature scaling using StandardScaler. python Copy code from sklearn.pipeline import make_pipeline from sklearn.preprocessing import StandardScaler from sklearn.linear_model import LogisticRegression

pipeline = make_pipeline( StandardScaler(with_mean=False), LogisticRegression(max_iter=200, solver='liblinear') ) Model Training:

Trained the model on the training data using the defined pipeline. python Copy code pipeline.fit(X_train_vectorized, y_train) Model
