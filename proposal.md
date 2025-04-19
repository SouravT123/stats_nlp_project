OPTION 2: Modeling Experiment
The experiment is about building a sentiment analysis model for an e-commerce platform. The goal is to automatically classify customer product reviews into Positive, Negative, or Neutral sentiment categories. This is needed because the platform has experienced significant growth, and the volume of reviews has become too large to monitor and analyze manually. The model aims to help the business proactively understand customer feedback and address issues, particularly negative sentiment.

Project Approach

The modeling approach involves several steps:
Text Preprocessing: 
	Cleaning the review text by removing special characters, converting to lowercase, removing extra whitespace, removing stopwords (using NLTK), and 	applying stemming (using NLTK PorterStemmer).

Text Vectorization: 
	Converting the cleaned text into numerical features using three different methods for comparison:
		Bag-of-Words (BoW) using Scikit-learn's CountVectorizer.
		Word2Vec embeddings using Gensim, representing each review by the average of its word vectors.
		GloVe pre-trained embeddings (loaded via Gensim's KeyedVectors), also representing each review by the average of its word vectors.

Classification Model: 
	Using a RandomForestClassifier from Scikit-learn as the predictive model.

Handling Imbalance: 
	Addressing the class imbalance observed in the sentiment distribution by using the class_weight='balanced' parameter in the Random Forest classifier.

Hyperparameter Tuning: 
	Using GridSearchCV to find the best hyperparameters for the Random Forest model, optimizing based on weighted recall score.



Plan for Evaluating the Model

The plan for evaluating the model includes:

Data Splitting: 
	Dividing the dataset into training (80%), validation (10%), and test (10%) sets using stratified splitting to maintain class proportions.
Metrics:
	 Using standard classification metrics:
Confusion Matrix: 
	To visualize the performance for each class (Negative, Neutral, Positive), showing true vs. predicted labels and percentages.
Classification Report:
	 To calculate precision, recall, F1-score, and support for each sentiment class.
Evaluation Sets: 
	Assessing model performance on:
	The training set (to check for overfitting).
	The validation set (during hyperparameter tuning and for intermediate evaluation).
	The final hold-out test set (to report the generalized performance of the chosen tuned model).

Focus: 
	Particular attention is paid to the model's ability to correctly identify the minority classes (Negative and Neutral reviews), especially after addressing class imbalance.


Datasets: https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset/data

Github link: https://github.com/SouravT123/stats_nlp_project
