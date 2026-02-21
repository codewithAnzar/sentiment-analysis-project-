Sentiment Analysis on Movie Reviews
Project Overview
This project demonstrates a complete machine learning pipeline for binary sentiment classification using movie reviews. The goal is to automatically classify reviews as positive or negative using text processing and machine learning techniques.
Problem Statement
Manual sentiment analysis of large volumes of text data is time-consuming. This project automates the classification of movie reviews into positive or negative sentiments, which is applicable to product reviews, social media monitoring, and customer feedback analysis.
Methodology
1. Data Preparation

Curated a dataset of 20 movie reviews labeled as positive (1) or negative (0)
Balanced dataset with equal distribution of both classes

2. Feature Extraction

Used TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
Extracted top 100 features from review text
Applied standard English stopword removal and lowercasing for preprocessing

3. Model Selection

Implemented Random Forest Classifier with 100 estimators
Chosen for its robustness, interpretability, and ability to capture non-linear relationships in text data

4. Model Training & Evaluation

80-20 train-test split
Evaluated using multiple metrics: Accuracy, Precision, Recall, and F1-Score
Generated confusion matrix for detailed performance analysis

Key Results

Accuracy: High performance on test set
Precision & Recall: Balanced performance across both classes
Top Features Identified: Words like "amazing," "terrible," "brilliant," "awful" strongly correlate with sentiment

Technical Skills Demonstrated
✓ Data preprocessing and cleaning
✓ Feature engineering with TF-IDF vectorization
✓ Machine learning model implementation (scikit-learn)
✓ Model evaluation and metrics interpretation
✓ Data visualization (confusion matrix, feature importance)
✓ Python libraries: pandas, numpy, scikit-learn, matplotlib
Learning Outcomes

Practical understanding of NLP and text classification workflows
Experience with scikit-learn's full ML pipeline
Hands-on knowledge of feature importance analysis
Ability to evaluate and interpret ML model performance

How to Run

Clone the repository or download the .py file
Install required libraries: pip install scikit-learn pandas numpy matplotlib
Run the Python script in Google Colab or Jupyter Notebook
Review output metrics and visualizations

Libraries Used

pandas - Data manipulation and analysis
numpy - Numerical computing
scikit-learn - Machine learning algorithms and utilities
matplotlib - Data visualization

Future Enhancements

Expand dataset with real movie reviews from IMDb or similar sources
Experiment with different models (SVM, Naive Bayes, Neural Networks)
Implement deep learning approaches using LSTM or BERT embeddings
Deploy as a web application
