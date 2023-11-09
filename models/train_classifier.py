import sys
import pandas as pd
from sqlalchemy import create_engine
import joblib

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    Load data from SQLite database and split into feature and target datasets.
    
    Parameters:
    database_filepath (str): File path of the SQLite database.
    
    Returns:
    tuple: Feature data (X), target data (Y), and category names.
    """
    # Connect to database and load data into dataframe
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', con=engine)
    
    # Split data into features and targets
    X = df['message'].values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and process text data (messages).
    
    Parameters:
    text (str): Input text to tokenize.
    
    Returns:
    list: List of clean tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Lemmatize, normalize case, and remove stopwords
    clean_tokens = [
        lemmatizer.lemmatize(tok.lower().strip()) 
        for tok in tokens if tok not in stopwords.words('english')
    ]

    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline with GridSearchCV.
    
    Returns:
    GridSearchCV: Grid search model object with pipeline and parameters.
    """
    # Define a separate random forest for each feature using class_weight=balanced
    # to reduce effect of class imbalances in the data
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    multi_output_model = MultiOutputClassifier(model, n_jobs=-1)

    # Create a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multi_output_model)
    ])

    # Define parameters for GridSearch
    parameters = {
        'clf__estimator__n_estimators': [100],
        'clf__estimator__max_depth': [5, 10],
        'clf__estimator__min_samples_leaf': [1, 2, 5, 10]
    }

    # Score the model based on macro averaged recall to minimize false negatives
    # and to not skew the score by weighting the averages by respective support
    scorer = make_scorer(recall_score, average='macro')

    # Grid search with cross-validation
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, cv=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model performance on test data and print out the classification report.
    
    Parameters:
    model: Trained machine learning model.
    X_test: Test features dataset.
    Y_test: Test target dataset.
    category_names: List of category names for target.
    """
    Y_pred = model.predict(X_test)
    print("Best Parameters:", model.best_params_)

    for i, col in enumerate(category_names):
        print(f'Classification Results for column: {col}')
        print(classification_report(Y_test[:, i], Y_pred[:, i]))
        print('-'*50)


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    
    Parameters:
    model: Trained machine learning model.
    model_filepath (str): File path to save the model.
    """
    joblib.dump(model, model_filepath)


def main():
    """
    Main function to execute the machine learning pipeline.
    
    Load data, build model, train, evaluate and save the trained model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
