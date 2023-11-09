import sys
import pandas as pd
from sqlalchemy import create_engine
import joblib
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
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', con=engine)
    X = df['message'].values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns

    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok.lower().strip()) for tok in tokens if tok not in stopwords.words('english')]

    return clean_tokens


def build_model():
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    multi_output_model = MultiOutputClassifier(model, n_jobs=-1)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()), #,
        ('clf', multi_output_model)
    ])

    parameters = {
       # 'tfidf__smooth_idf': [False] #,
    'clf__estimator__n_estimators': [100], #500, 750], #[25, 100, 300],
    'clf__estimator__max_depth': [5, 10], #, 10, 25]
    'clf__estimator__min_samples_leaf' : [1, 2, 5, 10]
    }

    scorer = make_scorer(recall_score, average='macro')

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, cv=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    print("Best Parameters:", model.best_params_)

    for i, col in enumerate(category_names):
        print(f'Classification Results for column: {col}')
        print(classification_report(Y_test[:, i], Y_pred[:, i]))
        print('-'*50)


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
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