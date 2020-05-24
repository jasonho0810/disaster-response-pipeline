import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', \
               'maxent_ne_chunker'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import ne_chunk

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    INPUT
    database_filepath - file path to the SQL database
    
    OUTPUT
    X - predictor disaster messages
    Y - response message category labels
    cat_names - list of category labels
    '''
   
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster', con = engine)
    X = df['message']
    cat_names = df.columns.values[4:]
    Y = df[cat_names]
    
    return X, Y, cat_names


def tokenize(text):
    '''
    INPUT
    text - a disaster message text
    
    OUTPUT
    clean_tokens - clean word tokens of text with normalized cases, \
    punctuations and stop words removed, and in their root form. 
    '''

    # Normalize text
    text = re.sub(r"(\d)[^a-zA-Z0-9_](\d)", "\\1\\2", text.lower())
    text = re.sub(r"[^a-zA-Z0-9_]", " ", text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    INPUT
    None
    
    OUTPUT
    cv - TfidfVectorizer and Random Forest classifier pipeline with optimal 
    hyperparameters for sublinear tf scaling and max tree depth.
    '''

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('rfclf', MultiOutputClassifier(
            RandomForestClassifier(n_jobs=-1, random_state=42), n_jobs=-1)),
    ])

    # params dict to tune a model
    parameters = {
        'tfidf__sublinear_tf': [True, False],
        'rfclf__estimator__max_depth': np.linspace(10, 100, 10, dtype=int),
    }
    
    # instantiate a gridsearchcv object with the params defined
    model = GridSearchCV(model, param_grid=parameters, cv=3)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - classification model
    X_test - testing predictor disaster messages
    Y_test - testing response message category labels
    category_names -  list of category labels
    
    OUTPUT
    None
    Prints a classification report evaluation for each category label
    '''
     
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test[category_names]):
        print(col)
        try:
            print(classification_report(Y_test[col], Y_pred[:, i], \
                                        target_names=['yes', 'no']))
        except:
            print('naive behaviour')


def save_model(model, model_filepath):
    '''
    INPUT
    model - classification model
    model_filepath - file path to classification model pickle file
    
    OUTPUT
    None
    Stores or replaces classification model as a pickle file
    '''
        
    pickle.dump(model, open(model_filepath, 'wb'))


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