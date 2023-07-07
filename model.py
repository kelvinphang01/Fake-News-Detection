# Import necessary libraries
import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import plot_partial_dependence
from pdpbox import pdp

def load_dataset(feature_set):
    # Load dataset
    df = pd.read_csv(os.path.join(os.getcwd(), "final_dataset.csv"))
    
    # Replace missing values with 0
    df = df.fillna(0)
    
    # Fix the 'url_protocol' column using one-hot encoding
    df = pd.get_dummies(df, columns=['url_protocol'])
    
    # Select features and the target variable (fakenews)
    y = df['fakenews']
    
    X1 = df.iloc[:, [2, 3, 4, 22, 23, 24]] #news
    X2 = df.iloc[:, 5:13] #other
    X3 = df.iloc[:, 13:22] #derived
    
    feature_sets = {
    'A': X1,
    'B': X2,
    'C': X3,
    'A+B': pd.concat([X1, X2], axis=1),
    'A+C': pd.concat([X1, X3], axis=1),
    'B+C': pd.concat([X2, X3], axis=1),
    'A+B+C': pd.concat([X1, X2, X3], axis=1)
    }

    if feature_set in feature_sets:
        X = feature_sets[feature_set]
    else:
        # Raise error when an invalid set is provided
        raise ValueError("Invalid feature set provided.")
    
    return X, y

def modelling(): 
    model = Model(None)
    
    X, y = load_dataset('A+B+C')
    best_model_select = model.modelling(X, y)
    
    return best_model_select
    
def train(best_model_select, feature_set):
    model = Model(best_model_select)
    
    X, y = load_dataset(feature_set)    
    best_model = model.best_model_train(X, y)
    model.feature_importances(best_model, X)
    
    return best_model
    
def predict(X, y):
    y_pred = best_model.predict(X)
    print(classification_report(y, y_pred, digits=4))    

    
class Model:
    def __init__(self, best_model_select):
        self.best_model_select = best_model_select
        
        # Define the models to try
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'LightGBM': LGBMClassifier(),
            'XGBoost': XGBClassifier(),
            'Naive Bayes': BernoulliNB(),
            'SVM': SVC(),
            'Neural Network': MLPClassifier()
        }
        
        # Define the parameter grids for hyperparameter tuning
        self.param_grids = {
            'Logistic Regression': {'logisticregression__C': [0.1, 1, 10], 
                                    'logisticregression__class_weight': [None, 'balanced']},
            'Decision Tree': {'decisiontreeclassifier__max_depth': [3, 5, 7], 
                              'decisiontreeclassifier__class_weight': [None, 'balanced']},
            'Random Forest': {'randomforestclassifier__n_estimators': [100, 200, 300], 
                              'randomforestclassifier__class_weight': [None, 'balanced']},
            'LightGBM': {'lgbmclassifier__num_leaves': [10, 20, 30], 
                         'lgbmclassifier__class_weight': [None, 'balanced']},
            'XGBoost': {'xgbclassifier__max_depth': [3, 5, 7], 
                        'xgbclassifier__scale_pos_weight': [0.425 / (1 - 0.425)]},
            'Naive Bayes': {'bernoullinb__alpha': [0.1, 0.5, 1.0]},
            'SVM': {'svc__C': [0.1, 1, 10],
                    'svc__kernel': ['linear', 'rbf']},
            'Neural Network': {'mlpclassifier__hidden_layer_sizes': [(32,), (64,), (128,)]}
        }
        
        # Define the data validation techniques
        self.validation_techniques = {
            'Train/Test Split': train_test_split,
            'K-Fold Cross Validation': cross_val_score
        }
        
        # Define the different train/test partition splits
        self.train_test_splits = [0.7, 0.8, 0.9]
        
        # Define the scoring metric for GridSearchCV
        self.scorer = make_scorer(f1_score, pos_label=1)
        
    def modelling(self, X, y):
        # Perform model evaluation using different techniques and hyperparameter tuning
        results = {}
    
        for model_name, model in self.models.items():
            for validation_name, validation_func in self.validation_techniques.items():
                if validation_name == 'Train/Test Split':
                    for split in self.train_test_splits:
                        X_train, X_test, y_train, y_test = validation_func(X, y, test_size=1-split, random_state=42)
                        pipeline = make_pipeline(StandardScaler(), model)
                        grid_search = GridSearchCV(pipeline, self.param_grids[model_name], 
                                                   cv=StratifiedKFold(n_splits=5, shuffle=True), scoring=self.scorer)
                        grid_search.fit(X_train, y_train)
                        y_pred = grid_search.predict(X_test)
                        f1 = f1_score(y_test, y_pred, pos_label=1)
                        results[(model_name, validation_name, split)] = f1
                elif validation_name == 'K-Fold Cross Validation':
                    pipeline = make_pipeline(StandardScaler(), model)
                    grid_search = GridSearchCV(pipeline, self.param_grids[model_name], 
                                               cv=StratifiedKFold(n_splits=5, shuffle=True), scoring=self.scorer)
                    f1_scores = cross_val_score(grid_search, X, y, cv=5, scoring=self.scorer)
                    results[(model_name, validation_name)] = f1_scores.mean()
        
        # Model Selector to select the best model based on average accuracy
        best_model_select = max(results, key=results.get)
        best_f1 = results[best_model_select]
        
        # Print the results
        print("Results:")
        for (model_name, validation_name, *split), f1 in results.items():
            if validation_name == 'Train/Test Split':
                print(f"Model: {model_name}\tValidation: {validation_name}\tSplit: {split}\tF1 Score: {f1}")
            else:
                print(f"Model: {model_name}\tValidation: {validation_name}\tF1 Score: {f1}")
        
        print("\nBest Model:")
        print(f"Model: {self.best_model_select}\tF1 Score: {best_f1}")
        
        return best_model_select
    
    def best_model_train(self, X, y):
        model = self.models[self.best_model_select[0]]
        if self.best_model_select[1] == 'Train/Test Split':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-self.best_model_select[2], random_state=42)
            pipeline = make_pipeline(StandardScaler(), model)
            best_model = GridSearchCV(pipeline, self.param_grids[self.best_model_select[0]], 
                                       cv=StratifiedKFold(n_splits=5, shuffle=True), scoring=self.scorer)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            print(classification_report(y_test, y_pred, digits=4))
            print("Best Parameters:", best_model.best_params_)
            
        else:
            pipeline = make_pipeline(StandardScaler(), model)
            best_model = GridSearchCV(pipeline, self.param_grids[self.best_model_select[0]], 
                                       cv=StratifiedKFold(n_splits=5, shuffle=True), scoring=self.scorer)
            y_pred = cross_val_predict(best_model, X, y, cv=5)

            print(classification_report(y, y_pred, digits=4))
            best_model.fit(X, y)
            print("Best Parameters:", best_model.best_params_)
            
        return best_model
        
    def feature_importances(self, best_model, X):
        best_estimator = best_model.best_estimator_
        best_model_component = best_estimator.steps[-1][1] # Assuming the model is the last step in the pipeline
        try:
            importances = best_model_component.feature_importances_
        except:
            # Handle other models that don't have feature importances
            importances = None
            
        # Create an array of feature names
        feature_names = np.array(X.columns)
        
        # Associate the feature importances with their corresponding variable names
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        print(feature_importances)

# Read the config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract algorithm and train_size from config
algorithm = config.get('algorithm')
train_size = config.get('train_size')

# Determine the best_model_select
if algorithm:
    if train_size:
        best_model_select = (algorithm, 'Train/Test Split', train_size)
    else:
        best_model_select = (algorithm, 'K-Fold Cross Validation')
else:
    best_model_select = modelling()
    
# Train the model
best_model = train(best_model_select, 'A+B+C')