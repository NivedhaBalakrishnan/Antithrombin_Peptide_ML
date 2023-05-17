# Nivedha - May 11 7:50 AM
- Peter ran
    - Hypertune all models with all features
    - RFE with SVC linear, Logistic Regression and Random Forest
    - Hypertune SVC linear, Logistic Regression and Random Forest
- Code Changes
    - Code is ready for second run with RFE with XGB and SFS with SVC RBF and KNN
- File Changes
    - Dependency
        - Merging tuned models for all features (before_fs) to main repo
        - Merging tuned models for SVC linear, Logistic Regression and Random Forest (after_fs) to main repo
        - Merging best features for SVC linear, Logistic Regression and Random Forest to main repo
    - Results
        - Merging results in hypertune and feature selection to main repo

# Nivedha - May 11 6:30 PM
- Running models with different ratio
    - Removing xgb from feature_selection.py for the sake of version mismatch (I assume)

# Nivedha - May 12 1.10 PM
- Peter XGB version is xgboost==1.6.1, installing that in my env
- Peter gave XGB results - right now we don't have feature selection results SVR RBF and KNN
- Getting Kfold results for all models for which we have results, and using SVR Linear results for SVR RBF.

# Nivedha - May 12 Meeting Notes
- The variance id test results of kfold is too large, find ways to sort it out, before proceeding further.

# Nivedha - May 17
- Tried SMOTE,