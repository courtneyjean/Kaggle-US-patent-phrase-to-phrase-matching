from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm

models = {
    "logistic_regression": linear_model.LogisticRegression(),
    "decision_tree": tree.DecisionTreeClassifier(),
    "XGBoost": XGBClassifier(),
    "random_forest": RandomForestClassifier(criterion='gini', max_depth=8, n_estimators=30, random_state=9),
    "light_gbm": lightgbm.LGBMClassifier(),
    "XGBoost_lr0.01_max_depth3": XGBClassifier(eta=0.01, max_depth=3),
    "XGBoost_lr0.01_max_depth6": XGBClassifier(eta=0.01, max_depth=6),
    "XGBoost_lr0.01_max_depth9": XGBClassifier(eta=0.01, max_depth=9),
    "XGBoost_lr0.06_max_depth3": XGBClassifier(eta=0.06, max_depth=3),
    "XGBoost_lr0.06_max_depth6": XGBClassifier(eta=0.06, max_depth=6),
    "XGBoost_lr0.06_max_depth9": XGBClassifier(eta=0.06, max_depth=9),
    "XGBoost_lr0.1_max_depth3": XGBClassifier(eta=0.1, max_depth=3),
    "XGBoost_lr0.1_max_depth6": XGBClassifier(eta=0.1, max_depth=6),
    "XGBoost_lr0.1_max_depth9": XGBClassifier(eta=0.1, max_depth=9),
    "XGBoost_lr0.2_max_depth3": XGBClassifier(eta=0.1, max_depth=3),
    "XGBoost_lr0.2_max_depth6": XGBClassifier(eta=0.1, max_depth=6),
    "XGBoost_lr0.2_max_depth9": XGBClassifier(eta=0.1, max_depth=9),
    "XGBoost_lr0.3_max_depth3": XGBClassifier(eta=0.1, max_depth=3),
    "XGBoost_lr0.3_max_depth6": XGBClassifier(eta=0.1, max_depth=6),
    "XGBoost_lr0.3_max_depth9": XGBClassifier(eta=0.1, max_depth=9),
}