#!/bin/sh

# python train.py --model XGBoost_lr0.01_max_depth3 --preproc ft
# python train.py --model XGBoost_lr0.01_max_depth6 --preproc ft
# python train.py --model XGBoost_lr0.01_max_depth9 --preproc ft
python train.py --model XGBoost_lr0.06_max_depth3 --preproc ft
python train.py --model XGBoost_lr0.06_max_depth6 --preproc ft
python train.py --model XGBoost_lr0.06_max_depth9 --preproc ft
python train.py --model XGBoost_lr0.1_max_depth3 --preproc ft
python train.py --model XGBoost_lr0.1_max_depth6 --preproc ft
python train.py --model XGBoost_lr0.1_max_depth9 --preproc ft
python train.py --model XGBoost_lr0.2_max_depth3 --preproc ft
python train.py --model XGBoost_lr0.2_max_depth6 --preproc ft
python train.py --model XGBoost_lr0.2_max_depth9 --preproc ft
python train.py --model XGBoost_lr0.3_max_depth3 --preproc ft
python train.py --model XGBoost_lr0.3_max_depth6 --preproc ft
python train.py --model XGBoost_lr0.3_max_depth9 --preproc ft

# python train.py --model logistic_regression
# python train.py --model decision_tree
# python train.py --model XGBoost