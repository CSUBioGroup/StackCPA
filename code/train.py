import os
from model import CpaStacking
import lightgbm as lgb
import catboost as cat
import xgboost as xgb
from sklearn import linear_model
from sklearn.model_selection import KFold
import pandas as pd

def input_file(path):
    """Check if input file exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File %s does not exist.' % path)
    return path


def output_file(path):
    """Check if output file can be created."""

    path = os.path.abspath(path)
    dirname = os.path.dirname(path)

    if not os.access(dirname, os.W_OK):
        raise IOError('File %s cannot be created (check your permissions).'
                      % path)
    return path

import argparse
parser = argparse.ArgumentParser(
    description='Prepare molecular data for the network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog='''This script reads the structures of ligands and pocket(s),
    prepares them for the neural network and saves in a HDF file.
    It also saves affinity values as attributes, if they are provided.
    You can either specify a separate pocket for each ligand or a single
    pocket that will be used for all ligands. We assume that your structures
    are fully prepared.\n\n

    Note that this scripts produces standard data representation for our network
    and saves all required data to predict affinity for each molecular complex.
    If some part of your data can be shared between multiple complexes
    (e.g. you use a single structure for the pocket), you can store the data
    more efficiently. To prepare the data manually use functions defined in
    tfbio.data module.
    '''
)

parser.add_argument('--train_data', '-iX', required=True, type=input_file, nargs='+',
                    help='data for train')
parser.add_argument('--train_label', '-iY', required=True, type=input_file, nargs='+',
                    help='label for train')
parser.add_argument('--output', '-o', default='../saved_model/model.pkl',
                    type=output_file,
                    help='saving path for the trained model')

parser.add_argument('--random_seed',default=0,type=int,help='for KFold')

lgb_params = parser.add_argument_group('lgb_params')
lgb_params.add_argument('--l_leaves',default=62,type=int,help='lgb_num_leaves')
lgb_params.add_argument('--l_lr',default=0.03,type=float,help='lgb_learning_rate')
lgb_params.add_argument('--l_subsample_freq',default=50,type=int,help='lgb_subsample_freq')
lgb_params.add_argument('--l_samples',default=100,type=int,help='lgb_min_child_samples')
lgb_params.add_argument('--l_estimators',default=2000,type=int,help='lgb_n_estimators')
lgb_params.add_argument('--l_verbosity',default=0,type=int,help='lgb_verbosity')

xgb_params = parser.add_argument_group('xgb_params')
xgb_params.add_argument('--x_lr',default=0.06,type=float,help='xgb_learning_rate')
xgb_params.add_argument('--x_jobs',default=8,type=int,help='xgb_n_jobs')
xgb_params.add_argument('--x_estimators',default=3000,type=int,help='xgb_n_estimators')

cat_params = parser.add_argument_group('cat_params')
cat_params.add_argument('--c_lr',default=0.06,type=float,help='cat_learning_rate')
cat_params.add_argument('--c_l2',default=3,type=int,help='cat_l2_leaf_reg')
cat_params.add_argument('--c_estimators',default=3000,type=int,help='cat_n_estimators')
cat_params.add_argument('--c_slient',default=True,type=bool,help='cat_verbosity')

bys_params = parser.add_argument_group('bys_params')
bys_params.add_argument('--bys_lambda_1',default=1e1,type=float)

args = parser.parse_args()


lgb_params = {
        'num_leaves': args.l_leaves,
        'learning_rate': args.l_lr,
        'subsample_freq':args.l_subsample_freq,
        'min_child_samples':args.l_samples,
        'n_estimators': args.l_estimators,
        'verbosity':args.l_verbosity
        }
xgb_params = {
        'n_estimators':args.x_lr,
        'learning_rate': args.x_jobs,
        'n_jobs': args.x_estimators
        }
cat_params = {
        'n_estimators': args.c_estimators,
        'learning_rate': args.c_lr,
        'l2_leaf_reg':args.c_l2,
        'silent': args.c_slient
        }
bys_params = {
        'lambda_1':args.bys_lambda_1
        }

model_list = [[lgb.LGBMRegressor(**lgb_params),cat.CatBoostRegressor(**cat_params),xgb.XGBRegressor(**xgb_params)],[lgb.LGBMRegressor(**lgb_params),cat.CatBoostRegressor(**cat_params),xgb.XGBRegressor(**xgb_params)]]
regression_model = linear_model.BayesianRidge(**bys_params)
model = CpaStacking(model_set=model_list,final_model=regression_model)


kf = KFold(n_splits=5, shuffle=True, random_state=args.random_seed)
res_df = pd.DataFrame(columns = ['rmse', 'pearson', 'spearman','ci'],dtype = object)

data = args.train_data
label = args.train_label
fold_cnt = 0
for train_idx, test_idx in kf.split([i for i in range(data.shape[0])]):
    train_data, test_data, train_label, test_label = data[train_idx,:], data[test_idx,:], label[train_idx], label[test_idx]
    res = model.train(train_data,train_label,test_data,test_label)
    res_df.loc[fold_cnt] = res
    fold_cnt += 1

model.save(args.output)

print('the result of kfold')
print(res_df)
mean = res_df.mean(axis=0)
print("Mean rmse, pearson, spearman and ci is %s, %s, %s,%s" % (mean.iloc[0], mean.iloc[1], mean.iloc[2],mean.iloc[3])) 
        
