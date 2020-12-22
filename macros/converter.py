
import numpy as np
import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
from cmsjson import CMSJson
from pdb import set_trace
import json
from features import *
import matplotlib.pyplot as plt
import ROOT
import uproot
import rootpy
import pandas as pd
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from datasets import tag, pre_process_data, target_dataset, get_models_dir, train_test_split
import os
from sklearn.externals import joblib
import xgboost as xgb



def get_model(pkl):
    model = joblib.load(pkl)

    def _monkey_patch():
        return model._Booster

    if isinstance(model.booster, basestring):
        model.booster = _monkey_patch
    return model

bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model( 'xgb_FullModel_Otto.model')
# load data
#base = get_model(
#    '../../LowPtElectrons/LowPtElectrons/macros/models/Otto_PFPF/'
 #   'Otto.model')
#based_features, _ = get_features('features')
#print based_features
bst.dump_model('xgb_FullVars_model.txt') #this command saves the model in .txt format, the model is readable trhough the FastForst application
