import numpy as np
import matplotlib
matplotlib.use('Agg')
from cmsjson import CMSJson
from pdb import set_trace
import os
from glob import glob
import pandas as pd
import json
from pprint import pprint
import matplotlib.pyplot as plt
from features import *
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.externals import joblib
import xgboost as xgb
from datasets import HistWeighter
#from xgbo.xgboost2tmva import convert_model
from itertools import cycle
from sklearn.metrics import roc_curve , roc_auc_score

def get_model(pkl):
    model = joblib.load(pkl)

    def _monkey_patch():
        return model._Booster

    if isinstance(model.booster, basestring):
        model.booster = _monkey_patch
    return model

# test dataset
test = pd.read_hdf(
    '../../LowPtElectrons/LowPtElectrons/macros/models/2020Apr/bdt_cmssw_mva_id/'
    '/bdt_cmssw_mva_id_testdata.hdf', key='data')
#test = test[np.invert(test.signal)] 
#test = test[np.invert(abs(test.trk_eta)>=2.4)] 
#test = test[np.invert(test.trk_pt<0.5)] 
#test = test[np.invert(test.trk_pt>15.)] 
print "test dataset done"
print test.size

# train dataset
train = pd.read_hdf(
    '../../LowPtElectrons/LowPtElectrons/macros/models/2020Apr/bdt_cmssw_mva_id/'
    '/bdt_cmssw_mva_id_traindata.hdf', key='data')
#train = train[np.invert(train.signal)] 
#train = train[np.invert(abs(train.trk_eta)>=2.4)] 
#train = train[np.invert(train.trk_pt<0.5)] 
#train = train[np.invert(train.trk_pt>15.)] 
print "train dataset done"
print train.size

# variables
base = get_model(
    '../../LowPtElectrons/LowPtElectrons/macros/models/2020Apr/bdt_cmssw_mva_id/'
    '/test__cmssw_mva_id_BDT.pkl')
based_features, _ = get_features('cmssw_mva_id')

# on test
test['base_out'] = base.predict_proba(test[based_features].as_matrix())[:,1]
test['base_out'].loc[np.isnan(test.base_out)] = -999 
test_base_roc = roc_curve(
    test.signal, test.base_out
    )
test_base_auc = roc_auc_score(test.signal, test.base_out)
print "test dataset ROC done"
print test_base_auc

# on train
train['base_out'] = base.predict_proba(train[based_features].as_matrix())[:,1]
train['base_out'].loc[np.isnan(train.base_out)] = -999 
train_base_roc = roc_curve(
    train.signal, train.base_out
    )
train_base_auc = roc_auc_score(train.signal, train.base_out)
print "train dataset ROC done"
print train_base_auc

# plots
print "Making plots ..."

# ROCs
plt.figure(figsize=[8, 12])
ax = plt.subplot(111)  
box = ax.get_position()   
ax.set_position([box.x0, box.y0, box.width, box.height*0.666]) 

plt.title('Trainings comparison')
plt.plot(
   np.arange(0,1,0.01),
   np.arange(0,1,0.01),
   'k--')

plt.plot(test_base_roc[0][:-1], test_base_roc[1][:-1], 
         linestyle='solid', 
         color='green', 
         label='Test dataset (AUC: %.3f)' %test_base_auc)

plt.plot(train_base_roc[0][:-1], train_base_roc[1][:-1], 
         linestyle='dashed', 
         color='red', 
         label='Train dataset (AUC: %.3f)' %train_base_auc)

plt.xlabel('Mistag Rate')
plt.ylabel('Efficiency')
plt.legend(loc='best')
plt.xlim(0., 1)
plt.gca().set_xscale('log')
plt.xlim(1e-4, 1)
plt.savefig('ROC_overtraining.png')
plt.clf()
