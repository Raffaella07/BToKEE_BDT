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
nores = pd.read_hdf(
    '../../LowPtElectrons/LowPtElectrons/macros/models/2020Apr_newPF_2Bkg/bdt_cmssw_mva_id/'
    '/bdt_cmssw_mva_id_testdata.hdf', key='data')
#test = test[np.invert(test.is_egamma)] 
#test = test[np.invert(abs(test.trk_eta)>=2.4)] 
#test = test[np.invert(test.trk_pt<0.5)] 
#test = test[np.invert(test.trk_pt>15.)] 
print nores.size

#res = pd.read_hdf(
  # '../../LowPtElectrons/LowPtElectrons/macros/models/2020Apr/bdt_cmssw_mva_id/'
  # '/bdt_cmssw_mva_id_testdata_nores.hdf', key='data')
#test = test[np.invert(test.is_egamma)] 
#test = test[np.invert(abs(test.trk_eta)>=2.4)] 
#test = test[np.invert(test.trk_pt<0.5)] 
#test = test[np.invert(test.trk_pt>15.)] 
#print res.size

# default variables
base = get_model(
    '../../LowPtElectrons/LowPtElectrons/macros/models/2020Apr_newPF_2Bkg/bdt_cmssw_mva_id/'
   '/test__cmssw_mva_id_BDT.pkl')

#base = xgb.Booster({'nthread': 4})  # init model
#base.load_model( 'xgb_FullModel_Otto.model')

#based_features, _ = base.feature_names()
based_features, _ = get_features('cmssw_mva_id')
#print based_features
nores['base_out'] = base.predict_proba(nores[based_features].as_matrix())[:,1]
nores['base_out'].loc[np.isnan(nores.base_out)] = -999 
base_roc_nores = roc_curve(
    nores.signal, nores.base_out
    )
base_auc_nores = roc_auc_score(nores.signal, nores.base_out)
print "nores ROC done"
print base_auc_nores

#res['base_out'] = base.predict_proba(res[based_features].as_matrix())[:,1]
#res['base_out'].loc[np.isnan(res.base_out)] = -999 
#base_roc_res = roc_curve(
 #   res.signal, res.base_out
 #   )
#base_auc_res = roc_auc_score(res.signal, res.base_out)
#print "res ROC done"
#print base_auc_res
# updated variables
#ecal = get_model(
#    '/eos/cms/store/user/crovelli/LowPtEle/ResultsFeb24/bdt_cmssw_mva_id_nnclean2/'
#    '/2020Feb24__cmssw_mva_id_nnclean2_BDT.pkl')
#ecal_features, _ = get_features('cmssw_mva_id_nnclean2')
#test['ecal_out'] = ecal.predict_proba(test[ecal_features].as_matrix())[:,1]
#test['ecal_out'].loc[np.isnan(test.ecal_out)] = -999 
#ecal_roc = roc_curve(
#    test.is_e, test.ecal_out
#)
#ecal_auc = roc_auc_score(test.signal, test.base_out)
#print ecal_auc

# training version in cmssw
#cmssw_roc = roc_curve(
#     test.is_e, test.ele_mva_value
#    )
#cmssw_auc = roc_auc_score(test.is_e, test.ele_mva_value)
#print cmssw_auc

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

#plt.plot(base_roc_res[0][:-1], base_roc_res[1][:-1], 
 #        linestyle='solid', 
  #       color='black', 
#         label='Aug22 variables (AUC: %.3f)' %base_auc)
   #      label=' resonant sample (AUC: %.3f)' %base_auc_res)
#         label='Extended set (AUC: %.3f)' %base_auc)

plt.plot(base_roc_nores[0][:-1], base_roc_nores[1][:-1], 
         linestyle='solid', 
        color='red', 
        label='non resonant sample (AUC: %.3f)' %base_auc_nores)

#plt.plot(cmssw_roc[0][:-1], cmssw_roc[1][:-1],
#         linestyle='dashed', 
#         color='red',
#         label='MVA ID cmssw (AUC: %.3f)' %cmssw_auc)

plt.xlabel('Mistag Rate')
plt.ylabel('Efficiency')
plt.legend(loc='best')
plt.xlim(0., 1)
plt.savefig('ROC_comparison.png')
plt.gca().set_xscale('log')
plt.xlim(1e-5, 1)
plt.savefig('ROC_comparison_log.png')
plt.clf()


print "Making plots2 ..."

# 1dim distribution
plt.title('BDT output')
basesignal_nores = nores.base_out.as_matrix()
#basesignal_res = res.base_out.as_matrix()
basesignal_nores = basesignal_nores[nores.signal==1]
#basesignal_res = basesignal_res[res.signal==1]
basebkg = nores.base_out.as_matrix()
basebkg = basebkg[nores.signal==0]
plt.hist(basesignal_nores, bins=70, color="orange", lw=0, label='non resonant signal ',normed=1,alpha=0.5)
#plt.hist(basesignal_res, bins=70, color="green", lw=0, label='resonant signal ',normed=1,alpha=0.5)
plt.hist(basebkg, bins=70, color="skyblue", lw=0, label='bkg',normed=1,alpha=0.5)
plt.show()
plt.legend(loc='best')
plt.savefig('OUTBase_comparison.png')
plt.clf()

# some working points
print ''
jmap = {}
with open('ROC.txt','w+') as f:
 for base_thr, ecal_thr, wpname in [
    (5, 5, 'T1'),
    (4.7 , 4.7, 'T2'),
    (4.5 , 4.5, 'T3'),
    (4.3 , 4.3, 'T4'),
    (4 , 4, 'T5'),
    (3.7 , 3.7, 'T6'),
    (3.5 , 3.5, 'T7'),
    (3.3 , 3.3, 'T8'),
    (3 , 3, 'T9'),
    (2.7 , 2.7, 'T10'),
    (2.5 , 2.5, 'T11'),
    (2.3 , 2.3, 'T12'),
    (2 , 2, 'T13'),
    (1.7 , 1.7, 'T14'),
    (1.5 , 1.5, 'T15'),
    (1.3 , 1.3, 'T16'),
    (1 , 1, 'T17'),
    (0.7 , 0.7, 'T18'),
    (0.5 , 0.5, 'T19'),
    (0.3 , 0.3, 'T20'),
    (0 , 0, 'T21'),
    (-0.3 , -0.3, 'T22'),
    (-0.5 , -0.5, 'T23'),
    (-0.7 , -0.7, 'T24'),
    (-1 , -1, 'T25'),
    (-1.3 , -1.3, 'T26'),
    (-1.5 , -1.5, 'T27'),
    (-1.7 , -1.7, 'T28'),
    (-2 , -2, 'T29'),
    (-2.3 , -2.3, 'T30'),
    (-2.5 , -2.5, 'T31'),
    (-2.7 , -2.7, 'T32'),
    (-3 , -3, 'T33'),
    (-3.3 , -3.3, 'T34'),
    (-3.5 , -3.5, 'T35'),
    (-3.7 , -3.7, 'T36'),
    (-4 , -4, 'T37'),
    (-4.3 , -4.3, 'T39'),
    (-4.5 , -4.5, 'T40'),
    (-4.7 , -4.7, 'T41'),
    (-5 , -5, 'T42'),
    (-5.3 , -5.3, 'T43'),
    (-5.5 , -5.5, 'T44'),
    (-5.7 , -5.7, 'T45'),
    (-6 , -6, 'T46'),
    (-6.3 , -6.3, 'T47'),
    (-6.5 , -6.5, 'T48'),
    (-6.7 , -6.7, 'T49'),
    (-7 , -7, 'T50'),
    (-7.3 , -7.3, 'T51'),
    (-7.5 , -7.5, 'T52'),
    (-7.7 , -7.7, 'T53'),
    (-8 , -8, 'T54'),
    (-8.5 , -8.5, 'T55'),
    (-9 , -9, 'T56'),
    (-9.5 , -9.5, 'T57'),
    (-10 , -10, 'T58'),
    (-10.5 , -10.5, 'T59'),
    (-11 , -11, 'T60'),
    (-11.5 , -11.5, 'T61'),
    (-12 , -12, 'T62'),
    (-12.5 , -12.5, 'T63'),
    (-13 , -13, 'T64'),
    (-13.5 , -13.5, 'T65'),
    (-14 , -14, 'T66'),
    (-14.5 , -14.5, 'T67'),
    (-15. , -15, 'T68'),
    ( 5.3 , 5.3, 'T69'),
    ( 5.5 , 5.5, 'T70'),
    ( 5.7 , 5.7, 'T71'),
    ( 6 , 6, 'T72'),
    ( 6.3 , 6.3, 'T73'),
    ( 6.5 , 6.5, 'T74'),
    ( 6.7 , 6.7, 'T75'),
    ( 7 , 7, 'T76'),
    ( 7.3 ,7.3, 'T77'),
    ( 7.5 ,7.5, 'T78'),
    ( 7.7 ,7.7, 'T79'),
    ( 8 , 8, 'T80'),
    ( 8.5 , 8.5, 'T81'),
    ( 9 , 9, 'T82'),
    ( 9.5 , 9.5, 'T83'),
    ( 10 , 10, 'T84'),
    ( 10.5 , 10.5, 'T85'),
    ]:
   print 'WP', wpname
   print 'base:',base_thr
   nores['base_pass'] = nores.base_out > base_thr
#   print 'ecal:'
#   test['ecal_pass'] = test.ecal_out > ecal_thr
    
   eff_base = ((nores.base_pass & nores.signal).sum()/float(nores.signal.sum()))
   mistag_base = ((nores.base_pass & np.invert(nores.signal)).sum()/float(np.invert(nores.signal).sum()))
   signal_base = (nores.base_pass & nores.signal).sum()
   bkg_base = (nores.base_pass & np.invert(nores.signal)).sum()
#   eff_ecal = ((test.ecal_pass & test.is_e).sum()/float(test.is_e.sum()))
#   mistag_ecal = ((test.ecal_pass & np.invert(test.is_e)).sum()/float(np.invert(test.is_e).sum()))
   fmerit = signal_base/np.sqrt(signal_base+ bkg_base)
   fmeritw = fmerit * eff_base
   jmap[wpname] = [mistag_base, eff_base]
   print 'eff (base): %.3f' % eff_base
   print 'mistag (base): %.6f' % mistag_base
   print "signal (base) : %.3f" % signal_base
   print "bkg (base) : %.3f" % bkg_base
   print "S/sqrt(S+B): %.3f"% fmerit
   print "S/sqrt(S+B)* eff(s): %.3f"% fmeritw
   f.write("%.3f %.7f %.7f\n"% (base_thr, eff_base,mistag_base))
   print "%.3f %.3f %.3f\n"% (base_thr, eff_base,mistag_base)
