## to be used with
##
## python convert_pkl_to_xml.py models/2020Feb24/bdt_cmssw_mva_id_nnclean2/
## with the .pkl file in models/2020Feb24/bdt_cmssw_mva_id_nnclean2/
##
## At the end, the fixed.xml file has to be lint with
## xmllint --format filename.fixed.xml > filename.fixed.lint.xml
## this is the file to be used in RecoEgamma/ElectronIdentification

import os
from glob import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('model', default='../../LowPtElectrons/LowPtElectrons/macros/models/2020Apr/bdt_cmssw_mva_id')
args = parser.parse_args()

print args.model

pkls = glob('%s/*.pkl' % args.model)
print pkls
if len(pkls) != 1 : raise RuntimeError('There must be one and only one pkl in the directory')

model = pkls[0]
what = 'cmssw_mva_id'

#this should be outsorced
from features import *
features, additional = get_features(what)
print 'Features'
print features
print 'Additional'
print additional

print 'Loading model'
from sklearn.externals import joblib
import xgboost as xgb
xml = model.replace('.pkl', '.xml')
model = joblib.load(model)

from xgboost2tmva import convert_model
from itertools import cycle

# xgb sklearn API assigns default names to the variables, use that to dump the XML
# then convert them to the proper name
print 'XML conversion'
xgb_feats = ['f%d' % i for i in range(len(features))]
print xgb_feats

convert_model(model._Booster.get_dump(),xgb_feats, xml)

xml_str = open(xml).read()
for idx, feat in reversed(list(enumerate(features))):
    print idx 
    print feat
    xml_str = xml_str.replace("f%d" % idx, feat)

with open(xml.replace('.xml', '.fixed.xml'), 'w') as XML:
    XML.write(xml_str)
