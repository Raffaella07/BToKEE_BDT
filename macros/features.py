basic_features = [
   'BToKEE_l1_ptN',
   'BToKEE_l2_ptN',
   'BToKEE_k_ptN',
   'BToKEE_ptN',
   'BToKEE_iso04',
  'BToKEE_l1_iso04',
  'BToKEE_l2_iso04',
  'BToKEE_k_iso04',
  'BToKEE_l1_pfmvaId',
   'BToKEE_l2_pfmvaId',
 'BToKEE_l1_mvaId',
  'BToKEE_l2_mvaId',
   'BToKEE_cos2D',
  'BToKEE_DeltaZ',
  'BToKEE_ptImb',
    'BToKEE_svprob',
   'BToKEE_l_xy_sig',
  'BToKEE_l1_dxySig',
  'BToKEE_l2_dxySig',
  'BToKEE_k_dxySig',
   'BToKEE_maxDR',
    'BToKEE_minDR',
   # 'trk_outp',
   ]

event_additional = [
 #'BToKEE_eta',
#  'trk_eta',
#  'trk_phi',
#  'trk_p',
#  # 'trk_charge',
#  'trk_nhits',
#  'trk_high_purity',
#  # 'trk_inp',
#  # 'trk_outp',
#  'trk_chi2red',
  ]

labeling = [ 'signal']


def get_features(ftype):
   add_ons = []
   if ftype.startswith('displaced_'):
      add_ons = ['trk_dxy_sig']
      ftype = ftype.replace('displaced_', '')
   if ftype == 'seeding':
      features = seed_features
      additional = seed_additional
   elif ftype == 'trkonly':
      features = trk_features
      additional = seed_additional
   elif ftype == 'betterseeding':
      features = seed_features+['rho',]
      additional = seed_additional
   elif ftype == 'fullseeding':
      features = fullseed_features
      additional = seed_additional
   elif ftype == 'improvedseeding':
      features = improved_seed_features
      additional = seed_additional
   elif ftype == 'improvedfullseeding':
      features = improved_fullseed_features
      additional = seed_additional
   elif ftype == 'id':
      features = id_features
      additional = id_additional
   elif ftype == 'mva_id':
      features = mva_id_inputs
      additional = id_additional
   elif ftype == 'combined_id':
      features = list(set(mva_id_inputs+id_features))#-to_drop-useless)
      additional = id_additional
   elif ftype == 'cmssw_displaced_improvedfullseeding':
      features = cmssw_displaced_improvedfullseeding
      additional = seed_additional
   elif ftype == 'cmssw_improvedfullseeding':
      features = cmssw_improvedfullseeding
      additional = seed_additional
   elif ftype == 'cmssw_mva_id':
      features = basic_features
      additional = event_additional# + ['preid_bdtout1','preid_bdtout2',
                                    #'gsf_bdtout1','gsf_bdtout2', ] # 'has_pfele','has_pfgsf',
   elif ftype == 'cmssw_mva_id_extended':
      features = cmssw_mva_id + ['preid_bdtout1']
      additional = id_additional + ['preid_bdtout1','preid_bdtout2'] # 'has_pfele','has_pfgsf',
   elif ftype == 'cmssw_displaced_improvedfullseeding_fixSIP':
      features = cmssw_displaced_improvedfullseeding_fixSIP
      additional = seed_additional
   elif ftype == 'cmssw_displaced_improvedfullseeding_fixInvSIP':
      features = cmssw_displaced_improvedfullseeding_fixInvSIP
      additional = seed_additional
   elif ftype == 'basic_plots_default':
      features = cmssw_mva_id \
          + cmssw_displaced_improvedfullseeding \
          + ['trk_dxy_sig', 'trk_dxy_sig_inverted'] \
          + ['sc_Nclus', 'ele_eta', 'gsf_eta']
      additional = seed_additional
   else:
      raise ValueError('%s is not among the possible feature collection' % ftype)
   return features+add_ons, additional
