import wfdb
import numpy as np
import scipy
import scipy.signal



def load_rec(rec_id):
  """
  Create wfdb record and annotation objects.
  
  Parameters
  ----------
  rec_id : int
    Record Id of the patient
  
  Returns
  -------
  ppg_record : wfdb record object
  """
  rec_ids = np.arange(1,54)
  if rec_id not in rec_ids:
    raise Exception("Enter a valid record number from 1 to 53") 
  elif (rec_id<10):
    rec_name = 'bidmc0' + str(rec_id)
  else:
    rec_name = 'bidmc' + str(rec_id)
  record = wfdb.rdrecord(rec_name, pn_dir = 'bidmc')
  chans = record.sig_name
  chan = chans.index('PLETH,')
  ppg_record = wfdb.rdrecord(rec_name, pn_dir = 'bidmc', channels=[chan])
  return ppg_record



def bpm(rec_id):
  """
  Heart rate is calculated by counting the number of systolic peaks per minute. 
  Systolic peaks and peak-to-peak intervals of a PPG signal are found

  """
  ppg_rec = load_rec(rec_id)
  p_sig = ppg_rec.p_signal[:,0]

  # median is not a good estimator here
  max_h = np.max(p_sig)
  min_h = np.min((ppg_rec.p_signal)[:,0])
  h = (max_h + min_h)/2

  peak_locs, properties = scipy.signal.find_peaks(p_sig, height = h)  # systole locations

  f_samp = ppg_rec.fs
  loc_sec = np.divide(peak_locs,f_samp)
  avg_rr_int = (loc_sec[-1])/loc_sec.shape[0]
  bpm_ = 60/avg_rr_int
  return bpm_