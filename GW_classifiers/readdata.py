#open root file with pandas
#source SCRIPTS/ENV/rootpy/bin/activate
#servono altri source non ancora chiaro perche
#source /home/waveburst/SOFT/GIT/cWB/library/cit_watenv.sh
#export LD_LIBRARY_PATH="./":$LD_LIBRARY_PATH
import io
import sys
import os.path
import ctypes
import ROOT
import numpy as np
import pickle
import pandas as pd
from ROOT import *
from ctypes import *
import string

def readfile(xfile, verbose):

   np.random.seed(seed=150914)

   if (xfile.find(".py")==-1):
     df = ROOT.RDataFrame("waveburst", xfile)   # read wave root file

   else:
     # load xfile list from python xfile
     execfile(xfile,globals())
     print('\nInput file list\n')
     for ifname in flist:
       print(ifname)
       if not os.path.isfile(ifname):
         print ("\nError: file ", ifname, " doesn't exist")
         exit(1)
     print('\n')
     df = ROOT.RDataFrame("waveburst", flist)   # read list wave root file
   
   bdf = df
   xdf=bdf.Define('rho0',       'rho[0]')
   xdf=xdf.Define('duration0',  'duration[0]')
   xdf=xdf.Define('bandwidth0', 'bandwidth[0]')
   xdf=xdf.Define('frequency0', 'frequency[0]')
   xdf=xdf.Define('netcc0',     'netcc[0]')
   xdf=xdf.Define('chirp1',     'chirp[1]')
   xdf=xdf.Define('time0',      'time[0]')
   xdf=xdf.Define('time1',      'time[1]')
   xdf=xdf.Define('dtL',        'abs(time[0]-time[2])')
   xdf=xdf.Define('dtH',        'abs(time[1]-time[3])')
   xdf=xdf.Define('netcc2',     'netcc[2]')
   xdf=xdf.Define('chirp3',     'chirp[3]')
   xdf=xdf.Define('chirp5',     'chirp[5]')
   xdf=xdf.Define('Qveto0',     'Qveto[0]')
   xdf=xdf.Define('Lveto1',     'Lveto[1]')
   xdf=xdf.Define('Lveto2',     'Lveto[2]')
   xdf=xdf.Define('size0',      'size[0]')
   xdf=xdf.Define('rho1',       'rho[1]')
   xdf=xdf.Define('Qveto1',     'Qveto[1]')
   xdf=xdf.Define('Qveto2',     'Qveto[2]')
   xdf=xdf.Define('Qveto3',     'Qveto[3]')
   xdf=xdf.Define('factor0',    'factor')
   xdf=xdf.Define('strain1',    'strain[1]')
   xdf=xdf.Define('era',        'erA[0]')   
   xdf=xdf.Define('snr0',       'snr[0]')
   xdf=xdf.Define('snr1',       'snr[1]')
   xdf=xdf.Define('snr2',       'snr[2]') 
   xdf=xdf.Define('time2',      'time[2]')
   xdf=xdf.Define('time3',      'time[3]')
   xdf=xdf.Define('theta1',     'theta[1]')
   xdf=xdf.Define('phi1',       'phi[1]')
   xdf=xdf.Define('theta0',     'theta[0]')
   xdf=xdf.Define('gps0',     'gps[0]')
   xdf=xdf.Define('gps1',     'gps[1]')
   xdf=xdf.Define('noise0',     'noise[0]')
   xdf=xdf.Define('noise1',     'noise[1]')
   xdf=xdf.Define('phi0',       'phi[0]')
   xdf=xdf.Define('ecor0',       'ecor')
   xdf=xdf.Define('ecor1',       'ECOR')
   xdf=xdf.Define('lag0',       'lag[0]')
   xdf=xdf.Define('lag1',       'lag[1]')
   xdf=xdf.Define('penalty0',    'penalty')
   #xdf=xdf.Define('ae0',         'ae')

    
   for n in range(0,2): xdf=xdf.Define('sSNR'+str(n), 'sSNR['+str(n)+']')
   
   xcolNames = xdf.GetColumnNames()
   if(verbose): print("xgb Colnames:\n{}\n".format(xcolNames))
   #print(xdf.GetEntry(0))
   xvars = [
            'rho0','rho1',
            'norm', 'duration0', 'bandwidth0', 'frequency0','noise0',
            'netcc0','netcc2','time1',
            'chirp1', 'chirp3', 'chirp5','snr0','snr1','snr2',
            'time0','time2','time3', 'dtL', 'dtH',
            'Qveto0', 'Qveto1', 'Qveto2', 'Qveto3','gps0','gps1',
            'Lveto1', 'Lveto2','era', 
            'penalty0', 'ecor0','ecor1','size0', 'likelihood','factor0','strain1',  'phi1', 'theta0', 'phi0', 'theta1','lag0','lag1','sSNR0','sSNR1','noise1'#,'ae0'
           ]
   bifar=False
   for var in xcolNames:
        if(var=='ifar'):  bifar=True
   if(bifar==True):  xvars.append('ifar')

   xnp = xdf.AsNumpy(columns=xvars)
   xnd = np.vstack([xnp[var] for var in xvars]).T
   xpd = pd.DataFrame(data=xnd, columns=xvars)
  
   return  xpd

