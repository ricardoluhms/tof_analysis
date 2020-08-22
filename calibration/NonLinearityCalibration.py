'''The following program computes the non-linearity correction coefficients using csv as the input'''
'''Example: python NonLinearityCalibration.py --file filename --phaseCorr 0 --modFreq 48 '''

 # 
 #
 # Copyright (C) {2017} Texas Instruments Incorporated - http://www.ti.com/ 
 # 
 # 
 #  Redistribution and use in source and binary forms, with or without 
 #  modification, are permitted provided that the following conditions 
 #  are met:
 #
 #    Redistributions of source code must retain the above copyright 
 #    notice, this list of conditions and the following disclaimer.
 #
 #    Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the 
 #    documentation and/or other materials provided with the   
 #    distribution.
 #
 #    Neither the name of Texas Instruments Incorporated nor the names of
 #    its contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 #  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 #  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 #  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
 #  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 #  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
 #  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 #  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 #  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
 #  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 #  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
#

import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import argparse
import pandas as pd


def csv_test(modFreq,modFreq2=0,max_phase_resol=100,csv_name="/tof/calibration/non_lin2.csv",show=True):
    
    c=299792458
    reach=c/(2*modFreq*(10**6))
    phase_range=np.arange(0,4095,max_phase_resol).astype("int")
    dist=np.round(phase_range*reach/4096,4)

    if modFreq2!=0:
        reach2=c/(2*modFreq2*(10**6))
        gcd_freq=np.gcd(modFreq,modFreq2)
        max_reach=c/(2*gcd_freq*(10**6))
        measurement_resol=max_reach/4.096
        phase_range2=np.round(dist*4096/reach2,1).astype("int")
        calib_arr=np.vstack((np.vstack((dist,phase_range)),phase_range2)).T

    else:
        calib_arr=np.vstack((dist,phase_range)).T
    if show:
        print("Modulation Freq= ",modFreq,"[MHz]","  Freq1 Reach= ",round(reach,4),"[m]")
        if modFreq2!=0:
            print("Modulation Freq2= ",modFreq2,"[MHz]","  Freq2 Reach= ",round(reach2,4),"[m]")
            print("Dealising Freq= ",gcd_freq,"[MHz]","  Dealising Reach= ",round(max_reach,4),"[m]")
            print("Dealising Phase Resolution = ",round(measurement_resol,2),"[mm]")
    pd.DataFrame(calib_arr).to_csv(csv_name)

def NonLinearityCalibration(filename, modFreq, modFreq2 = 0, phaseCorr = 0, phaseCorr2 = 0, chipset = 'TintinCDKCamera', period = 0):
    """This function computes non linearity coefficients. 
    
    **Parameters**::
        -fileName: CSV file containing data of measured distance and captured phase
        -phaseCorr: The phase offset value for the given modulation frequency
        -modFreq: The modulation frequency at which data is captured. 
        -chipset: The chipset used. Default is TintinCDKCamera
        -period: The period for non linearity - this calue can be 0, 1 or 2 in case of tintin. For calculus, it's 0 or 1
    **Optional Parameters, when dealiasing is available **::
        -modFreq2: Modulation frequency 2
        -phaseCorr2: Phase Corr 2 
        
    **Returns**::
        -Non linearity coefficients for the given profile
        """
    c = 299792458
    if not os.path.exists(filename):
        print("wrong path or ", filename," does not exist")
        return
    
    df=pd.read_csv(filename,index_col=[0])
    if df.shape[1]==2:
        df=df.rename(columns={0: "distance", 1: "phase1"})

    elif df.shape[1]==3:
        df=df.rename(columns={"0": "distance", "1": "phase1", "2":"phase2"})
        phases2 = df["phase2"].astype("float").values
    distances = df["distance"].astype("float").values
    phases = df["phase1"].astype("float").values
    

    distancesToPhase = distances*modFreq*4096*2e6/(c)
    distancesToPhase = distancesToPhase[distances.argsort()]
    measuredPhase = phases[distances.argsort()]
    #from IPython import embed; embed()
    
    if chipset == 'TintinCDKCamera':
        y = getCoefficients(measuredPhase, distancesToPhase, phaseCorr, period)
        if df.shape[1]==3:
            distancesToPhase2 = distances*modFreq2*4096*2e6/c
            distancesToPhase2 = distancesToPhase2[distances.argsort()]
            measuredPhase2 = phases2[distances.argsort()]
            y1 = getCoefficients(measuredPhase2, distancesToPhase2, phaseCorr2, period)
        else:
            y1 = None

        return True, y, y1

    ##: Add calculus non-linearity correction
    return False

def getCoefficients(measuredPhase, distancesToPhase, phaseCorr, period):
    if not phaseCorr:
        phaseCorr1 = (-distancesToPhase[int(round(np.size(measuredPhase)/2))]  
                      +measuredPhase[int(round(np.size(measuredPhase)/2))])
    else:
        phaseCorr1 = 0              
    expectedPhase = distancesToPhase + phaseCorr + phaseCorr1
    expectedPhase = expectedPhase%4096    
    measuredPhase = (measuredPhase + phaseCorr)%4096
    indices = measuredPhase.argsort()
    measuredPhase = measuredPhase[indices]
    expectedPhase = expectedPhase[indices]
    expectedPhase = checkBoundaryConditions(measuredPhase, expectedPhase, period)
    offsetPoints = np.arange(0., 4096./2**(2-period), 256./2**(2-period))
    y = np.around(np.interp(offsetPoints, measuredPhase, expectedPhase))
    indexes = []
    for val in np.arange(len(y)-1):
        indexes.append(y[val] == y[val+1])
    for val in np.arange(len(indexes)-1):
        if indexes[val] == True and indexes[val+1] == False:
            y[0:val+1] = offsetPoints[0:val+1]
        if indexes[val] == False and indexes[val+1] == True:
            y[val+1:] = offsetPoints[val+1:]
            y = y.astype(int)          
    return y

def checkBoundaryConditions(measuredPhase, expectedPhase, period):
    '''
    Function to check boundary conditions after sorting expected phases and measured phases based on the measured phase. 
    Two boundary conditions are possible: measured phase has wrapped around, while expected phase has not. This will happen 
    in the beginning. Or expected phase has wrapped around, while measured phase has not. 
    This will happen at the end of the array.
    '''

    #FIXME: Assuming that the wraparound happens only for the first and the last elements 
    # of the array. This seems okay, considering the fact that common phase 
    # offsets are subtracted from both the expected phases as well as the measured phases

    if np.abs(measuredPhase[0] - expectedPhase[0]) > 3000: #difference in phase is 3000 - very conservative
        expectedPhase[0] = expectedPhase[0] - 4096/2**(2-period)

    if np.abs(measuredPhase[-1] - expectedPhase[-1]) > 3000:
        expectedPhase[-1] = expectedPhase[-1] + 4096/2**(2-period)

    return expectedPhase
            
def parseArgs (args = None):
    
    parser = argparse.ArgumentParser(description='Calculate Common Phase Offsets')
    #parser.add_argument('-f', '--file', help = 'CSV file', required = True, default= None)
    #parser.add_argument('-m', '--modFreq', type = float, help = 'Modulation Frequency', required = True, default= 10)
    #parser.add_argument('-n', '--modFreq2', type = float, help = 'Modulation Frequency 2', required = False, default = 0)
    #parser.add_argument('-p', '--phaseCorr', help = 'Phase Corr Value', type = int,  required = True, default = 0)     
    #parser.add_argument('-q', '--phaseCorr2', help = 'Phase Corr 2 Value', type = int, required = False, default = 0)
    parser.add_argument('-c', '--chipset', help = 'Camera Type', required = False, default = 'TintinCDKCamera')
    parser.add_argument('-e', '--period', help = 'Phase Linearity Period (0, 1 or 2)', required = False, default = 0, type = int)
    return parser.parse_args(args)           

if __name__ == '__main__':

    csv_name="/tof/calibration/non_lin.csv"
    modFreq=40
    modFreq2=60
    phaseCorr=730
    phaseCorr2=844
    val = parseArgs(sys.argv[1:])

    csv_test(modFreq=modFreq,modFreq2=modFreq2,csv_name=csv_name)

    # ret = NonLinearityCalibration(filename= val.file, phaseCorr = val.phaseCorr, 
    #                               modFreq = val.modFreq, modFreq2 = val.modFreq2, 
    #                               phaseCorr2 = val.phaseCorr2, chipset = val.chipset, period = val.period)

    ret = NonLinearityCalibration(filename= csv_name, phaseCorr = 0, 
                                modFreq = modFreq, modFreq2 = modFreq2, 
                                phaseCorr2 = phaseCorr2, chipset = val.chipset, period = val.period)
    if not ret:
        print("Can't get the nonlinearity coefficients")
        sys.exit()
    else:
        boo, y, y1 = ret
        with open ("non_lin_phase.txt", 'w') as f:
            for num,y_val in enumerate(y):
                f.write("phase_lin_coeff0_"+str(num)+"= "+str(int(y_val))+"\n")
        data = ''
        data1 = ''
        for c in y:
            if c < 0:
                c += 4096 # Negative values cannot be programmed. So, they should be wrapped around. 
            data += str(c) + ' '
    
    print ("phase_lin_coeff0 = " + data)
    if y1 is not None:
        with open ("non_lin_phase.txt", 'a') as f:
            for num,y_val in enumerate(y):
                f.write("phase_lin_coeff1_"+str(num)+"= "+str(int(y_val))+"\n")
        for c1 in y1:
            if c1 < 0:
                c1 += 4096
            data1 += str(c1) + ' '
        print ("phase_lin_coeff1 = " + data1)
    print ("phase_lin_corr_period = {}".format(val.period))
    with open ("non_lin_phase.txt", 'a') as f:
        f.write("phase_lin_corr_period= "+str(val.period)+"\n")
        f.write("phase_lin_corr_en= "+str(1)+"\n")
    #from IPython import embed; embed()

