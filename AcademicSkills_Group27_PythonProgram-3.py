# -*- coding: utf-8 -*-
"""
NewsVendorGroup27.py

Purpose:
    Analysis of the newsvendor problem: calculating expected profits, 
    calculating the companies risk aversion, giving insightful plots, 
    data, and advice.
    
Version:
    2
    
Date: 
    22/06/2020
    
Author:
    Floris Six Dijkstra
    
"""

##################################################
###Importing packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

##################################################

#Set magics
sIn = 'C:/Users/flori/Downloads/nv1g50v2data.xlsx'
dP  =3.0
dC  = 1.0
vX  = [70, 77]

#Set parameters
iLB = 45
iUB = 140
vXi = np.arange (iLB, iUB)
dDiff = 0.05                #parameter, percentage of test parameter... 
dF = 3.0                    #...that determines the radius of the test inteval
ir = 1
dAlpha = 0.05
N = 'N'                     #for testing binomial dist.
Pi = 'Pi'

#Initialise
vX = np.array(vX)
df = pd.read_excel(sIn, usecols = ['G27V1', 'G27V2'])
mY = df.values

#################################################
### vProbB = ConstructBinom(vY)
def BinomParam(vY):
    """
    Purpose:
        Construct the parameters needed for a 
        binomial approximation of vY
        
    Inputs:
        vY      iY vector, demand
          
    Return value:
        iN      integer, parameter of binomial distribution
        dPi     double,  parameter of binomial distribution
        dMu     double, the average of the demand
    """
    dMu = np.mean(vY)
    dS2 = np.var(vY)
    iN = np.round(dMu**2/(dMu-dS2)).astype(int)
    dPi = dMu/iN
    print("The values of the parameters for the binomial are N=%g, Pi=%.3f" %(iN, dPi))
    return (iN, dPi, dMu)

#################################################
### mS= Sales(vX, vY)
def Sales(vX, vY):
    """
    Purpose:
        Find sales mS for supply vX and demand vY

    Inputs:
        vX      iX vector, supply
        vY      iY vector, demand

    Return value:
        mS      iX x iY matrix, sales
    """
    vX = np.asarray(vX).reshape(-1,1)
    vY = np.asarray(vY).reshape(1,-1)
    mS= np.minimum(vX,vY)
    return mS

##################################################
### Profit(vX, mS, dP, dC)
def Profit(vX, mS, dP, dC):
    """
    Purpose:
        Find profit for supply vX with sales mS

    Inputs:
        vX      iX vector, supply
        mS      iX x iY matrix, sales
        dP      double, sales price
        dC      double, cost

    Return value:
        mPr     iX x iY matrix, profit
    """
    
    vX = np.asarray(vX).reshape(-1,1)
    mPr = dP*mS - dC*vX
    return mPr

###########################################################
### (vEPr, vEPr2)= MomProfit(vX, dP, dC, iN, dPi, dMu, dF= 5.0)
def MomProfit(vX, dP, dC, iN, dPi, dMu, dF = 5.0):
    
    """
    Purpose:
        Find the first two moments of profit for supply vX, with demand
        coming from a Binomial approximation

    Inputs:
        vX      iX vector, supply
        dP      double, sales price
        dC      double, cost
        iN      integer, parameter of the binomial distrubution
        dPi     double,  parameter of the binomial distribution 
        dMu     double, average of the demand
        dF      (optional, default= 5.0) double, factor to use in deciding on range of y

    Return value:
        vEPr    iX vector, expected profit
        vEPr2   iX vector, expected squared profit
    """
    
    vYi    = np.arange(0,dF*dMu)
    vProbB = st.binom.pmf(vYi, iN, dPi)
    
    mS = Sales(vX, vYi)
    mPr = Profit(vX, mS, dP, dC)
    mPr2 = mPr**2
    vEPr = mPr@vProbB
    vEPr2 = mPr2@vProbB
    
    return (vEPr, vEPr2)

###########################################################
### vU = Utility(vX, dP, dC, iN, dPi, dMu, dGamma, dF= 5.0)
def Utility(vX, dP, dC, iN, dPi, dMu, dGamma, dF= 5.0):
    """
    Purpose:
        Find the utility for supply vX, with demand
        coming from a Binomial density 

    Inputs:
        vX      iX vector, supply
        dP      double, sales price
        dC      double, cost
        iN      integer, parameter of the binomial distrubution
        dPi     double,  parameter of the binomial distribution 
        dMu     double, average of the demand
        dGamma  double, risk aversion
        dF      (optional, default= 5.0) double, factor to use in deciding on range of y

    Return value:
        vU      iX vector, utility
    """
    (vEPr, vEPr2) = MomProfit(vX, dP, dC, iN, dPi, dMu, dF)
    vVarPr = vEPr2 - vEPr**2
    vU = vEPr - dGamma*vVarPr
    return vU

#############################################################
### (vUOpt, vXOpt) = OptimalUX(vEPr, vVarPr)   
def OptimalUX(vEPr, vVarPr):
    """
    Purpose:
        To find the maximal utility with corresponding profit 
        for a range of gamma's
    
    Inputs:
        vEPr    iX vector, the expected profit
        vVarPr  iX vector, the variance
        
    Return value:
        vUOpt   iG vector, containing the maximal 
                utility for a ragne of gamma's
        vXOpt   iG vector, containing corresponding supply
    """
    vGamma = np.arange(0,5,0.01)
    iG = vGamma.shape[0]
    vUOpt = np.zeros_like(vGamma)
    vXOpt = np.zeros_like(vGamma)
    for g in range (iG):
        dGammaLoop = vGamma[g]
        vU = vEPr - dGammaLoop*vVarPr
        i = np.argmax(vU)
        vUOpt[g] = vU[i]
        vXOpt[g] = vXi[i]
    return (vUOpt, vXOpt)
    
##################################################
### Describe(mY)
def Describe(mY):
    """
    Purpose:
        To use describtive analysis on the given data
        plotting a histogram an printing data
    
    Input: 
        mY    Y matrix, data of the demand for both villages
        
    Return values:
        None
    
    """
    vMu= np.mean(mY, axis=0)
    vS= np.std(mY, axis=0)
    vQuan= (.05, .25, .5, .75, .95)         #divide into quantiles
    mQ= np.quantile(mY, vQuan, axis = 0)
     
    mRes = np.vstack((vMu, vS, mQ))  
    asLab = ['mean', 'sdev' ] + ['q(%g)' % q for q in vQuan]
    asColumn = ['Village 1', 'Village 2']
    dfR = pd.DataFrame(mRes, index = asLab, columns = asColumn)
    print (dfR.to_string(float_format='%.2f'))
    plt.figure()
    plt.title('Histogram demand')
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.hist(mY, label = ['Village 1', 'Village 2'])
    plt.axvline(vMu[0], color = 'lightblue',     linestyle= '--', linewidth = 3)
    plt.axvline(vMu[1], color = 'moccasin',   linestyle= '--', linewidth = 3)
    plt.legend()
    plt.show()
   
#################################################
### dGamma = EstimateGamma(vXi, vEPr, vVarPr)
def EstimateGamma(iX, vEPr, vVarPr, dD = 0.01): 
    """
    Purpose:
        To estimate the gamma which the company prefers
        according to historical data
        
    Inputs:
        iX      integer, the supply in the preceding period
        vEPr    iX vector, the expected profit
        vVarPr  iX vector, the variance
        dD      double, constant of how many decimals
                decimals the gamma is
        
    Return values:
        dGamma  double, the gamma based on the 
                preceding actions of the company
    """
    vGamma = np.arange(0,5, dD)
    iG = vGamma.shape[0]
    vUOpt = np.zeros_like(vGamma)
    vXOpt = np.zeros_like(vGamma)
    
    for g in range (iG):
        dGammaLoop = vGamma[g]
        vU = vEPr - dGammaLoop*vVarPr
        i = np.argmax(vU)
        vUOpt[g] = vU[i]
        vXOpt[g] = vXi[i]
            
    plt.figure()
    plt.subplot(2,1,2)
    plt.xlabel('Gamma')
    plt.ylabel('Utility')
    plt.plot(vGamma, vUOpt)
    plt.subplot(2,1,1)
    plt.title('Change in Gamma')
    plt.ylabel('Supply')
    plt.plot(vGamma, vXOpt)
    vI= vXOpt == iX
    dGamma = vGamma[vI].mean()
    plt.axvline(dGamma, color = 'r', linestyle = '--')
    plt.axhline(iX, color = 'r', linestyle = '--')
    print( "Optimal supply x=%i is found for Gamma=%.2f"  %(iX ,dGamma))
    plt.show()
    return dGamma

######################################################
### (iAS, dAU) = MaxUtility(dGamma, vUOpt, vXOpt, dD = 0.01)
def MaxUtility(dGamma, vUOpt, vXOpt, dD = 0.01):
    """
    Purpose:
        Find the maximal utility with corresponding supply
        for a given gamma

   Inputs:
       dGamma   double, parameter for risk aversion
       vUOpt    iG vector, containing optimal utility for a
                range of gamma's
       vXOpt    iG vector, containing the corresponding
                supply for this maximal utility
        dD      double, constant of how many decimals
                decimals the gamma is
                
    Return values:
        iAS     integer, the adviced supply to reach max 
                utility
        dAU     double, the utility reached with this 
                supply
    """
    vI = np.arange(0,5,dD) == ((dGamma/dD).astype(int))*dD      #find the boolean matrix that indicates the 
    iAS = vXOpt[vI]                                             #given gamma, rounded to 2 decimals                    
    dAU = vUOpt[vI]
    return (iAS.astype(int), dAU)

######################################################
### (iMPS, dMP) = MaxProfit(vEPr)
def MaxProfit(vEPr):
    """
    Purpose:
        Find the maximal profit

    Inputs:
       vEPr    iX vector, the expected profit
                        
    Retrun values:
        iMPS    integer, the adviced supply to reach max 
                profit
        dMP     double, the profit reached with this 
                supply
    """
    i    = np.argmax(vEPr)
    dMP  = vEPr[i]
    iMPS = vXi[i]
    return (iMPS, dMP)

######################################################
### Robustness(dGamma, vY, iN, dPi, dMu)
def Robustness(dGamma, vY, iN, dPi, dMu):  
    """
    Purpose: 
        to test The robustness of our used estimators, 
        using MLL estimators and the MinMax graph and returns the
        1-aplha intervals
            
    Input:   
        dGamma   double, parameter for risk aversion
        vY      iY vector, demand
        iN      integer, parameter of the binomial distrubution
        dPi     double,  parameter of the binomial distribution 
        dMu     double, average of the demand
       
    Return value: 
        Return value:
        (vNCI[0],   2x2 matrix, doubles, border values of the 1-alpha interval
         vNCI[iLN], 
         vPiCI[0], 
         vPiCI[iLPi])
    """
    (vLRN, vN, dCritN)    = BinomLR(N,  vY, iN, dPi)
    (vLRPi, vPi, dCritPi) = BinomLR(Pi, vY, iN, dPi)
    return MinMax(vLRN, vN, dCritN, vLRPi, vPi, dCritPi, dMu, dGamma)

######################################################
### BinomLR(sNorPi, vY, iN, dPi)
def BinomLR(sNorPi, vY, iN, dPi):
    """
    Purpose: to find the Likelihood Ratio for the 
             binomial parameters
             
    Inputs: sNorPi  string, either N or Pi, determines which
                    parameter is tested
            vY      iY vector, demand
            iN      integer, parameter of the binomial distrubution
            dPi     double,  parameter of the binomial distribution 
    
    Return values:
            vLR     iNum vector, the likelihood ratio of
                    the entered parameter
            vP      iNum vector, the test space over which
                    the parameter is tested
            dCrit   double, the critical value for which
                    vLR is in the confidence interval
    """
    #Set test for The parameter up
    iNum = 100
    if(sNorPi==N):
         dP0=iN
    elif(sNorPi==Pi):
        dP0 = dPi
    vP = np.linspace((1-dDiff)*dP0,(1+dDiff)*dP0, num = iNum)
    vLLg = np.zeros(iNum)
    #Set figure
    plt.figure()
    plt.subplot(2,1,2)
    plt.ylabel('Likelihood')
    #Choose for n or pi
    if(sNorPi==N):
        plt.xlabel('N')
        for i in range(iNum):
            dN0 = vP[i]
            vLL = st.binom.logpmf(vY, dN0, dPi)
            vLLg[i] = vLL.sum()
    elif(sNorPi==Pi):
        plt.xlabel(r'$\pi$')
        for i in range(iNum):
            dPi0 = vP[i]
            vLL = st.binom.logpmf(vY, iN, dPi0)
            vLLg[i] = vLL.sum()
    else:
        print('ERROR: PARAMETER NOT PRESENT IN BINOM (change sNorPi to N or to Pi)')
        return
    
    plt.plot(vP, vLLg, label = 'Likelihood')
    plt.legend()
    i = np.argmax(vLLg)
    dLLMax = vLLg[i]
    dCrit = st.chi2.ppf(1-dAlpha, ir)
    #take max likelihood ratio
    vLR = -2*(vLLg - dLLMax)
    plt.subplot(2,1,1)
    plt.title('Maximum Likelihood Test')
    plt.ylabel('Likelihood ratio')
    plt.plot(vP, vLR, label = 'Likelihood ratio')
    plt.axhline(dCrit, color = 'red', label = 'Critical value')
    plt.legend()
    plt.show()
    return(vLR, vP, dCrit)

##########################################
### (vNCI[0], vNCI[iLN], vPiCI[0], vPiCI[iLPi]) = MinMax(vLRN, vN, dCritN, vLRPi, vPi, dCritPi, dMu, dGamma):
def MinMax(vLRN,  vN,  dCritN,
           vLRPi, vPi, dCritPi, dMu, dGamma):
    """
    Purpose: 
        to sketch a MinMax graph, that scetches 
        the "worst case" senario by testing al possible parameters
        en looking for the best scenario then
        
    Inputs:
        vLRN/vLRPi  vectors, containing the likelihood ratio
                    of N/Pi
        vN/vPi      vectors, containing the test space
                    of N/Pi
        dCritN/dCritPi doubles, the critical value for which
                    vLRN/vLRPi is in the confidence interval
        dMu         double, average of the demand, used for
                    determinding the interval for the utility
    Return value:
        (vNCI[0],   2x2 matrix, doubles, border values of the 1-alpha interval
         vNCI[iLN], 
         vPiCI[0], 
         vPiCI[iLPi])
    """
    #Initialise
    vI = vLRN <= dCritN
    vNCI = vN[vI] 
    vI = vLRPi <= dCritPi
    vPiCI = vPi[vI] 
    iLN = vNCI.shape[0]
    iLPi = vPiCI.shape[0]
    iXi = vXi.shape[0]
    mU = np.zeros((iXi, iLN, iLPi))
    #Set figure
    plt.figure()
    plt.title('U for all parameters in 1-Alhpa interval')
    plt.xlabel('Supply')
    plt.ylabel('Utility')
    
    for i in range(iLN):        #do not use for loop for larger datasets, then take all combinations of boundary cases
        for g in range(iLPi):
            vU = Utility(vXi, dP, dC, vNCI[i], vPiCI[g], dMu, dGamma)
            mU[:,i, g]=vU
            plt.plot(vXi, vU, color='grey')
    #Take minimum      
    vUMin = mU.min(axis=1).min(axis = 1)
    #Take and plot the max
    i = np.argmax(vUMin)
    plt.plot(vXi, vUMin, color = 'blue', label = 'Worst-Case')
    plt.axvline(vXi[i], color = 'red', label = 'Best Worst-Case')
    plt.legend()
    plt.show()
    #return the 95% confidence interval
    return ([vNCI[0], vNCI[iLN-1]], [vPiCI[0], vPiCI[iLPi-1]])
    
##########################################
### PlotGraphs(vEPr, vVarPr, vU, iAS, dAU)
def PlotGraphs(vEPr, vVarPr, vU, iAS, dAU):
     """
     Purpose:
         To sketch the graph of the Expected profit
         the Variance and the Utility, highlighting
         the supply associated with max Utility
         
     Inputs:
         vEPr    iX vector, the expected profit
         vVarPr  iX vector, the variance
         vU      iX vector, the utility
         iAS     integer, adviced supply for max Utility
         iAU     integer, associated Utility for iAS
         
     Return value:
         None   
     """
     #plot EPr, VarPr and U
     plt.figure()
     plt.subplot(1,5,1)
     plt.title('EPr')
     plt.xlabel('Supply')
     plt.plot(vXi, vEPr)
     plt.axvline(iAS, color = 'r', linestyle = '--')
     plt.subplot(1,5,3)
     plt.title('vVarPr')
     plt.xlabel('Supply')
     plt.plot(vXi, vVarPr)
     plt.axvline(iAS, color = 'r', linestyle = '--')
     plt.subplot(1,5,5)
     plt.title('vU')
     plt.xlabel('Supply')
     plt.plot(vXi, vU)
     plt.axvline(iAS, color = 'r', linestyle = '--')
     plt.axhline(dAU, color = 'b', linestyle = '--')
     plt.show()
     
     vStdPr = vVarPr**0.5
     
     #print graph of the expected profit and its spread
     plt.figure()
     plt.title('Profit')
     plt.xlabel('Supply')
     plt.ylabel('Euro')
     plt.plot(vXi, vEPr, color = 'blue', label = 'Expected Profit')
     plt.plot(vXi, vEPr+2*vStdPr, color = 'lightblue', label = r'EPr +/- 2$\sigma$')
     plt.plot(vXi, vEPr-2*vStdPr, color = 'lightblue')
     plt.legend()
     plt.show()

##########################################
###Jaque bera
def JarqueBera(vY):
    
    """
    Purpose:
        To test if the data, vY, is indeed
        normally distributed, by computing
        the Jarque Bera test paramete
        
    Input:
        vY  iY vector, data of the demand
        
    Output:
        None

    """
    
    #Initialise
    iN = vY.shape[0]
    dMu=vY.mean()
    dS2=vY.var()
    dMu3=0
    dMu4=0
    
    #Compute
    for i in range(iN):
        #d2 = (vY[i]-dMu)**2
        d3 = (vY[i]-dMu)**3
        d4 = (vY[i]-dMu)**4
        #dS2 +=d2
        dMu3+=d3
        dMu4+=d4
    dS = (dMu3/iN)/(dS2**1.5)
    dK = (dMu4/iN)/(dS2**2)
    dJB = (iN/6)*((dS**2)+0.25*((dK-3)**2))
    
    #Print
    print('\ndK = %.2f, dS = %.2f:'  %(dK, dS))
    print('The outcome of the JB test is: %.2f' % dJB)
    
#########################################################
### ninesevenfive(vEPr, vVarPr)
def ninesevenfive(vEPr, vVarPr):
    """
    Purpose:
        To compute the interval of which you know, there's a
        95.5% chance that your profit will be higher than this
        number
    Input:
        vEPr    iX vector, the expected profit
        vVarPr  iX vector, the variance
        
   Return value:
       none
    """
    vStdPR = vVarPr**0.5
    v975Pr = vEPr - 2*vStdPR
    i = np.argmax(v975Pr)
    g = vXi[i]
    print('Argmax of the 97.5 interval = %g' %g)
    print('when the minimum of the interval is %.2f with a expected profit of %.2f' %(v975Pr[i], vEPr[i]))
       
##########################################
### main()
def main():
    Describe(mY)
    iVillages = np.shape(vX)[0]
    
    for v in range(iVillages):
        print('\nThe following advice is for Village %g:' %(v+1))
        #Initialise
        vY = mY[:,v]
        iX = vX[v]
        #Construct parameters, E and Var
        (iN, dPi, dMu) = BinomParam(vY)
        (vEPr, vEPr2)  = MomProfit(vXi, dP, dC, iN, dPi, dMu) 
        vVarPr = vEPr2-vEPr**2
        #Estimate Gamma
        dGamma = EstimateGamma(iX, vEPr, vVarPr)
        #Find ompima for all Gamma
        (vUOpt, vXOpt) = OptimalUX(vEPr, vVarPr)
        #Advice for previous Gamma
        (iAS, dAU) = MaxUtility(dGamma, vUOpt, vXOpt)
        vU = vEPr-dGamma*vVarPr
        PlotGraphs(vEPr, vVarPr, vU, iAS, dAU)
        print('The adviced supply, to reach a maximal utility of: %.2f,\nis %g, using a gamma of %.2f, the corresponding profit is: %.2f, with a variance of %.2f' %(dAU, iAS, dGamma, vEPr[iAS], vVarPr[iAS-iLB]))
        (iMPS, dMP) = MaxProfit(vEPr)
        #Advice for maximal profit
        print('The maximal profit you could reach is attained at %g which reaches a profit of %.2f' %(iMPS, dMP))
        #1-alpha interval, and best worstcase
        mB = Robustness(dGamma, vY, iN, dPi, dMu)
        asLab = ['N', 'Pi']
        asColumns = ['Lower Bound', 'Upper Bound']
        dfB = pd.DataFrame(mB, index=asLab, columns=asColumns)
        print('\nThe 1-Alpha interval our estimators should be in:\n', dfB.to_string(float_format='%.3f'))
        #print the jarque bera test
        JarqueBera(vY)
        #look for the optimum in the 97.5% curve
        ninesevenfive(vEPr, vVarPr)
        
##########################################################
### start main
if __name__ == "__main__" : 
    main()