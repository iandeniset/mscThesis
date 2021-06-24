import numpy as np
import matplotlib.pyplot as plt

def poisson(vP=None, vS=None, sigma=None):
    
    sigma = np.asarray(sigma)
    vP = np.asarray(vP)
    vS = np.asarray(vS)

    if sigma.all() == None:
        top = vP**2 - 2*(vS**2)
        btm = (vP**2)  - (vS**2)
        returnVal = 0.5*(top/btm) #calculate Poisson's Ratio
        #print('Poisson')

    if vP.all() == None:
        top = 0.5 - sigma
        btm = 1 - sigma
        returnVal = vS / np.sqrt(top/btm)  #calculate vP
        #print('Vp')

    if vS.all() == None:
        top = 0.5 - sigma
        btm = 1 - sigma
        returnVal = vP * np.sqrt(top/btm)  #calculate vS
        #print('Vs')

    return returnVal

def horSlowness(c, theta):
    'Take velocity of medium as well as incidence angle and return horizontal slowness'

    return (1/c)*np.sin(np.deg2rad(theta))

def vsSantamarina2001(sigma, beta, theta):
    '''
    Takes a value for the effective stress and two modifiers to calculate a theoretical shear velocity in the near surface
    Values of beta and theta determine if the value is for a soft NC clay, a loose or dry sand, or a stiff OC clay
    Sigma: mean effective stress
    Beta: exponent factor
    Theta: velocity factor
    '''

    return theta*(sigma**beta)


def lambLimits(h, v):
    '''
    Takes an upper layer thickness and shear velcoty
    Returns the approximate frequency limits where Lamb wave approximation is valid for a stiff upper layer
    '''

    vR = v * 0.92 #calculate theoretical Rayleigh wave velcoty given shear velocity

    if isinstance(h, list):
        h = np.asarray(h)
    else:
        h = [h]
        h = np.asarray(h)

    fig, ax = plt.subplots(1,1, figsize=(8,8))

    for thick in range(h.size):

        lowerF = vR/(6*h[thick])
        upperF = vR/(h[thick])

        waveLengthRange = np.arange((0.5*h[thick]), (10*h[thick]), 0.01) #calculate a cont range of wavelength values to calculate frequencies for
        fRange = vR/waveLengthRange


        ax.plot(fRange, waveLengthRange, c='k', alpha=0.75)
        ax.scatter([lowerF, upperF], [(6*h[thick]), (h[thick])], alpha=0.8, 
            label='Lower/Upper Freq: %.2f, %.2f for h of %0.1f' %(lowerF, upperF, h[thick]))
        #ax.scatter(upperF, (h[thick]), alpha=0.8, label='Upper Frequency Limit of %.2f for h of %0.1f' %(upperF, h[thick]))
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Rayleigh Wavelength [m]')

        ax.grid(which='both')
        ax.legend()
        ax.set_title('Upper and Lower Frequency Ranges for Lambs Approx; $V_s$ of %.1f' %v)

    plt.show()


############################
###LAMB WAVE APPROXIMATION
############################


def lambApprox(fMin, fMax, cMin, vp, vs, h, df=0.5, dc=0.5, plot=True):    
    
    #define inner functions
    def alpha(omega, vp, k):
        return np.lib.scimath.sqrt((((omega**2)/(vp**2)) - k**2))

    def beta(omega, vs, k):
        return np.lib.scimath.sqrt((((omega**2)/(vs**2)) - k**2))

    def lambLHS(omega, c, vs, vp, h):
        k = omega/c
        a = alpha(omega, vp, k)
        b = beta(omega, vs, k)
        
        top = np.tan(b*0.5*h)
        bttm = np.tan(a*0.5*h)
        
        return top/bttm

    def lambRHS(omega, c, vs, vp, h, modeType='A'):
        k = omega/c
        a = alpha(omega, vp, k)
        b = beta(omega, vs, k)
        
        top = (4*a*b*k**2)
        bttm = (k**2-b**2)**2
        
        if modeType == 'A': return -1*(top/bttm)**(-1)
        else: return -1*(top/bttm)**(1)
    
    #calculate frequency and velocity testing values
    freqs = np.arange(fMin, fMax, df)
    omeg = 2*np.pi*freqs
    cPhase = np.arange(cMin, vs, dc)
    
    #create array of nan's to hold solution
    fSol = np.full_like(cPhase, np.nan, dtype=np.double)
    
    #loop through testing phase velocities and compare LHS/RHS
    for i, c in enumerate(cPhase):
        k = omeg/c
    
        lamb_LHS = lambLHS(omeg, c, vs, vp, h)
        lamb_RHS = lambRHS(omeg, c, vs, vp, h, modeType='A')
    
        if c < (0.99*vs):
            idx = np.argwhere(np.diff(np.sign(lamb_RHS - lamb_LHS))).flatten()
            if (idx.size > 0):
                fSol[i] = freqs[idx][:1][0]
                
    #calculate frequency cutoff for approximation
    fApprox = vs/(6*h)
    
    #PLOT
    if plot:
        fig, ax = plt.subplots(1,1,figsize=(14,7))
        
        ax.plot(fSol, cPhase, label='Anti-Symmetric Lamb Wave Approximation')
        ax.axvline(x=fApprox, c='r', alpha=0.9, lw=0.75, label='Lamb Approximation Limit %.0f HZ' %fApprox)
        ax.set_xlim(0)
        ax.set_ylabel('Phase Velocity [m/s]')
        ax.set_xlabel('Frequency [Hz]')
        ax.grid(which='both')
        
        plt.legend()
        plt.show()
    
    return fSol, cPhase


############################
###Rayleigh Wave Velocity
############################

def rayleighVel(vP=None, vS=None, sigma=None):
    '''
    Takes either a set of Vp and Vs values or a singular Poisson Ratio value for a medium and returns the Rayleigh wave velocity

    Taken from:
    https://en.wikipedia.org/wiki/Rayleigh_wave
    '''

    sigma = np.asarray(sigma)
    vP = np.asarray(vP)
    vS = np.asarray(vS)

    if sigma.all() == None:
        sigma = poisson(vP=vP, vS=vS)

        top = 0.862 + 1.14 * sigma 
        btm = 1 + sigma 
        returnVal = (top / btm) * vS

    if (vP.all() == None) and (vS.all() == None):
        #note, in this case the function returns a ratio of Rayleigh wave vel over Vs
        top = 0.862 + 1.14 * sigma 
        btm = 1 + sigma
        returnVal = top / btm

    if (vP.all() == None) and (vS.all() != None) and (sigma.all() != None):
        top = 0.862 + 1.14 * sigma 
        btm = 1 + sigma 
        returnVal = (top / btm) * vS

    if (vP.all() != None) and (vS.all() == None) and (sigma.all() != None):
        vS = poisson(vP=vP, sigma=sigma)

        top = 0.862 + 1.14 * sigma 
        btm = 1 + sigma 
        returnVal = (top / btm) * vS

    rayleighVel = returnVal

    return rayleighVel


############################
###LAMB WAVE CUTOFF FREQUENCIES
############################


def lambCutoff(vP, vS, thick, n=1, plot=True):
    '''
    Takes in the P and S wave velocities of a plate of given thickness and returns 
    the theoretical cutoff frequencies
    
    thick is the plate thickness
    n is the number of cutoff values to calculate

    http://www.ase.uc.edu/~pnagy/ClassNotes/AEEM7028%20Ultrasonic%20NDE/AEEM-7028%20lecture,%20Part%204%20Rayleigh%20and%20Lamb%20Waves.pdf
    '''

    #SYMMETRIC CASES
    #shear waves
    fc_SS = n*(vS/(2*thick))

    #p waves
    fc_SP = (2*n -1)*(vP/(4*thick))

    #ANTI-SYMMETRIC CASES
    #shear waves
    fc_AS = (2*n -1)*(vS/(4*thick))

    #p waves
    fc_AP = n*(vP/(2*thick))

    #put solution in dict
    FC = {
    'Symmetric Shear' : fc_SS,
    'Symmetric Pressure' : fc_SP,
    'Anti-Symmetric Shear' : fc_AS,
    'Anti-Symmetric Pressure' : fc_AP,
    }

    if plot == False:
        print(FC)

    if plot:
        fig, ax = plt.subplots(1,1, figsize=(8,12))
        
        ax.axvline(x=fc_SS, label='Sym Shear', c='k')
        ax.axvline(x=fc_SP, label='Sym Pressure', c='b')
        ax.axvline(x=fc_AS, label='Anti Shear', c='r')
        ax.axvline(x=fc_AP, label='Anti Pressure', c='g')

        ax.set_xlabel('Frequency [Hz]')
        ax.set_xlim(0,150)
        ax.legend()
        ax.grid(which='both')

        plt.show()


    return FC



#################################
#LAYER RATIO FOR INVERIOSN
#################################

def layerRatioINV(lamMin, lamMax, ratio):
    '''
    Takes in (for now) a min and max wavelength for an experimental DC curve and outputs
    layer thickness bounds based on the provided layer ratio value

    Base off of Cox and Teague 2016 - 'Layering ratios: A systematic approach to the inversion of 
    surface wave data in the absence of a priori information'
    '''

    #first cal max resolvable depth
    maxDepth = lamMax * 0.5 

    #initiate depth vectors w/ first values
    dMin = np.array([lamMin/3])
    dMax = np.array([lamMin/2])

    #calculate and append remaining values while dMax is less than maxDepth
    i = 0
    while (np.nanmax(dMax) < maxDepth):
        
        if i == 0: #layer 1
            #initiate depth vectors w/ first values
            dMin = np.array([lamMin/3])
            dMax = np.array([lamMin/2])


        else:
            #calculate and append next dMin
            dMin = np.append(dMin, dMax[i-1])

            #calculate and append next dMax
            if i == 1: #second layer condition
                dMax = np.append(dMax, dMin[i] + ratio*(lamMin*0.5))
            else:
                dMax = np.append(dMax, dMin[i] + ratio*(dMax[i-1] - dMin[i-1]))

        i += 1


    return dMin, dMax


#################################
#DIFFRACTION CODE
#################################

def plotDiffraction(vel, offset, depth, Xt, xLim=None, c='k', lw=1, label='Theoretical Diffraction', ax=None):
    '''
    Plots predicted diffraction profile
    After Xia et al 2007: 'Feasibility of detecting near-surface feature with Rayleigh-wave diffraction'
    
    Input:
    vel - velocity of the encasing soil (note, this is a weighted average of all layers with preference to the overlying layers)
    offset - this is the offset of the anomaly from the source
    depth - this is the depth TO THE TOP of the anomaly
    Xt - this is the total spread length of the MASW survey (i.e source offset and receivers)
    '''
    
    #first create range for calculations along spread
    Xt_range = np.linspace(0,Xt,44)
    
    #then calculate distance from boulder to each side of survey line
    xx = offset-Xt_range
    
    #calculate arrive time of diffraction using formula from paper
    t_diff = (1/vel)*(offset+(xx**2+depth**2)**0.5)
    
    #modify for limits; NOTE: ORDER MATTERS HERE!
    if xLim != None:
        t_diff = t_diff[(Xt_range>xLim[0]) & (Xt_range<xLim[1])]
        Xt_range = Xt_range[(Xt_range>xLim[0]) & (Xt_range<xLim[1])]

    if ax != None:
        ax = ax
        msg = 'Master Plot'

    else:
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        msg = 'Master Plot'


    ax.plot(Xt_range, t_diff, c=c, alpha=0.9, label=label, lw=lw)
    
    #invert y
    if ax == None:
        ax.invert_yaxis()
    
    
    if ax == None:
        plt.show()

    return

def calcPhaseVel_diffraction(tx1, tx2, x1, x2):
    '''
    Takes two pairs of time and offset from source pairs and returns an estimated
    phase velocity using an observed diffraction pattern

    NOTE: time mist be in seconds
    '''

    print(np.absolute((x2-x1)))
    print(np.absolute((tx2-tx1)))

    return np.absolute((x2-x1))/np.absolute((tx2-tx1))

def calcDiffractorDepth(tApex, tx, x, d):
    '''
    Takes the time for a diffraction apex, time for a diffraction arrive at a FAR OFFSET, 
    distance between the source and anomaly, and distance between anomaly and trace used for time
    tApex: the time of the diffraction apex 
    tx: time of diffraction arrival on a trace in FAR OFFSET
    x: the distance from the anomaly and the trace used for 'tx'
    d: distance between the source and the anomaly
    '''

    timeRatio = tx/tApex
    print('Time Ratio: ',timeRatio)

    a = (timeRatio**(2)) - 1
    b = 2 * timeRatio*(timeRatio-1)*d
    c = ( timeRatio**(2)  - 2*timeRatio + 1)*(d**2)-(x**2)
    print('a: ',a)
    print('b: ',b)
    print('c: ',c)

    top = -1*b + ((b**2)-4*a*c)**(0.5)
    print('Top: ', top)
    bttm = 2*a
    print('Bottom: ', bttm)

    return top/bttm 