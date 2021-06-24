####################################
###CODE FOR MASW DATA
####################################

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import os
import sys
import obspy as op
from obspy import read

import warnings
warnings.filterwarnings('ignore')


def loadGatherProfile(path, fileNameMod=None):    
    ########
    ### Re-name files in given dir to ensure they are read in order
    ########
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.dat'):
                if len(f.replace('.dat','')) < 2:
                    os.rename(os.path.join(path, f), os.path.join(path, '0' + f))

    ########
    ### Re-name files in given dir with specified name modifier (i.e. what site and date and line)
    ########
    if fileNameMod != None:
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.dat'):
                    os.rename(os.path.join(path, f), os.path.join(path, fileNameMod + f.replace('.dat','')[-2:]) + '.dat')

    
    ########
    ### Load all shot gathers into a dictionary
    ########
    '''
    Note, all files MUST HAVE two digit shot number at the end of their file name; i.e. '01.dat' or 'shotlocationanddate_01.dat'
    This is due to the way the shots are stored in the returned dictionary and the functionality of being able to rename all files at once
    '''


    gatherStreams = {}
    gatherTraces = {}

    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.dat'):

                #print(f)

                #store all gathers into dictionary of stream objects
                gatherStreams['Shot {0}'.format(f.replace('.dat','')[-2:])] = read(os.path.join(path, f), 'seg2')
                
                #get current dictionary key
                stream = 'Shot {0}'.format(f.replace('.dat','')[-2:])
                
                #convert and store all shot gather streams into raw trace data
                gatherTraces['Shot {0}'.format(f.replace('.dat','')[-2:])] = np.stack(t.data for t in gatherStreams[stream].traces)

    #print out stats for user
    print('There are a total of {0}'.format(len(gatherStreams)), 'shot gathers in the specififed directory \n')

    print('Gathers have been returned in dictionary format with the following keys:')
    for key in gatherStreams:
        print(key)

    return gatherStreams, gatherTraces


def loadShotGather(file):
    """Load and return shot gather data along with original Obspy stream"""
    
    from obspy import read
    import numpy as np

    #load in raw seg2 file to obspy stream
    strm = read(file, 'seg2')
    
    #generator function here to save memory for big files; basically list comprehension 
    trcs = np.stack(t.data for t in strm.traces)
    
    return trcs, strm


def gatherParameters(stream):
    '''Take in Obspy stream object and extract parameters to dictionary'''
    
    #print(stream[3].stats)
    
    numTrc = len(stream)
    numSamps = stream[0].stats.npts
    dt = float(stream[0].stats.delta)
    R1 = float(stream[0].stats.seg2['RECEIVER_LOCATION']) #first receiver location
    dx = float(stream[1].stats.seg2['RECEIVER_LOCATION'])-float(stream[0].stats.seg2['RECEIVER_LOCATION'])
    sLoc = float(stream[0].stats.seg2['SOURCE_LOCATION']) #source location
    #X1 = abs(float(stream[0].stats.seg2['SOURCE_LOCATION']))
    X1 = abs(sLoc) + R1
    date = stream[0].stats.seg2.ACQUISITION_DATE
    
    return dict(numTrc=numTrc, numSamps=numSamps, dt=dt, dx=dx, X1=X1, date=date, R1=R1, sLoc=sLoc)

######################ORIGINAL
def plotShotGather_OLD(data, startTime, endTime, percentile=99.99, gain=1, traceNorm=None, title='Shot Record', refract=None):
    '''Takes either a shot gather file or raw stream file and plots the data'''
    
    if isinstance(data, str):
        #load in file to get raw data matrix and stream then extract parameters
        gather, gatherStream = loadShotGather(data)
        gatherParams = gatherParameters(gatherStream)
        
        #plotting parameters
        perc = np.percentile(gather, percentile)
        gain = gain
        nTrace = gatherParams['numTrc']
        dx = gatherParams['dx']
        npts = gatherParams['numSamps']
        dt = gatherParams['dt']
        srcOffset = gatherParams['X1']
        start = int(startTime/dt)    #start time
        end = int(endTime/dt)   #end time
        
        xOffset = np.arange(0,(nTrace)*dx,dx)
        t = np.arange(0,npts*dt,dt)


        #set tick parameters
        majorLocator = MultipleLocator(0.1)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator = MultipleLocator(0.01)

        
        #plot the data
        fig, ax = plt.subplots(1,1, figsize=(15,15))

        trNum = 1

        for xpos, tr in zip(xOffset, gather):
            #optional normalization
            if traceNorm == 'Per Trace': amp = gain * tr / np.nanmax(tr) + (xpos+srcOffset)
            else: amp = gain * tr / perc + (xpos+srcOffset)

            ax.plot(amp[start:end], t[start:end], c='k', lw=0.5)
            ax.fill_betweenx(t[start:end], amp[start:end], 
                             (xpos+srcOffset), amp[start:end] > (xpos+srcOffset), color='r', lw=0, alpha=0.7)
            ax.fill_betweenx(t[start:end], amp[start:end], 
                             (xpos+srcOffset), amp[start:end] < (xpos+srcOffset), color='blue', lw=0, alpha=0.7)
            ax.text(xpos+srcOffset, 0, str(trNum), horizontalalignment='center')

            trNum += 1


        if refract:
            ax.yaxis.set_major_locator(majorLocator)
            ax.yaxis.set_minor_locator(minorLocator)
            ax.set_xlim(0,np.nanmax(xOffset)+srcOffset*1.5)
            ax.grid(axis='y', which='both')
            ax.set_axisbelow(True)

        ax.set_ylabel('Time [s]')
        ax.set_xlabel('Offset from Source [m]')
        ax.set_title(title)
        ax.invert_yaxis()
        plt.show()
        
    else:
        #extract traces from stream and get acquisition parameters
        gather = np.stack(t.data for t in data.traces)
        gatherParams = gatherParameters(data)
        
        #plotting parameters
        perc = np.percentile(gather, percentile)
        gain = gain
        nTrace = gatherParams['numTrc']
        dx = gatherParams['dx']
        npts = gatherParams['numSamps']
        dt = gatherParams['dt']
        srcOffset = gatherParams['X1']
        start = int(startTime/dt)    #start time
        end = int(endTime/dt)   #end time
        
        xOffset = np.arange(0,(nTrace)*dx,dx)
        t = np.arange(0,npts*dt,dt)

        #set tick parameters
        majorLocator = MultipleLocator(0.1)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator = MultipleLocator(0.01)
        
        #plot the data
        fig, ax = plt.subplots(1,1, figsize=(15,15))

        trNum = 1

        for xpos, tr in zip(xOffset, gather):
            #optional normalization
            if traceNorm == 'Per Trace': amp = gain * tr / np.nanmax(tr) + (xpos+srcOffset)
            else: amp = gain * tr / perc + (xpos+srcOffset)

            ax.plot(amp[start:end], t[start:end], c='k', lw=0.5)
            ax.fill_betweenx(t[start:end], amp[start:end], 
                             (xpos+srcOffset), amp[start:end] > (xpos+srcOffset), color='r', lw=0, alpha=0.7)
            ax.fill_betweenx(t[start:end], amp[start:end], 
                             (xpos+srcOffset), amp[start:end] < (xpos+srcOffset), color='blue', lw=0, alpha=0.7)
            ax.text(xpos+srcOffset, 0, str(trNum), horizontalalignment='center')

            trNum += 1

        if refract:
            ax.yaxis.set_major_locator(majorLocator)
            ax.yaxis.set_minor_locator(minorLocator)
            ax.set_xlim(0,np.nanmax(xOffset)+srcOffset*1.5)
            ax.grid(axis='y', which='both')
            ax.set_axisbelow(True)


        ax.set_ylabel('Time [s]')
        ax.set_xlabel('Offset from Source [m]')
        ax.set_title(title)
        ax.invert_yaxis()
        plt.show()


def plotShotGather(data, startTime, endTime, percentile=99.99, gain=1, traceNorm=None, title='Shot Record', refract=None, ax=None):
    '''Takes either a shot gather file or raw stream file and plots the data'''
    
    if isinstance(data,str):
        #load in file to get raw data matrix and stream then extract parameters
        gather, gatherStream = loadShotGather(data)
        gatherParams = gatherParameters(gatherStream)

    else:
        #extract traces from stream and get acquisition parameters
        gather = np.stack(t.data for t in data.traces)
        gatherParams = gatherParameters(data)
        
    #plotting parameters
    perc = np.percentile(gather, percentile)
    gain = gain
    nTrace = gatherParams['numTrc']
    dx = gatherParams['dx']
    npts = gatherParams['numSamps']
    dt = gatherParams['dt']
    srcOffset = gatherParams['X1']
    start = int(startTime/dt)    #start time
    end = int(endTime/dt)   #end time
    
    xOffset = np.arange(0,(nTrace)*dx,dx)
    t = np.arange(0,npts*dt,dt)


    #set tick parameters
    majorLocator = MultipleLocator(0.1)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(0.01)

    
    #plot the data
    if ax != None:
        ax = ax
        msg = 'Master Plot'
    else:
        fig, ax = plt.subplots(1,1, figsize=(15,15))
        msg = 'No Master Plot'

    trNum = 1

    for xpos, tr in zip(xOffset, gather):
        #optional normalization
        if traceNorm == 'Per Trace': amp = gain * tr / np.nanmax(tr) + (xpos+srcOffset)
        else: amp = gain * tr / perc + (xpos+srcOffset)

        ax.plot(amp[start:end], t[start:end], c='k', lw=0.5)
        ax.fill_betweenx(t[start:end], amp[start:end], 
                         (xpos+srcOffset), amp[start:end] > (xpos+srcOffset), color='r', lw=0, alpha=0.7)
        ax.fill_betweenx(t[start:end], amp[start:end], 
                         (xpos+srcOffset), amp[start:end] < (xpos+srcOffset), color='blue', lw=0, alpha=0.7)
        ax.text(xpos+srcOffset, 0, str(trNum), horizontalalignment='center', fontsize=8)

        trNum += 1


    if refract:
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.set_xlim(0,np.nanmax(xOffset)+srcOffset*1.5)
        ax.grid(axis='y', which='both')
        ax.set_axisbelow(True)

    ax.set_ylabel('Time [s]')
    ax.set_xlabel('Offset from Source [m]')
    ax.set_title(title)
    ax.invert_yaxis()
    if msg == 'No Master Plot':
        plt.show()
        
    return



def dcPhaseShift(data, minVel, maxVel, minFreq, maxFreq, padLen=None, dv=0.1, title='Dispersion Curve', ax=None):
    '''Takes either a gather file or an Obspy stream along with min/max velocity values to check
    and min/max frequency values to check between; output is a dispersion curve.
    
    After Park et al. 1998'''
    
    if isinstance(data, str):
        #load in file to get raw data matrix and stream then extract parameters
        gather, gatherStream = loadShotGather(data)
        params = gatherParameters(gatherStream)
        
    else:
        #extract traces from stream and get acquisition parameters
        gather = np.stack(t.data for t in data.traces)
        params = gatherParameters(data)
        

    #pad time axis if specified
    if padLen != None:
        gather = np.pad(gather, ((0,0),(padLen,padLen)), 'constant', constant_values=(0,0))

    #calculate offset vector
    xx = np.arange(params['X1'], params['X1'] + params['numTrc']*params['dx'], params['dx'])
    
    #compute fft and associated freqs
    G = np.fft.fft(gather)
    freqs = np.fft.fftfreq(gather[0].size, params['dt'])
    
    #select only positive frequencies from fft ouput; i.e. first half
    Gp = G[:,:freqs.size//2]
    freqsp = freqs[:freqs.size//2]
    
    #select frequencies
    df = freqs[10]-freqs[9]
    fMax = int(maxFreq/df)
    fMin = int(minFreq/df)

    
    #set up velocities to test
    #dv = 0.1   #velocity step to test
    testVels = np.arange(minVel, maxVel, dv)
    
    #create empty array to hold transformed data and mode picks
    V = np.zeros((len(freqsp[fMin:fMax]), len(testVels)))
    M0 = np.zeros((len(freqsp[fMin:fMax])))
    
    ######TRANSFORM
    #run through freqs first
    for f in range(len(freqsp[fMin:fMax])):
        #then run through each test velocity
        #print(freqsp[f+fMin])
        for v in range(len(testVels)):
            V[f,v] = np.abs( np.sum( Gp[:,f+fMin]/np.abs(Gp[:,f+fMin]) * np.exp(1j*2*np.pi*freqsp[f+fMin]*xx /testVels[v]) ) )

    #normalize by the numbre of traces in the gathre (as suggested by Olafsdottir 2018)
    Vnorm = V/params['numTrc']        
    
    
    
    ###########
    #PLOT
    ###########
    if ax != None:
        ax = ax
        msg = 'Master Plot'
    else:
        fig, ax = plt.subplots(1,1, figsize=(15,8))
        msg = 'No Master Plot'


    majorXLocator = MultipleLocator(10) #sets the major tick interval to 10
    majorXFormatter = FormatStrFormatter('%d')
    minorXLocator = MultipleLocator(5) #sets the minor tick interval to every 5
    majorYLocator = MultipleLocator(50) #sets the major tick interval to every 50
    majorYFormatter = FormatStrFormatter('%d')
    minorYLocator = MultipleLocator(25) #sets the minor tick interval to every 25


    #plot calculated dispersion image
    dispC = ax.imshow(Vnorm.T, aspect='auto', interpolation='none', extent=[fMin*df,fMax*df,maxVel,minVel])
    ax.invert_yaxis()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase Velocity [m/s]')
    ax.xaxis.set_major_locator(majorXLocator)
    ax.xaxis.set_major_formatter(majorXFormatter)
    ax.xaxis.set_minor_locator(minorXLocator)
    ax.yaxis.set_major_locator(majorYLocator)
    ax.yaxis.set_major_formatter(majorYFormatter)
    ax.yaxis.set_minor_locator(minorYLocator)
    ax.set_title(title)

    ax.grid(which='both', linestyle='--', alpha=0.75)
    
    if msg == 'No Master Plot':
        fig.colorbar(dispC, ax = ax, shrink = 0.7)
        plt.show()    
    
    return 



def dcVelSum(data, minVel, maxVel, minFreq, maxFreq, dv=1, padLen=None, title='Summed Velocity'):
    '''Takes either a gather file or an Obspy stream along with min/max velocity values to check
    and min/max frequency values to check between; output is a plot showing summed DC amplitudes for all
    frequencies at a given velocity value.

    Shows (in theory) non-dispersive energy in the DC image
    '''  
   
    
    ################
    #Calculate Image
    ################

    if isinstance(data, str):
        #load in file to get raw data matrix and stream then extract parameters
        gather, gatherStream = loadModelDAT(data)
        params = getModelParams(gatherStream)
        
    else:
        #extract traces from stream and get acquisition parameters
        gather = np.stack(t.data for t in data.traces)
        params = getModelParams(data)
        

    #pad time axis if specified
    if padLen != None:
        gather = np.pad(gather, ((0,0),(padLen,padLen)), 'constant', constant_values=(0,0))

    #calculate offset vector
    xx = np.arange(params['X1'], params['X1'] + params['numTrc']*params['dx'], params['dx'])

    #compute fft and associated freqs
    G = np.fft.fft(gather)
    freqs = np.fft.fftfreq(gather[0].size, params['dt'])
    
    #select only positive frequencies from fft ouput; i.e. first half
    Gp = G[:,:freqs.size//2]
    freqsp = freqs[:freqs.size//2]
    
    #select frequencies
    df = freqs[10]-freqs[9]
    fMax = int(maxFreq/df)
    fMin = int(minFreq/df)

    
    #set up velocities to test
    testVels = np.arange(minVel, maxVel, dv)
    
    #create empty array to hold transformed data and mode picks
    V = np.zeros((len(freqsp[fMin:fMax]), len(testVels)))
    numF = V.shape[0]
    
    ######TRANSFORM
    #run through freqs first
    for f in range(len(freqsp[fMin:fMax])):
        #then run through each test velocity
        #print(freqsp[f+fMin])
        for v in range(len(testVels)):
            V[f,v] = np.abs( np.sum( Gp[:,f+fMin]/np.abs(Gp[:,f+fMin]) * np.exp(1j*2*np.pi*freqsp[f+fMin]*xx /testVels[v]) ) )

    #normalize by the numbre of traces in the gathre (as suggested by Olafsdottir 2018)
    Vnorm = V/params['numTrc']

    #calculate summation for all frequencies along single velocity value
    vSum = Vnorm.sum(axis=0) / numF  #normalize by number of frequency values
    vSum_array = np.asarray([testVels, vSum])


    #########
    #PLOT
    #########
    
    #get min/max range of velocity values and appropriate step
    tickStep = int(np.ceil(((maxVel - minVel)/20) / 10) * 10) #divide vel range into 20 and round to nearest 10
    yTicks = np.arange(minVel, maxVel+tickStep, tickStep)

    
    fig = plt.figure(figsize=(15,6))
    gs = gridspec.GridSpec(1,2, width_ratios=[0.9,0.1], wspace=0.01)

    vsAX = plt.subplot(gs[0,1])
    vsAX.plot(vSum, testVels)
    vsAX.set_xlabel('Amplitude Summation')
    vsAX.set_yticks(yTicks)
    vsAX.yaxis.tick_right()
    vsAX.set_ylim(minVel, maxVel)
    vsAX.grid(which='both')
    vsAX.set_title('Vel. Sum.')

    #DC image
    dcAX = plt.subplot(gs[0,0])
    dispC = dcAX.imshow(Vnorm.T, aspect='auto', interpolation='none', extent=[fMin*df,fMax*df,maxVel,minVel])
    dcAX.invert_yaxis()
    dcAX.set_xlabel('Frequency [Hz]')
    dcAX.set_ylabel('Phase Velocity [m/s]')
    dcAX.set_title('Dispersion Image')
    dcAX.grid(which='both')
    #fig.colorbar(dispC, ax = ax, shrink = 0.7)

    #Plot title
    supT = fig.suptitle(title, x=0.445, y=0.98, fontsize=14)

    plt.show()

    return



def gatherFilter(data, fmin, fmax):
    '''
    Takes either a gather file or a Obspy stream file and bandpass filters it between the specified min and max frequencies
    '''
    
    if isinstance(data, str):
        #load in file to get raw data matrix and stream then extract parameters
        gather, gatherStream = loadShotGather(data)
        
    else: gatherStream = data
    
    #must filter a copied stream in order to preserve original gather
    gatherStream_COPY = gatherStream.copy()
    
    return gatherStream_COPY.filter('bandpass', freqmin=fmin, freqmax=fmax)


def plotSurvey(N, dx, X1, dSR, totalG, refX=0, flat=False):
    '''
    Takes basic MASW survey parameters and plots source and receiver locations for each shot gather
    
    N is the number of receivers used in gather
    dx is the spacing between each receiver
    X1 is the source offset from first receiver
    dSR is the survey move up distance between shots
    totalG is the total number of gathers to collect
    refX is the reference UTM coordinate for the first SOURCE location; default is 0
    '''
    
    #calculate base receiver locations and associated 'Y' coordinates
    Rx = np.arange(0,N*dx,dx)
    Y = np.full((1,N), 1)
    
    #set location for source 'Y' position
    sY = 1
    
    fig, ax = plt.subplots(1,1, figsize=(15,10))

    if flat==False:
        for gather in range(totalG):
            ax.scatter(refX, sY, label='Source' if gather==0 else '', s=100, c='r', marker='*')
            ax.scatter(refX + X1 + Rx, Y, label='Receiver' if gather==0 else '', s=100, c='b', marker='v')

            refX += dSR
            sY += 1
            Y += 1
            
    if flat==True:
        for gather in range(totalG):
            ax.scatter(refX, sY, label='Source' if gather==0 else '', s=100, c='r', marker='*')
            ax.scatter(refX + X1 + Rx, Y, label='Receiver' if gather==0 else '', s=100, c='b', marker='v')
            refX += dSR
        
    plt.legend()
    ax.set_xlabel('Easting')
    ax.set_ylabel('Source-Receiver Line Number')
    ax.set_axisbelow(True)    #put gridlines behind source/receivers
    ax.grid(which='both', ls='--')
    plt.show()
    
    return


def offsetSpectrum(data, fmax, norm=True):
    '''
    Takes either a gather file or an Obspy stream; output is a plot of each traces amplitude spectrum vs offset.
    
    '''
    
    if isinstance(data, str):
        #load in file specified to get raw data matrix and stream then extract parameters
        gather, gatherStream = loadShotGather(data)
        params = gatherParameters(gatherStream)
        
    else:
        #extract traces from stream and get acquisition parameters
        gather = np.stack(t.data for t in data.traces)
        params = gatherParameters(data)

    
    #fft the gather
    G = np.fft.fft(gather)
    f = np.fft.fftfreq(gather[0].size, d=params['dt'])

    #selet only positive frquencies and coeff
    G_pos = G[:,:len(G[0])//2]
    f_pos = f[:len(f)//2]


    #####calculate image
    #get params
    fEnd = fmax
    df = f_pos[1] - f_pos[0]
    endIdx = int(fEnd/df)  #find index for specified end frequency 

    #create initial empy image
    fftImg = np.empty([params['numTrc'], endIdx])

    #loop through and store in image array
    for t in range(params['numTrc']):
        
        if norm:
            fftImg[t,:] = abs(G_pos[t,0:endIdx])/np.amax(abs(G_pos[t,0:endIdx]))   #normalized

        else:
            fftImg[t,:] = abs(G_pos[t,0:endIdx])   #not normalized


    #####PLOT
    fig, ax = plt.subplots(figsize=(16,8))

    img = ax.imshow(fftImg, aspect='auto', extent=[0,fEnd,params['numTrc'],0], interpolation='hanning')
    ax.invert_yaxis()
    ax.set_ylabel('Trace Number')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_title('Normalized Amplitude Spectrum vs Trace Offset')
    fig.colorbar(img, ax = ax, shrink=0.7)

    plt.show()


    return

def fkPlot(data, fMin, fMax, padLen=None, interp='hanning', title='f-k Domain'):
    '''
    Take either a gather file or Obspy stream and outputs its f-k spectrum
    ''' 
    if isinstance(data, str):
    #load in file to get raw data matrix and stream then extract parameters
        gather, gatherStream = loadShotGather(data)
        params = gatherParameters(gatherStream)

    else:
    #extract traces from stream and gets acquisition parameters
        gather = np.stack(t.data for t in data.traces)
        params = gatherParameters(data)

    #calculate 2D FFT and select only positive frequencies and associated unwrapped wavenumbers
    if padLen != None:
        gather = np.pad(gather, ((padLen,padLen),(padLen,padLen)), 'constant', constant_values=(0,0))
    FK = np.fft.fft2(gather).T / gather.size #need to transpose due to shape of returned array
    numPosF = FK[:,1].size//2 #number of positive frequencies; along 2nd col
    fkUnwrap = FK[numPosF:,:] #select only positive frequencies; 
    fkUnwrap = np.flipud(fkUnwrap) #flips array to make indexing more intuitive

    #calculate frequency and wavenumber axis
    f_np = np.fft.fftfreq(FK[:,0].size, d=params['dt'])
    k_np = np.fft.fftfreq(FK[0,:].size, d=params['dx'])
    df = f_np[1] - f_np[0] #frequency step
    dk = k_np[1] - k_np[0] #wavenumber step

    #Plotting extent calculations
    minK, maxK = 0, 2*(1/(2*params['dx'])) #NOTE: maxK is two times the Nyquist k
    fmin, fmax = int(np.rint(fMin/df)), int(np.rint(fMax/df)) #index of desired min and max f
    kmin, kmax = 0, k_np.size-1 #default to select all wavenumber values possible
    #kmin, kmax = int(np.rint(minK/dk)), k_np.size-1


    ###PLOT####
    fig, ax = plt.subplots(1,1, figsize=(12,10))

    fkPlot = ax.imshow(abs(fkUnwrap[fmin:fmax, kmin:kmax]), interpolation=interp, aspect='auto', origin='lower',
        extent=[minK, maxK, fMin, fMax])
    fig.colorbar(fkPlot, ax = ax, shrink = 0.7)
    ax.set_title(title)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Wavenumber [cycles/m]')
    ax.grid(which='both', alpha=0.25)
    ax.set_axisbelow(True)

    plt.show()

    return 



def dcVelSum(data, minVel, maxVel, minFreq, maxFreq, dv=1, padLen=None, title='Summed Velocity'):
    '''Takes either a gather file or an Obspy stream along with min/max velocity values to check
    and min/max frequency values to check between; output is a plot showing summed DC amplitudes for all
    frequencies at a given velocity value.

    Shows (in theory) non-dispersive energy in the DC image
    '''  
   
    
    ################
    #Calculate Image
    ################

    if isinstance(data, str):
        #load in file to get raw data matrix and stream then extract parameters
        gather, gatherStream = loadModelDAT(data)
        params = gatherParameters(gatherStream)
        
    else:
        #extract traces from stream and get acquisition parameters
        gather = np.stack(t.data for t in data.traces)
        params = gatherParameters(data)
        

    #pad time axis if specified
    if padLen != None:
        gather = np.pad(gather, ((0,0),(padLen,padLen)), 'constant', constant_values=(0,0))

    #calculate offset vector
    xx = np.arange(params['X1'], params['X1'] + params['numTrc']*params['dx'], params['dx'])
    
    #compute fft and associated freqs
    G = np.fft.fft(gather)
    freqs = np.fft.fftfreq(gather[0].size, params['dt'])
    
    #select only positive frequencies from fft ouput; i.e. first half
    Gp = G[:,:freqs.size//2]
    freqsp = freqs[:freqs.size//2]
    
    #select frequencies
    df = freqs[10]-freqs[9]
    fMax = int(maxFreq/df)
    fMin = int(minFreq/df)

    
    #set up velocities to test
    testVels = np.arange(minVel, maxVel, dv)
    
    #create empty array to hold transformed data and mode picks
    V = np.zeros((len(freqsp[fMin:fMax]), len(testVels)))
    numF = V.shape[0]
    
    ######TRANSFORM
    #run through freqs first
    for f in range(len(freqsp[fMin:fMax])):
        #then run through each test velocity
        #print(freqsp[f+fMin])
        for v in range(len(testVels)):
            V[f,v] = np.abs( np.sum( Gp[:,f+fMin]/np.abs(Gp[:,f+fMin]) * np.exp(1j*2*np.pi*freqsp[f+fMin]*xx /testVels[v]) ) )

    #normalize by the numbre of traces in the gathre (as suggested by Olafsdottir 2018)
    Vnorm = V/params['numTrc']

    #calculate summation for all frequencies along single velocity value
    vSum = Vnorm.sum(axis=0) / numF  #normalize by number of frequency values
    vSum_array = np.asarray([testVels, vSum])


    #########
    #PLOT
    #########
    
    #get min/max range of velocity values and appropriate step
    tickStep = int(np.ceil(((maxVel - minVel)/20) / 10) * 10) #divide vel range into 20 and round to nearest 10
    yTicks = np.arange(minVel, maxVel+tickStep, tickStep)

    
    fig = plt.figure(figsize=(15,6))
    gs = gridspec.GridSpec(1,2, width_ratios=[0.9,0.1], wspace=0.01)

    vsAX = plt.subplot(gs[0,1])
    vsAX.plot(vSum, testVels)
    vsAX.set_xlabel('Amplitude Summation')
    vsAX.set_yticks(yTicks)
    vsAX.yaxis.tick_right()
    vsAX.set_ylim(minVel, maxVel)
    vsAX.grid(which='both')
    vsAX.set_title('Vel. Sum.')

    #DC image
    dcAX = plt.subplot(gs[0,0])
    dispC = dcAX.imshow(Vnorm.T, aspect='auto', interpolation='none', extent=[fMin*df,fMax*df,maxVel,minVel])
    dcAX.invert_yaxis()
    dcAX.set_xlabel('Frequency [Hz]')
    dcAX.set_ylabel('Phase Velocity [m/s]')
    dcAX.set_title('Dispersion Image')
    dcAX.grid(which='both')
    #fig.colorbar(dispC, ax = ax, shrink = 0.7)

    #Plot title
    supT = fig.suptitle(title, x=0.445, y=0.98, fontsize=14)

    plt.show()

    return


def dcPhaseShiftTEST(data, minVel, maxVel, minFreq, maxFreq, padLen=None, title='Dispersion Curve', pick=False):
    '''Takes either a gather file or an Obspy stream along with min/max velocity values to check
    and min/max frequency values to check between; output is a dispersion curve.
    
    After Park et al. 1998'''
    
    #####
    #Get data
    #####
    if isinstance(data, str):
        #load in file to get raw data matrix and stream then extract parameters
        gather, gatherStream = loadShotGather(data)
        params = gatherParameters(gatherStream)
        
    else:
        #extract traces from stream and get acquisition parameters
        gather = np.stack(t.data for t in data.traces)
        params = gatherParameters(data)
        
    

    ####
    #Spectral Work
    ####

    #pad time axis if specified
    if padLen != None:
        gather = np.pad(gather, ((0,0),(padLen,padLen)), 'constant', constant_values=(0,0))

    #calculate offset vector
    xx = np.arange(params['X1'], params['X1'] + params['numTrc']*params['dx'], params['dx'])
    
    #compute fft and associated freqs
    G = np.fft.fft(gather)
    freqs = np.fft.fftfreq(gather[0].size, params['dt'])
    
    #select only positive frequencies from fft ouput; i.e. first half
    Gp = G[:,:freqs.size//2]
    freqsp = freqs[:freqs.size//2]
    
    #select frequencies
    df = freqs[10]-freqs[9]
    fMax = int(maxFreq/df)
    fMin = int(minFreq/df)

    #print(fMin, fMax)
    
    #set up velocities to test
    dv = 0.1   #velocity step to test
    testVels = np.arange(minVel, maxVel, dv)
    
    #create empty array to hold transformed data and mode picks
    V = np.zeros((len(freqsp[fMin:fMax]), len(testVels)))
    
    ####
    #TRANSFORM
    ####

    #run through freqs first
    for f in range(len(freqsp[fMin:fMax])):
        #then run through each test velocity
        #print(freqsp[f+fMin])
        for v in range(len(testVels)):
            V[f,v] = np.abs( np.sum( Gp[:,f+fMin]/np.abs(Gp[:,f+fMin]) * np.exp(1j*2*np.pi*freqsp[f+fMin]*xx /testVels[v]) ) )

    #normalize by the numbre of traces in the gathre (as suggested by Olafsdottir 2018)
    Vnorm = V/params['numTrc']        
    
    
    ####
    #PLOT
    ####

    if pick == False:
        fig, ax = plt.subplots(1,1, figsize=(12,10))
        
        dispC = ax.imshow(Vnorm.T, aspect='auto', interpolation='none', extent=[fMin*df,fMax*df,maxVel,minVel])
        fig.colorbar(dispC, ax = ax, shrink = 0.7)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Phase Velocity [m/s]')
        ax.set_title(title)
        
        plt.legend()
        plt.grid(which='both')
        plt.show()

    if pick == True:

        import IPython
        shell = IPython.get_ipython()
        shell.enable_matplotlib(gui='qt')

        fig, ax = plt.subplots(1,1, figsize=(12,10))
        
        dispC = ax.imshow(Vnorm.T, aspect='auto', interpolation='none', extent=[fMin*df,fMax*df,maxVel,minVel])
        fig.colorbar(dispC, ax = ax, shrink = 0.7)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Phase Velocity [m/s]')
        ax.set_title(title)
        
        plt.legend()
        plt.grid(which='both')

        #https://matplotlib.org/api/_as_gen/matplotlib.pyplot.ginput.html
        M0 = plt.ginput(n=-1, timeout=0, show_clicks=True)

        plt.show()

        #reset to inline 
        shell = IPython.get_ipython()
        shell.enable_matplotlib(gui='inline')
    
    
    if pick == False: return Vnorm
    if pick == True: return np.asarray(M0) 