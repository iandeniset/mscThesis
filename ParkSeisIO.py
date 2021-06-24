####################################
###CODE FOR ParkSEIS I/O
####################################

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from scipy.interpolate import griddata
import seaborn as sns
import os
from pathlib import Path
import sys
import obspy as op
from obspy import read


import warnings
warnings.filterwarnings('ignore')


def loadModelDAT(file):
	'''
	Takes a ParkSeis '.dat' file generated during modeling and returns an Obspy stream object and raw traces
	'''

	#load in raw .dat file
	strm = read(file)
	
	#generator function here to save memory for big files; basically list comprehension 
	trcs = np.stack(t.data for t in strm.traces)
	
	return trcs, strm


def loadParkSeisDataTEXT(file):
	'''
	Takes in a seismic data file exported from ParkSEIS as a .TXT file and returns
	model parameters and traces loaded into a numpy array of shape (trNum, sampNum)
	'''
	
	#load in the text file using np.genfromtext
	tr, t, amp = np.genfromtxt(file, unpack=True)
	
	#get model parameters and load into dict
	nTrace = int(np.unique(tr)[-1])
	npts = int(tr.size/nTrace)
	dt = t[5]-t[4]
	
	modParamsTXT = {
		'numTrc' : nTrace,
		'numSamps' : npts,
		'dt' : dt
	}
	
	#load amplitudes into array to return as 'traces'
	txtTraces = np.full((nTrace, npts), np.nan) #create empty array of proper shape and fill with np.nan values
	
	for trace in np.unique(tr): #load into created empty array
		trIdx = int(trace)-1
		txtTraces[trIdx] = amp[tr==trace]
		
	return modParamsTXT, txtTraces


def getModelParams(stream):
	'''
	Takes in a Obspy stream object for a ParkSeis model and returns a dictionary of useful parameters
	'''

	numTrc = len(stream)
	numSamps = stream[0].stats.npts
	dt = float(stream[0].stats.delta)

	return dict(numTrc=numTrc, numSamps=numSamps, dt=dt)


def plotModelData(data, startTime, endTime, X1, dx, percentile=99.99, gain=1, traceNorm=None, title='Shot Record', ax=None):
	'''
	Takes either a model file or Obspy stream and plots the data
	 INPUT:
	 X1 --> Offset from source to first receiver
	 dx --> receiver spacing
	'''

	if data.endswith('.TXT'):
		#case if model file is exported as TXT file
		modParams, traces = loadParkSeisDataTEXT(data)

	elif isinstance(data, str):
		#case if data is a file path; loads stream and parameters then stacks traces into array
		traces, stream = loadModelDAT(data)
		modParams = getModelParams(stream)

	else:
		#case if data is a Obspy stream object
		traces = np.stack(t.data for t in data.traces)
		modParams = getModelParams(data)

	#params for plotting
	perc = np.percentile(traces, percentile)
	gain = gain
	nTrace = modParams['numTrc']
	npts = modParams['numSamps']
	dt = modParams['dt']
	dx = dx
	srcOffset = X1
	start = int(startTime/dt)
	end = int(endTime/dt)

	xOffset = np.arange(0, (nTrace)*dx, dx)
	t = np.arange(0, npts*dt, dt)


	#plot
	if ax != None:
		ax = ax
		msg = 'Master Plot'
	else:
		fig, ax = plt.subplots(1,1, figsize=(15,15))
		msg = 'No Master Plot'

	trNum = 1

	for xpos, tr in zip(xOffset, traces):
		#optional normalization
		if traceNorm == 'Per Trace': amp = gain * tr / np.nanmax(tr) + (xpos+srcOffset) 
		else: amp = gain * tr / perc + (xpos+srcOffset)

		ax.plot(amp[start:end], t[start:end], c='k', lw=0.5)
		ax.fill_betweenx(t[start:end], amp[start:end], 
			(xpos+srcOffset), amp[start:end] > (xpos+srcOffset), color='r', lw=0, alpha=0.7)
		ax.fill_betweenx(t[start:end], amp[start:end], 
			(xpos+srcOffset), amp[start:end] < (xpos+srcOffset), color='blue', lw=0, alpha=0.7)
		ax.text(xpos+srcOffset, 0, str(trNum), horizontalalignment='center', fontsize=9)

		trNum += 1

	ax.set_ylabel('Time [s]')
	ax.set_xlabel('Offset from Source [m]')
	ax.set_title(title)
	ax.invert_yaxis()
	if msg == 'No Master Plot':
		plt.show()


	return

def dcModelPhaseShift(data, minVel, maxVel, minFreq, maxFreq, X1, dx, dv=0.1, padLen=None, title='Dispersion Image', overLayDC=None, 
	velSum=False, ax=None):
	'''Takes either a gather file or an Obspy stream along with min/max velocity values to check
	and min/max frequency values to check between; output is a dispersion curve.
	
	After Park et al. 1998'''
	
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
	xx = np.arange(X1, X1 + params['numTrc']*dx, dx)
	
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
	numF = V.shape[0]
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

	#calculate summation for all frequencies along single velocity value
	vSum = Vnorm.sum(axis=0) / numF  #normalize by number of frequency values
	vSum_array = np.asarray([testVels, vSum])      
		
	
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

	#plot theoretical dispersion curve if specified; must be first to not overide extents
	if overLayDC != None:
		#loop through dictionary of theor DC curves and plot
		for dc in overLayDC:
			if overLayDC[dc][0].size > 1: #ensures mode curve exists in file
				dcF, dcC = overLayDC[dc][0,:], overLayDC[dc][1,:] #get mode frequency and phase vels
				if dc == 'Experimental DC':
					ax.plot(dcF, dcC, lw=3, c='m', label=dc)
				elif dc == 'Inverted Model DC':
					ax.plot(dcF, dcC, lw=3, label=dc, ls=':', c='k')
				else:
					ax.plot(dcF, dcC, lw=3, alpha=1, label=dc)
				#ax.plot(dcF, dcC, lw=1, c='white', alpha=0.8)
				#ax.scatter(dcF, dcC, marker='o', s=25, lw=1.5, alpha=0.5, label=dc)

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

				
	ax.legend()
	ax.grid(which='both', linestyle='--', alpha=0.75)
	
	if msg == 'No Master Plot':
		fig.colorbar(dispC, ax = ax, shrink = 0.7)
		plt.show()

	
	if velSum: return vSum_array
	else: return


#######################
##THEOR. DISPERSION CURVE CODE
#######################


def getTheorDCParams(fname):
	'''
	Takes in a file path to a ParkSeis generate theoretical dispersion curve and extracts the number of points and mode number
	'''
	with open(fname, 'r') as dispFile:
		lines = dispFile.readlines()

		#get number of points on DC 
		numPts = int(lines[0][7:]) #number of points begins after the 7th character

		#get the mode number
		if fname.endswith('(Model).DC'):
			modeNum = 0
		else:
			modeNum = int(fname[-5:-4])	#get mode number from provided file name

	return numPts, modeNum


def loadTheorDC(fname, asDict=False, dictKey=''):
	'''
	Take in a file path to a ParkSeis generate theoretical dispersion curve and extracts the curve data into an array
	'''

	numPts, modeNum = getTheorDCParams(fname) #read file and get info

	M = np.genfromtxt(fname, skip_header=1, usecols=(1,2), max_rows=numPts)

	if asDict == True: #for loading singular experimental curves from inversions
		mDict = {} 
		mDict[dictKey] = M.T
		return mDict
	else:
		return M.T


def loadTheorDC_PATH(path):
	'''
	Takes a path to a folder containing all theoretical dispersion curves generated for a given model in ParkSeis and returns
	a dictionary 

	NOTE: as of now it will not properly read in more than 10 curves - that is, M9 is the last one
	'''

	dcCurves = {}

	for root, dirs, files in os.walk(path):
		fNum = 0
		for f in files:

			#this method allows for more than 9 modes and also loading in AM0 curves
			if f.endswith('.DC'):
				key = f.split('(')[-1].split(')')[0] #this is to get the mode number label from the bracket value in file name

				dcCurves[key] = loadTheorDC(os.path.join(path, f))


	return dcCurves 


def getModelFolderPaths(path):

	'''
	Loop through a specified path and return modeled DC folder paths, paths to modeled seismic files, and paths to modle .LYR files
	
	NOTE: 'path' specified must contain a folder called 'DC' holding dispersion curves and a LYR file, as well as a 'seis' folder
	containing all modeled seismic files
	'''

	dcModelFolders = []
	seisModelFolders = []
	seisFiles = []
	lyrFiles = []
	
	#get folder paths fopr DC curves and seismic files
	for root, dirs, files in os.walk(path):
		if root.endswith('DC'):
			#print('Dispersion curves found in folder: ', root)
			dcModelFolders.append(root)
		if root.endswith('seis'):
			#print('Seismic folder found in folder: ', root)
			seisModelFolders.append(root)
			
	#get seismic file paths
	for p in seisModelFolders:
		for entry in os.listdir(p):
			if entry.endswith('v.dat'): #SELECT ONLY VERTICAL COMP OUTPUT
				seisFiles.append(os.path.join(p,entry))

	#get layer file paths
	for p in dcModelFolders:
		for entry in os.listdir(p):
			if entry.endswith('.LYR'):
				lyrFiles.append(os.path.join(p,entry))

	#for debugging
	# for f in dcModelFolders:
	# 	print(f.split('\\')[-1])
	# for f in seisFiles:
	# 	print(f.split('\\')[-1])
				
	if len(dcModelFolders) != len(seisFiles):
		raise Exception('Caution! The number of model dispersion curve folders is not the same as the number of model seismic files!')
	if len(dcModelFolders) != len(lyrFiles):
		raise Exception('Caution! The number of model dispersion curve folders is not the same as the number of model .LYR files!')
				
	return dcModelFolders, seisFiles, lyrFiles






#######################
##F-K PLOT CODE
#######################


def fkModelPlot(data, fMin, fMax, X1, dx, padLen=None, interp='hanning', title='f-k Domain', ax=None):
	'''
	Take either a gather file or Obspy stream and outputs its f-k spectrum
	''' 
	

	if isinstance(data, str):
		#load in file to get raw data matrix and stream then extract parameters
		gather, gatherStream = loadModelDAT(data)
		params = getModelParams(gatherStream)
		
	else:
		#extract traces from stream and get acquisition parameters
		gather = np.stack(t.data for t in data.traces)
		params = getModelParams(data)


	#calculate 2D FFT and select only positive frequencies and associated unwrapped wavenumbers
	if padLen != None:
		gather = np.pad(gather, ((padLen,padLen),(padLen,padLen)), 'constant', constant_values=(0,0))
	FK = np.fft.fft2(gather).T / gather.size #need to transpose due to shape of returned array
	numPosF = FK[:,1].size//2 #number of positive frequencies; along 2nd col
	fkUnwrap = FK[numPosF:,:] #select only positive frequencies; 
	fkUnwrap = np.flipud(fkUnwrap) #flips array to make indexing more intuitive

	#calculate frequency and wavenumber axis
	f_np = np.fft.fftfreq(FK[:,0].size, d=params['dt'])
	k_np = np.fft.fftfreq(FK[0,:].size, d=dx)
	df = f_np[1] - f_np[0] #frequency step
	dk = k_np[1] - k_np[0] #wavenumber step

	#Plotting extent calculations
	minK, maxK = 0, 2*(1/(2*dx)) #NOTE: maxK is two times the Nyquist k
	fmin, fmax = int(np.rint(fMin/df)), int(np.rint(fMax/df)) #index of desired min and max f
	kmin, kmax = 0, k_np.size-1 #default to select all wavenumber values possible
	#kmin, kmax = int(np.rint(minK/dk)), k_np.size-1


	###PLOT####
	if ax != None:
		ax = ax
		msg = 'Master Plot'
	else:
		fig, ax = plt.subplots(1,1, figsize=(12,10))
		msg = 'No Master Plot'

	fkPlot = ax.imshow(abs(fkUnwrap[fmin:fmax, kmin:kmax]), interpolation=interp, aspect='auto', origin='lower',
		extent=[minK, maxK, fMin, fMax])
	ax.set_title(title)
	ax.set_ylabel('Frequency [Hz]')
	ax.set_xlabel('Wavenumber [cycles/m]')
	ax.grid(which='both', alpha=0.25)
	ax.set_axisbelow(True)

	if msg == 'No Master Plot':
		fig.colorbar(fkPlot, ax = ax, shrink = 0.7)
		plt.show()

	return 


######
#READ AND PLOT LYR MODEL FILE
######

def readLYRFile(fname):
	''''
	Takes a layer file and returns all parameter values for all layers
	'''
	#get number of layers in model; note returned value is number of distinct layers PLUS one (i.e. includes half-space)
	with open(fname, 'r') as lyrFile:
		lines = lyrFile.readlines()
		numLayers = int(lines[2][14:]) #number of layers is on 3rd line starting after the 14 col
		paramLines = lines[3:(3+numLayers)] #gets all lines containing model parameters
	
	
	#load in the model parameters
	z, h, vs, vp, pr, rho, qs, qp, vsU, vsL = np.genfromtxt(fname, skip_header=4, usecols=(1,2,3,4,5,6,7,8,9,10), 
												  max_rows=numLayers, unpack=True)
	
	#calculate parameters needed
	hsADD = 1.5 #depth value to add to half-space for plotting
	hsDepth = z[numLayers-2] #get depth to half-space
	z[-1] = hsDepth + hsADD #sets depth of half-space base
	h[-1] = z[-1] - z[-2] #sets thickness of half-space
	
	dz = 0.0001
	zProf = np.arange(0, hsDepth+hsADD, dz) #calculate depth vector
	vsProf = np.full_like(zProf, np.nan)
	vpProf = np.full_like(zProf, np.nan)
	prProf = np.full_like(zProf, np.nan)
	rhoProf = np.full_like(zProf, np.nan)
	qsProf = np.full_like(zProf, np.nan)
	qpProf = np.full_like(zProf, np.nan)
	vsUProf = np.full_like(zProf, np.nan)
	vsLProf = np.full_like(zProf, np.nan)
	
	for layer in range(numLayers):
		upper = z[layer] - h[layer]
		lower = z[layer]
		
		vsProf[(zProf >= upper) & (zProf <= lower)] = vs[layer]
		vpProf[(zProf >= upper) & (zProf <= lower)] = vp[layer] 
		prProf[(zProf >= upper) & (zProf <= lower)] = pr[layer] 
		rhoProf[(zProf >= upper) & (zProf <= lower)] = rho[layer]
		qsProf[(zProf >= upper) & (zProf <= lower)] = qs[layer] 
		qpProf[(zProf >= upper) & (zProf <= lower)] = qp[layer]
		vsUProf[(zProf >= upper) & (zProf <= lower)] = vsU[layer]
		vsLProf[(zProf >= upper) & (zProf <= lower)] = vsL[layer]

	layerFileValues = {
	'Depth':zProf,
	'Vs':vsProf,
	'Vp':vpProf,
	'Poissons Ratio':prProf,
	'Density':rhoProf,
	'Qs':qsProf,
	'Qp':qpProf,
	'Vs Upper':vsUProf,
	'Vs Lower':vsLProf
	}

	return layerFileValues
	

def plotLYRFile(fname, title='Layer Model Values', ax=None):
	'''
	Takes in a path to a ParkSeis layer file and outputs a visual plot of the model parameters
	'''
	
	lyrFileVals = readLYRFile(fname)
	z = lyrFileVals['Depth']
	vs = lyrFileVals['Vs']
	vp = lyrFileVals['Vp']
	rho = lyrFileVals['Density']

	#PLOT
	if ax != None:
		ax = ax
		msg = 'Master Plot'
	else:
		fig, ax = plt.subplots(1,3, figsize=(6,10))
		msg = 'No Master Plot'


	majorYLocator = MultipleLocator(1) #sets the major tick interval to every meter
	majorYFormatter = FormatStrFormatter('%d')
	minorYLocator = MultipleLocator(0.25) #sets the minor tick interval to every 0.25 meter

	ax[0].plot(vs, z, c='r', label='Vs')
	ax[0].set_xlabel('Vs [m/s]')
	ax[0].set_ylabel('Depth [m]')
	ax[0].locator_params(axis='x', tight=True, nbins=12)
	
	ax[1].plot(vp, z, c='b', label='Vp')
	ax[1].set_xlabel('Vp [m/s]')
	ax[1].locator_params(axis='x', tight=True, nbins=8)
	
	ax[2].plot(rho, z, c='k', label='Density')
	ax[2].set_xlabel('Density [kg/m$^3$]')
	ax[2].set_xlim(1,3)
	ax[2].locator_params(axis='x', tight=True, nbins=4)

	for i in range(len(ax)):
		ax[i].invert_yaxis()
		ax[i].yaxis.set_major_locator(majorYLocator)
		ax[i].yaxis.set_major_formatter(majorYFormatter)
		ax[i].yaxis.set_minor_locator(minorYLocator)
		ax[i].grid(which='both', linestyle='--', alpha=0.75)
		ax[i].tick_params(axis='x', labelrotation=90)
		if i > 0:
			ax[i].set_yticklabels([])
			ax[i].tick_params(axis='y', which='both', left='off')
	
	
	if msg == 'No Master Plot':
		fig.suptitle(title, y=0.92)
		plt.subplots_adjust(wspace=0.05, hspace=0)
		plt.show()

	
	return

#################
#PRINT LYR FILE TABLE
#################

def printLYRfile(fname, inv=False):
	'''
	Takes in a layer file from ParkSEIS and prints out the table of model parameters
	'''
	with open(fname, 'r') as lyrFile:
		lines = lyrFile.readlines()
		numLayers = int(lines[2][14:]) #number of layers is on 3rd line starting after the 14 col
		paramLines = lines[1:(4+numLayers)] #gets all lines containing model parameters

		#for printing inversion file parameters
		if inv:
			#get inv parameters
			sectionLineNums = []
			for i, l in enumerate(lines):
				if l.startswith('------------------------------------------------------------'):
					sectionLineNums.append(i)
	
			if sectionLineNums: #if no section lines exists, still print results
				invSecStart, invSecEnd = sectionLineNums[0]+1, sectionLineNums[1]-2 #gets the line range for inversion parameters
				matchSecStart, matchSecEnd = sectionLineNums[1]+1, sectionLineNums[2] #same but for match info
	
				invParams = lines[invSecStart:invSecEnd]
				matchParams = lines[matchSecStart:matchSecEnd]

		
	print('---- PARAMETERS FOR MODEL %s: \n' %fname)
	for l in paramLines:
		print(l)

	if inv and (sectionLineNums): #again, make sure sections actually exist so can still print without error
		print('----INVERSION PARAMETERS')
		for l in invParams:
			print(l)
		print('----INVERSION MATCH')
		for l in matchParams:
			print(l)

	return


#################
#SUPER MODEL PLOT OF AWESOMENESS 
#################

def plotModelAnalysis(modPath, fmin=2, fmax=100, vmin=50, vmax=1500, dv=0.1, dx=1.5, X1=9, printParams=False, save=False):
	'''
	Takes a path to a model folder where results are stored in folders of:
		'DC' - all dispersion curves and associated .LYR file
		'seis' - all seismic model files generated (i.e. .dat files)
	'''
	
	#get DC folders, seismic paths, and LYR paths first
	dcFolder, seisFiles, lyrFiles = getModelFolderPaths(modPath)
	
	###
	#MAIN LOOP THROUGH MODEL FILES
	###
	
	for dcF, seisF, lyrF in zip(dcFolder, seisFiles, lyrFiles):

		if printParams: #print out the model parameters
			printLYRfile(lyrF)

		#LOAD SEISMIC
		modTrcs, modStrm = loadModelDAT(seisF)
		
		#LOAD IN EXPERIMENTAL DC FILES
		dcDict = loadTheorDC_PATH(dcF)
		
		###
		#PLOT
		###
		
		fig = plt.figure(figsize=(17,17))
		gs = gridspec.GridSpec(2,4, width_ratios=[0.5,0.5,0.5,2.5])
		
		#get plot title
		plotTitle = seisF.split('\\')[-1] #gets title for model
		
		#LAYER PLOT
		lyrAX = [plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2])] #create list of axes to pass to function
		plotLYRFile(lyrF, ax=lyrAX)
	
		#F-K PLOT
		fkAX = plt.subplot(gs[0,3])
		fkModelPlot(modStrm, fMin=fmin, fMax=fmax, X1=X1, dx=dx, padLen=200, ax=fkAX)
		fig.colorbar(fkAX.images[0], ax=fkAX, shrink=0.7)
		
		#DISP IMAGE PLOT
		dcAX = plt.subplot(gs[1,:])
		dcModelPhaseShift(modStrm, minVel=vmin, maxVel=vmax, dv=dv, minFreq=fmin, maxFreq=fmax, padLen=500,
			X1=X1, dx=dx, overLayDC=dcDict, velSum=True, ax=dcAX)
		fig.colorbar(dcAX.images[0], ax=dcAX, shrink=0.7)
				
		#Set title and save

		supT = fig.suptitle(plotTitle, x=0.445, y=0.92, fontsize=14)

		if save:
			 figTitle = plotTitle.replace('.dat','')  
			 plt.savefig(figTitle, dpi=300, bbox_inches='tight', bbox_extra_artists=[supT])      
	   
	
		plt.show()
		
	return



#################
#MODEL SUMMARY PLOTS FOR THESIS DOCUMENT 
#################

def plotModelAnalysis_thesis(modPath, fmin=2, fmax=100, vmin=50, vmax=1500, dv=0.1, dx=1.5, X1=9, printParams=False,
	fs=12, save=False):
	'''
	Takes a path to a model folder where results are stored in folders of:
		'DC' - all dispersion curves and associated .LYR file
		'seis' - all seismic model files generated (i.e. .dat files)
	'''
	
	#get DC folders, seismic paths, and LYR paths first
	dcFolder, seisFiles, lyrFiles = getModelFolderPaths(modPath)
	
	###
	#MAIN LOOP THROUGH MODEL FILES
	###
	
	for dcF, seisF, lyrF in zip(dcFolder, seisFiles, lyrFiles):

		if printParams: #print out the model parameters
			printLYRfile(lyrF)

		#LOAD SEISMIC
		modTrcs, modStrm = loadModelDAT(seisF)
		
		#LOAD IN EXPERIMENTAL DC FILES
		dcDict = loadTheorDC_PATH(dcF)
		
		###
		#PLOT
		###
		plt.rcParams.update({'font.size': fs}) #global font size

		fig = plt.figure(figsize=(15,5))
		gs = gridspec.GridSpec(1,5, width_ratios=[0.4,0.4,0.4,0.3,3.25], wspace=0.05)
		
		#get plot title
		plotTitle = seisF.split('\\')[-1] #gets title for model
		
		#LAYER PLOT
		lyrAX = [plt.subplot(gs[0,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2])] #create list of axes to pass to function
		plotLYRFile(lyrF, ax=lyrAX)
		lyrAX[0].set_xlim(50,350)
		lyrAX[0].locator_params(axis='x', tight=True, nbins=8)
		lyrAX[0].tick_params(axis='x', which='major', labelsize=fs*0.75)
		lyrAX[1].tick_params(axis='x', which='major', labelsize=fs*0.75)
		lyrAX[2].set_xlabel('$\\rho$ [kg/m$^3$]')
		lyrAX[2].tick_params(axis='x', which='major', labelsize=fs*0.75)
		
		#BLANK SPACER PLOT
		axSPACE = plt.subplot(gs[0,3])
		axSPACE.axis('off')

		#DISP IMAGE PLOT
		dcAX = plt.subplot(gs[0,4])
		dcModelPhaseShift(modStrm, minVel=vmin, maxVel=vmax, dv=dv, minFreq=fmin, maxFreq=fmax, padLen=500,
			X1=X1, dx=dx, overLayDC=dcDict, velSum=True, ax=dcAX)
		dcAX.set_title('')
		dcAX.get_legend().remove() #removes legend; over crowded and cant see response
		#fig.colorbar(dcAX.images[0], ax=dcAX, shrink=0.7)
				
		#Set title and save

		if save == False:
			supT = fig.suptitle(plotTitle, x=0.445, y=0.99, fontsize=14)

		if save:
			 figTitle = plotTitle.replace('.dat','')  
			 #plt.savefig(figTitle, dpi=300, bbox_inches='tight', bbox_extra_artists=[supT])      
			 plt.savefig(figTitle, dpi=100, bbox_inches='tight')      
	   
	
		plt.show()
		
	return



######################
#CREATE LAYER FILES
######################
#NOTE: this code is pretty hacked together and not user friendly

def createLYRFile(h, vs, vp, rho, sigma, qs, qp, fname, directory, fMod='', makeDir=False):
	'''
	Takes in parameters for a layered earth model and outputs a '.LYR' for use in ParkSEIS modeling
	- Earth properties are passed as lists
	- Number of parameters passed for all but thickness (h) is n+1, where n is the number of distinct layers
	- h is the thickness of each layer; list of length 'n'
	- vs, vp, and rho are the shear, compressional, and density (g/cc) of each layer; lists of length 'n+1'
	- sigma is Poissons ratio for each layer; list of length 'n+1'
	- qs and qp are the attenuation coefficients for each layer; lists of length 'n+1'
	- fname is the base name of the layer file being created (NOT including .LYR!!!)
	- directory is the directory in which file will be saved
	- fMode is a file anme modifier to be added to the base file name
	'''
	
	#if specified create sub model directories after checking they dont exist
	if makeDir:
		if os.path.exists(os.path.join(directory, fMod)):
			print('Yo dog, directory already exists.  Walk away and come back when you are paying closer attention.')
			return
		else:
			subDir = os.path.join(directory, fMod)
			os.makedirs(subDir)  #creates sub model directory
			directory = os.path.join(subDir, 'DC') #sets directory to save LYR file to 
			os.makedirs(directory)  #makes DC curve directory for sub model
			os.makedirs(os.path.join(subDir, 'seis'))  #makes seis directory for sub model
	
	
	#first check if file exists
	if fMod != '':
		fname = fname + fMod + '.LYR'  #adds file anme modifier if given
	else:
		fname = fname + '.LYR'
	if os.path.exists(os.path.join(directory, fname)):
		print('Sorry bro, file already exists.  Take a breather and come back to work when you\'re paying more attention.')
		return
	
	#create initial empty file
	Path(os.path.join(directory,fname)).touch()
	
	#write file
	#######
	#DEFINE STANDARD LINES
	#######
	
	#get number of layers in model
	numLay = len(vs)

	#first 3 lines of .LYR file
	first3Lines = f'Layer Model\nThicknessModel = UserDefinedThickness\nNumberOfLayer {numLay}\n'
	
	#parameter table header
	tableHead = '{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}\n'.format(
	'Layer#'.ljust(10),
	'Depth(m)'.ljust(10),
	'Thck(m)'.ljust(10),
	'Vs(m/s)'.ljust(10),
	'Vp(m/s)'.ljust(10),
	'Poisson'.ljust(10),
	'Density*'.ljust(10),
	'Qs*'.ljust(10),
	'Qp*'.ljust(10),
	'Vs-Lower*'.ljust(10),
	'Vs-Upper*'.ljust(10),
	)
	
	#'*Note' Section
	note = '{0}{1}{2}'.format(
	'\n*Note: Densities are in gram per cubic centimeters (gm/cc).\n',
	'Qs and Qp: Quality (Q) factors for S and P waves, respectively (...not used for dispersion calculation)\n'.rjust(111),
	'Vs-Lower and Vs-Upper, if assigned, indicate lower and upper limits of 99 percent (%) confidence in solution.\n'.rjust(117)
	)
	
	#Inofrmation Lines
	infoLines = '{0}{1}{2}{3}'.format(
	'\nVs Bounds (%): 0\n',
	'Record Number: 0\n',
	'Xcoord: 0.000\n',
	'Distance Unit: meter\n'.rjust(24)
	)
	
	#history section
	hist = [
	'\n>>> Begin of History <<<\n',
	f'   Number of Layer = {numLay}\n',
	f'   Depth to Half Space = {sum(h)}\n',
	'   Depth Conversion Ratio = Automatic\n',
	'   Constant Poisson\'s Ratio =0.400\n',
	f'   Saving As {os.path.join(directory,fname)}\n'
	'>>> End of History <<<\n',
	'v. 2011'
	]
	
	###########
	#WRITE
	###########
	
	#open file in 'append' mode
	f = open(os.path.join(directory,fname), 'a')
	
	f.write(first3Lines)  #write first 3 lines
	f.write(tableHead)  #write the table header
	
	#loop through parameters provided and write to table with formating 
	for l in range(numLay):
		lay = str(l+1) #conver current layer to string
		z = '%.3f' % sum(h[0:l+1]) #gets depth to layer through cumulative summation
		VS, VP, PR, RHO, QS, QP = '%.3f' %vs[l], '%.3f' %vp[l], '%.3f' %sigma[l], '%.3f' %rho[l], '%.3f' %qs[l], '%.3f' %qp[l] #str con
		
		if (l+1) == numLay:  #for the final half-space layer 
			f.write(
			'{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}\n'.format(
			lay.ljust(10),
			'HalfSpace'.ljust(10),
			'N/A'.ljust(10),
			VS.ljust(10),
			VP.ljust(10),
			PR.ljust(10),
			RHO.ljust(10),
			QS.ljust(10),
			QP.ljust(10),
			'0.000'.ljust(10),
			'0.000'.ljust(10)
			)
			)

		else:  #for all layers except half-space
			H = '%.3f' % h[l]
			f.write(
			'{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}\n'.format(
			lay.ljust(10),
			z.ljust(10),
			H.ljust(10),
			VS.ljust(10),
			VP.ljust(10),
			PR.ljust(10),
			RHO.ljust(10),
			QS.ljust(10),
			QP.ljust(10),
			'0.000'.ljust(10),
			'0.000'.ljust(10)
			)
			)
		
	f.write(note)  #write note section
	f.write(infoLines)  #write history lines
	for i in range(len(hist)): #loops through history lines and write
		f.write(hist[i])

	#close file
	f.close()
	
	return

def dcVelSum(data, minVel, maxVel, minFreq, maxFreq, X1, dx, dv=1, padLen=None, title='Summed Velocity'):
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
	xx = np.arange(X1, X1 + params['numTrc']*dx, dx)
	
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

def dcVelSum_MODEL(modPath, fmin=2, fmax=100, vmin=50, vmax=1500, dv=1, dx=1.5, X1=9, padLen=1000):

	#get DC folders, seismic paths, and LYR paths first
	dcFolder, seisFiles, lyrFiles = getModelFolderPaths(modPath)
	
	###
	#MAIN LOOP THROUGH MODEL FILES
	###
	
	for dcF, seisF, lyrF in zip(dcFolder, seisFiles, lyrFiles):

		#LOAD SEISMIC
		modTrcs, modStrm = loadModelDAT(seisF)
		
		#LOAD IN EXPERIMENTAL DC FILES
		dcDict = loadTheorDC_PATH(dcF)

		#PLOT
		#get plot title
		plotTitle = seisF.split('\\')[-1] #gets title for model

		dcVelSum(modStrm, minVel=vmin, maxVel=vmax, minFreq=fmin, maxFreq=fmax, X1=X1, dx=dx, 
			dv=dv, padLen=padLen, title=plotTitle)

		plt.show()


######################
#PLOT INV LAYER FILES
######################

def plotLYRFile_INV(invFile, originalFile=None, title='Vs Profile', ax=None, vBounds=None, zBounds=None, legendFS=10, 
		save=False, fs=None, **kwargs):
	'''
	Takes at min one layer file which is output from inversion and plots it
	Optional second argument is the original model layer file which can be plotted for comparison
	'''

	#parse kwargs
	saveTitle = kwargs.get('saveTitle',None)

	#take file and turn into dict if only one is passed
	if isinstance(invFile, dict) == False:
		fileDict = {'':invFile}
	else:
		fileDict = invFile

	#load original file if passed
	if originalFile != None:
		origFileVals = readLYRFile(originalFile)
		z_orig = origFileVals['Depth']
		vs_orig = origFileVals['Vs']
		vp_orig = origFileVals['Vp']
		rho_orig = origFileVals['Density']

	#PLOT
	if fs != None: #set global font size for thesis plots
		plt.rcParams.update({'font.size': fs}) #global font size

	if ax != None:
		ax = ax
		msg = 'Master Plot'
	else:
		fig, ax = plt.subplots(1,1, figsize=(5,10))
		msg = 'No Master Plot'

	#plot original file first
	if originalFile != None:
		ax.plot(vs_orig, z_orig, c='k', lw=2, ls='--', label='Original')

	#plot all files in dict
	for fKey in fileDict:
		#load layer files
		invFileVals = readLYRFile(fileDict[fKey])
		z_inv = invFileVals['Depth']
		vs_inv = invFileVals['Vs']
		vp_inv = invFileVals['Vp']
		rho_inv = invFileVals['Density']


		majorYLocator = MultipleLocator(1) #sets the major tick interval to every meter
		majorYFormatter = FormatStrFormatter('%d')
		minorYLocator = MultipleLocator(0.25) #sets the minor tick interval to every 0.25 meter

		majorXLocator = MultipleLocator(100) #sets the major tick interval to every meter
		majorXFormatter = FormatStrFormatter('%d')
		minorXLocator = MultipleLocator(50) #sets the minor tick interval to every 0.25 meter

		ax.plot(vs_inv, z_inv, lw=2, label='Inverted: \n %s' %fKey)

	#format
	ax.set_xlabel('Vs [m/s]')
	ax.set_ylabel('Depth [m]')
	ax.invert_yaxis()
	#yticks
	ax.yaxis.set_major_locator(majorYLocator)
	ax.yaxis.set_major_formatter(majorYFormatter)
	ax.yaxis.set_minor_locator(minorYLocator)
	#xticks
	ax.xaxis.set_major_locator(majorXLocator)
	ax.xaxis.set_major_formatter(majorXFormatter)
	ax.xaxis.set_minor_locator(minorXLocator)
	
	ax.grid(which='both', linestyle='--', alpha=0.75)
	ax.set_axisbelow(True)
	ax.tick_params(axis='x', labelrotation=90)
	ax.legend(fontsize=legendFS)
	ax.set_title(title)

	#apply x and y limits if specified
	if vBounds != None:
		vB = np.asarray(vBounds)
		ax.set_xlim(vB[0], vB[1])
	if zBounds != None:
		zB = np.asarray(zBounds)
		ax.set_ylim(zB[0], zB[1])
		ax.invert_yaxis() #re invert after setting bounds

	
	if msg == 'No Master Plot':
		#fig.suptitle(title, y=0.92)
		plt.subplots_adjust(wspace=0.05, hspace=0)
		plt.show()

	if save:      
		plt.savefig(saveTitle, dpi=100, bbox_inches='tight')
		plt.show()


	return


#################
#INV SUMMARY PLOT 
#################

def plotINVSummary(seisFile, expDC, invLYRFile, invDC, oneD=True, origLYRFile=None, fmin=2, fmax=100, 
	vmin=50, vmax=1500, dv=0.1, dx=1.5, X1=9, vBounds=None, zBounds=None, save=False, fs=None):
	'''
	Takes at minimum a original seismic file, extracted experimental DC, and a inverted model contained in a .LYR file and plots
	a summary figure.  
	Optional input is a original .LYR for comparison which can either be the known true model or initial model for inversion

	INPUTS:
	seisFile - .dat file (vertical comp.) containing seismic record that was inverted
	expDC - the extracted fundamental mode file from the DC image generated from the provided seisFile
	invDC - file pointing to the theoretical DC calculated from the inverted model
	invLYRFile - .LYR file containing the resultant layered earth model generated during inversion
	origLYRFile - the original or initial .LYR file containing the layered earth model
	'''
	
	#load traces and stream from provided seismic file
	t, s = loadModelDAT(seisFile)

	#load the extracted experimental DC curve provided into dict
	DC_exp = loadTheorDC(expDC, asDict=True, dictKey='Experimental DC')

	#if provided inverted DC is single file, load the modeled DC curve provided into dict
	if isinstance(invDC, dict) == False:
		DC_inv = loadTheorDC(invDC, asDict=True, dictKey='Inverted Model DC')
	elif isinstance(invDC, dict):
		DC_inv = invDC #store provided dict into var to be merged

	#merge two DC into single dict for plotting
	DC_plotDict = {**DC_exp, **DC_inv}
	
	###
	#PLOT SUMMARY FIG
	###

	if fs != None: #set global font size for thesis plots
		plt.rcParams.update({'font.size': fs}) #global font size
	
		
	fig = plt.figure(figsize=(14,5.5))
	gs = gridspec.GridSpec(1,2, width_ratios=[0.2,0.9])
		
	#get plot title
	plotTitle = 'Inversion Summary for ' + seisFile.split('\\')[-1] #gets title for model
		
	#LAYER PLOT
	lyrAX = plt.subplot(gs[0,0])
	plotLYRFile_INV(invFile=invLYRFile, originalFile=origLYRFile, vBounds=vBounds, zBounds=zBounds, ax=lyrAX)
	if fs != None:
		lyrAX.legend(fontsize=fs*.75) #super stinky way to get this done; quick and dirty
		lyrAX.set_title('')

		
	#DISP IMAGE PLOT
	dcAX = plt.subplot(gs[0,1])
	dcModelPhaseShift(s, minVel=vmin, maxVel=vmax, minFreq=fmin, dv=dv, maxFreq=fmax, padLen=1000,
		X1=X1, dx=dx, overLayDC=DC_plotDict, ax=dcAX)
	if fs != None:
		dcAX.legend(fontsize=fs*.75) #super stinky way to get this done; quick and dirty
		dcAX.set_title('')
	if save == False:	
		fig.colorbar(dcAX.images[0], ax=dcAX, shrink=0.7)
				
	#Set title and save

	if save == False: supT = fig.suptitle(plotTitle, x=0.445, y=0.97, fontsize=14)

	if save:
		figTitle = 'INVsummary_' + seisFile.split('\\')[-1]
		figTitle = figTitle.replace('.dat','')  
		#plt.savefig(figTitle, dpi=75, bbox_inches='tight', bbox_extra_artists=[supT])      
		plt.savefig(figTitle, dpi=100, bbox_inches='tight')      
	   
	
	plt.show()


	############
	#PRINT INV SUMMARY
	############
	if isinstance(invLYRFile, dict) == False:
		oneORtwo = oneD #if the inverted layer files are from a 1D or 2D inversion
		printLYRfile(invLYRFile, inv=oneORtwo)
	
	return

#################
#APPROX INV 
#################

def approxINV(expDCFile, zConv=0.3, vsConv=0.92):
	'''
	Take and experimental DC and calculates then plots an approximate inversion.
	Uses 0.3 of wavelength for depth proxy and 0.92 as velocity proxy
	'''
	
	#load in the exp DC
	dc = loadTheorDC(expDCFile, asDict=True, dictKey='Experimental DC')

	#get frequency and phase velocity
	f = dc['Experimental DC'][0]
	c = dc['Experimental DC'][1]

	#convert from frequency to wavelength
	lamb = c/f

	#calculate aprox depths and Vs values
	z = lamb*zConv
	vs = c/vsConv

	######
	#PLOT
	######

	fig, ax = plt.subplots(1,1, figsize=(6,10)) #w,h


	majorYLocator = MultipleLocator(1) #sets the major tick interval to every meter
	majorYFormatter = FormatStrFormatter('%d')
	minorYLocator = MultipleLocator(0.25) #sets the minor tick interval to every 0.25 meter

	majorXLocator = MultipleLocator(50) #sets the major tick interval to every meter
	majorXFormatter = FormatStrFormatter('%d')
	minorXLocator = MultipleLocator(25) #sets the minor tick interval to every 0.25 meter

	ax.plot(vs, z, marker='o', fillstyle='none')
	
	ax.set_xlabel('Vs [m/s]')
	ax.set_ylabel('Depth [m]')
	ax.invert_yaxis()
	#yticks
	ax.yaxis.set_major_locator(majorYLocator)
	ax.yaxis.set_major_formatter(majorYFormatter)
	ax.yaxis.set_minor_locator(minorYLocator)
	#xticks
	ax.xaxis.set_major_locator(majorXLocator)
	ax.xaxis.set_major_formatter(majorXFormatter)
	ax.xaxis.set_minor_locator(minorXLocator)
	
	ax.grid(which='both', linestyle='--', alpha=0.75)
	ax.tick_params(axis='x', labelrotation=90)


	plt.show()


	return

def experDC_quicklook(file):
	'''
	Takes an experimental DC curve file and plots a quick look overview
	Includes a scatter plot with histograms showing value distribution,
	a summary of the min and max values, and a quick inversion
	'''

	#extract experimental DC curve to dict
	DC = loadTheorDC(fname=file, asDict=True, dictKey='Experimental DC')
	DC_f, DC_v = DC['Experimental DC'][0], DC['Experimental DC'][1]

	#seaborn join plot
	jp = sns.jointplot(DC_f, DC_v, space=0, marginal_kws=dict(bins=20), stat_func=None)
	jp.set_axis_labels('Frequency [Hz]', 'Phase Velocity [m/s]')
	plt.show()

	#min max summary
	print('The bounding velocities in the experimental DC are: ', np.nanmin(DC_v), np.nanmax(DC_v))
	print('The bounding frequencies in the experimental DC are: ', np.nanmin(DC_f), np.nanmax(DC_f))
	print('The minimum and maximum wavelengths are: ', (np.nanmin(DC_v)/np.nanmax(DC_f)), (np.nanmax(DC_v)/np.nanmin(DC_f)))

	#plot quick inversion
	approxINV(expDCFile=file, zConv=0.25)


	return


def approxINV_2D(path, dx, zConv=0.3, vsConv=0.92, legend=True, returnDC=False, flip=False):
	'''
	Take a path to a folder containing experimental DC for a 2D profile and
	plots them as an approximate inversion
	path - path to extracted dispersion curves
	dx - spacing between each measurement
	zConv - wavelength conversion for depth
	vsConv - relation between measured phase velocity and shear velocity 

	**********NOTE************
	THIS ASSUMES SHOTS ARE NAMED IN SEQUENCE WITHIN THE FOLDER WITH
	THEIR RELATIVE POSITION AT THE END OF THE FILE NAME
	eg:
	Line01_Shot01(Model).DC
	Line01_Shot02(Model).DC
	...
	etc.
	'''

	##########
	#LOAD THE EXP. DC'S
	##########

	expDCs = {}

	for root, dirs, files in os.walk(path):
		for f in files:
			if f.endswith('.DC'):

				key = f.replace('.DC', '')

				expDCs[key] = loadTheorDC(os.path.join(path, f))

	#get reversed order dict keys for plotting to match DC data
	dcKeys_sorted = sorted(expDCs, reverse=True)
	
	##########
	#LOOP THROUGH AND PLOT
	##########

	#first calculate offset position vector
	xx = np.arange(0, dx*len(expDCs), dx)

	#format settings
	majorYLocator = MultipleLocator(1) #sets the major tick interval to every meter
	majorYFormatter = FormatStrFormatter('%d')
	minorYLocator = MultipleLocator(0.25) #sets the minor tick interval to every 0.25 meter

	majorXLocator = MultipleLocator(5) #sets the major tick interval to every meter
	majorXFormatter = FormatStrFormatter('%d')
	minorXLocator = MultipleLocator(1) #sets the minor tick interval to every 0.25 meter


	#set up and plot
	fig, ax = plt.subplots(1,2, figsize=(15,6), gridspec_kw={'width_ratios': [3, 1]})

	for (xPos, line, sortedKey) in zip(xx, expDCs, dcKeys_sorted):
		if flip:
			f, c = expDCs[sortedKey][0], expDCs[sortedKey][1]
		else:	
			f, c = expDCs[line][0], expDCs[line][1]

		#convert from frequency to wavelength
		lamb = c/f

		#calculate aprox depths and Vs values
		z = lamb*zConv
		vs = c/vsConv

		#create x positional vector for scatter 
		xcord = np.full_like(z, xPos)

		#set marker size base on spacing
		mSize = int(np.nanmax(xx)*4.5)
		ax[0].scatter(xcord, z, c=vs, cmap='plasma', s=mSize)
		ax[1].plot(vs, z, marker='o', fillstyle='none', label=line)

	####ax formatting
	def axFormat(ax):
		ax.set_ylabel('Depth [m]', fontsize=16)
		ax.tick_params(axis='both', labelsize=16)
		ax.invert_yaxis()
		#yticks
		ax.yaxis.set_major_locator(majorYLocator)
		ax.yaxis.set_major_formatter(majorYFormatter)
		ax.yaxis.set_minor_locator(minorYLocator)
		
		ax.set_axisbelow(True)
		ax.grid(which='both', linestyle='--', alpha=0.75)

	axFormat(ax=ax[0])
	axFormat(ax=ax[1])

	ax[0].set_xlabel('Offset [m]', fontsize=16)
	
	ax[1].set_xlabel('Vs [m/s]', fontsize=16)
	if legend:
		ax[1].legend()

	#plt.show()

	if returnDC:
		return expDCs
	else: 
		return


############################
#2D INVERSION PLOTTING AND SUMMARIES
############################


def plotInverted2DSection(file, vLimits=None, zLimits=None, flip=False, ax=None, cBar='viridis'):
	
	#######
	#READ AND MANIPULATE DATA
	#######
	#read file to get number of data lines (rows)
	with open(file, 'r') as textFile:
		lines = textFile.readlines()
		dataLinesNum = len([l for l in lines if l.startswith(' ')]) #list comp
		
	#read file into arrays using genfromtext
	x, z, vs = np.genfromtxt(file, usecols=(0,1,2), max_rows=dataLinesNum, unpack=True)
	z = -1*z #get rid of negative sign
	
	#get some basic info
	uniqueXpos = np.unique(x).size #number of x locations (i.e. shots)
	numLay = dataLinesNum/uniqueXpos #number of layers in inverted model
	
	#create new arrays which also contain 'top layer' measurements for plotting
	epsilon = 0.001 #depth offset of top and bottom layer of adjacent rows
	
	xLay, zLay, vsLay = np.array([]), np.array([]), np.array([])
	
	for row in range(dataLinesNum): #loop through and append
		if row%numLay == 0: #for first data point at each new x location
			xLay, zLay, vsLay = np.append(xLay, x[row]), np.append(zLay, 0), np.append(vsLay, vs[row])
		else:
			xLay, zLay, vsLay = np.append(xLay, x[row]), np.append(zLay, z[row-1]+epsilon), np.append(vsLay, vs[row])
		
		xLay, zLay, vsLay = np.append(xLay, x[row]), np.append(zLay, z[row]), np.append(vsLay, vs[row])
		
	#######
	#PLOT
	#######
	
	if ax != None:
		ax = ax
		msg = 'Master Plot'
	else:
		fig, ax = plt.subplots(1,1,figsize=(15,8))
		msg = 'No Master Plot'
	
	#flip so matches resistivity data
	if flip:
		xLay = xLay[::-1]

	if vLimits != None: #allow control of colorbar extents
		if np.nanmin(vsLay) < vLimits[0] and np.nanmax(vsLay) > vLimits[1]: ext = 'both'
		if np.nanmin(vsLay) < vLimits[0] and np.nanmax(vsLay) < vLimits[1]: ext = 'min'
		if np.nanmin(vsLay) > vLimits[0] and np.nanmax(vsLay) > vLimits[1]: ext = 'max'
		if np.nanmin(vsLay) > vLimits[0] and np.nanmax(vsLay) < vLimits[1]: ext = 'neither'

		#modify color bar accordingly
		levs = np.linspace(vLimits[0],vLimits[1], 50) #get levels for cbar
		ctr = ax.tricontourf(xLay, zLay, vsLay, levels=levs, extend=ext, cmap=cBar)
		ctr.set_clim(vLimits[0],vLimits[1])

	else:
		levs = np.linspace(min(vsLay), max(vsLay), 50)
		ctr = ax.tricontourf(xLay,zLay,vsLay, levels=levs, cmap=cBar)
	
	#plot original data location
	ax.scatter(xLay, zLay, c='k', marker='+', s=35, alpha = 0.55)
	
	#formatting
	if zLimits != None:
		ax.set_ylim(zLimits[0], zLimits[1])
	else:
		ax.set_ylim(min(zLay), max(zLay))
	ax.invert_yaxis()
	ax.set_ylabel('Depth [m]')
	
	ax.set_xlim(min(xLay), max(xLay))
	ax.set_xlabel('Position Along Line [m]')

	ax.set_title('Inverted Shear Velocity Profile')
	
	
	#this is a super stanky way to do this, no time to clean it up though
	if msg == 'No Master Plot':
		cbar = fig.colorbar(ctr)
		cbar.set_label('Inverted Shear Velocity [m/s]')
		plt.show()

		return
	
	else:
		return ctr


def inversionSUMMARY_2D(file, vLimits=None, zLimits=None, cBar='viridis', legend=True, flip=False, returnMODLyrs=False):
	'''
	Takes in a .TXT file from ParkSEIS that contains 2D inverted model and plots
	the 2D section and all the individual layer file inversions which make it up
	'''

	########
	#First get individual layer files
	########
	XRLmodFile = file.replace('(2DVs)','')
	XRLmodFile = XRLmodFile.replace('.TXT', '(Model).XRL')

	#read the file
	with open(XRLmodFile, 'r') as XRLfile:
		lines = XRLfile.readlines()
		numLayerFiles = len(lines)

	#load layer file paths into array
	modLYRfiles = np.genfromtxt(XRLmodFile, usecols=(2), max_rows=numLayerFiles, unpack=True, dtype='str') #load file paths
	LYRFilesXpos = np.genfromtxt(XRLmodFile, usecols=(0), max_rows=numLayerFiles, unpack=True, dtype='str') #load shot numbers
	LYRFilesXpos = [float(xPos.replace(',', '')) for xPos in LYRFilesXpos] #convert shot numbers to numbers

	modelLYRdict = dict(zip(LYRFilesXpos, modLYRfiles)) #store layer file paths and associated shot nums in dict

	#########
	#PLOT
	#########

	fig, ax = plt.subplots(1,2, figsize=(20,8), gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.29})

	#plot 2D section
	ctr = plotInverted2DSection(file, vLimits=vLimits, zLimits=zLimits, cBar=cBar, flip=flip, ax=ax[0])
	cbar = fig.colorbar(ctr, ax=ax[0], shrink=0.9, fraction=0.05, aspect=30, format='%d')
	cbar.set_label('Inverted Shear Velocity [m/s]', fontsize=16)
	cbar.ax.tick_params(labelsize=14)

	#plot layer files
	plotLYRFile_INV(invFile=modelLYRdict, ax=ax[1])
	ax[1].set_ylim(ax[0].get_ylim()) #sets y limits of plots to be the same for alllll the asssthetics
	if legend == False:
		ax[1].get_legend().remove()

	for item in ([ax[0].title, ax[0].xaxis.label, ax[0].yaxis.label] + ax[0].get_xticklabels() + ax[0].get_yticklabels()):
		item.set_fontsize(20)
	for item in ([ax[1].title, ax[1].xaxis.label, ax[1].yaxis.label] + ax[1].get_xticklabels() + ax[1].get_yticklabels()):
		item.set_fontsize(20)

	plt.tight_layout()

	if returnMODLyrs:
		return modelLYRdict
	else:
		return


def plotBSAsection(file, time=None, clim=None):
    '''
    Takes a BSA section from ParkSEIS that has been exported as a TXT file and plots it
    
    NOTE: this is a dumb function as it will only work for the geometry I designed it for
    i.e. dSR of 2*dx, dx of 1.5m, and SR of 9 m with 24 geophones
    
    time: time range to plot
    '''
    
    #read data from the file
    x, t, amp = np.genfromtxt(file, usecols=(0,1,2), unpack=True)
    
    #get various paramters for calculations and loading
    numTr = np.unique(x).size #get the number of unique x positions or traces
    n = int(x.size / numTr) #number of sample in each trace
    dt = t[6] - t[5] #sample interval
    maxT = n * dt #max time
    
    #create empty array to store data
    bsaSection = np.full((numTr, n), np.nan)
    
    #loop through data and store
    trace = 0
    tSamp = 0
    for row in range(x.size): 
        bsaSection[trace, tSamp] = amp[row]
        tSamp += 1
        if tSamp%n == 0 and tSamp > 0:
            trace += 1
            tSamp = 0
            
    ##################################
    #PLOT
    ##################################
    fig, ax = plt.subplots(1,1, figsize=(15,8))
    
    majorXLocator = MultipleLocator(10) #sets the major tick interval to 10
    majorXFormatter = FormatStrFormatter('%d')
    minorXLocator = MultipleLocator(2) #sets the minor tick interval to every 5
    
    if time != None:
        tMin, tMax = int(time[0]/dt), int(time[1]/dt)
        eTop, eBttm = time[0], time[1]
    else:
        tMin, tMax = 0, (n-1)
        eTop, eBttm = 0, maxT
    ax.imshow(np.fliplr(bsaSection[:,tMin:tMax].T), aspect='auto', interpolation='bicubic', 
    	extent=[0-17.25,45+17.25+9,eBttm,eTop], clim=clim)
    ax.set_xlabel('Surface Location [m]', fontsize=20)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel('Time [s]', fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(which='both', linestyle='--', alpha=0.75)
    
    ax.xaxis.set_major_locator(majorXLocator)
    ax.xaxis.set_major_formatter(majorXFormatter)
    ax.xaxis.set_minor_locator(minorXLocator)


    return



def depthBR_estimate(file, maxV, maxF=None, plot=False):
	'''
	Takes an experimental DC curve file and attempts to estimate the depth to BR
	using the low frequency minimum inflection point
	Prints out two options:
	1 - Based on wavelength at inflection point multiplied by 0.333
	2 - Based on average velocity of exp DC above the inflection point and a constant multiplier 

	maxV - this is to limit the velocities over which the derivative is calculated ensuring the LFM is selected
	'''

	#extract experimental DC curve to dict
	DC = loadTheorDC(fname=file, asDict=True, dictKey='Experimental DC')
	DC_f, DC_v = DC['Experimental DC'][0], DC['Experimental DC'][1]

	#select only velocites and frequencies below specified max vel; also reverse for calculations
	if maxF != None:
		v = DC_v[(DC_v < maxV) & (DC_f < maxF)][::-1]
		f = DC_f[(DC_v < maxV) & (DC_f < maxF)][::-1]
	else:
		v = DC_v[DC_v < maxV][::-1]
		f = DC_f[DC_v < maxV][::-1]

	#calculate deriv and select max fo selected velocity range
	dv = np.gradient(v) 
	dvMax = np.nanmax(dv)

	#find index where velocity equals max deriv and associated frequency/vel
	dvMax_idx = np.where(dv == dvMax)[0][0]
	velGradMax = v[dvMax_idx]
	fGradMax = f[dvMax_idx]

	#calcuate 
	vAvg = np.average(DC_v[DC_f > fGradMax][::-1]) #NOTE: cant use simply 'v' here since if maxF is given, not all vels are averaged!
	zAvg = (vAvg/fGradMax) * 0.73

	zPoint = (velGradMax/fGradMax) * 0.333

	#print
	#print('Deriv.: ', dv)
	#print('Vels: ', v)
	#print('Freqs.: ', f)
	print('The velocity and frequency for the inflection point are', velGradMax, 'm/s, and', fGradMax, 'Hz')
	print('The average velocity above the inflection point is', vAvg, 'm/s')
	print('Bedrock depth estimate from average velocity method: ', zAvg, 'm')
	if vAvg > velGradMax: #the avg vel above the inflection shouldnt be higher than the inflection itself
		print('WARNING: Average velocity above inflection point is greater than inflection velocity itself!')
	if vAvg*1.025 > velGradMax: #the avg vel above the inflection shouldnt be close to same as the inflection point itself
		print('WARNING: Average velocity above inflection point is within 2.5% of the inflection velocity itself!')
	#print('Bedrock depth estimate from single point method: ', zPoint, 'm\n')

	if plot:

		fig, ax = plt.subplots(1,1,figsize=(15,5))

		ax.plot(DC_f, DC_v, marker='o', fillstyle='none', label='Experimental DC')
		ax.plot(f, v, lw=2.5, alpha=0.75, label='Truncated DC')
		ax.axhline(y=maxV, alpha=0.85, label='Max. Velocity', c='k')
		if maxF != None:
			ax.axvline(x=maxF, alpha=0.85, label='Max. Frequency', c='k')
		ax.scatter(fGradMax, velGradMax, c='r', label='Inflection Point')

		ax.set_xlabel('Frequency [Hz]')
		ax.set_ylabel('Phase Velocity [m/s]')


		plt.grid(which='both')
		plt.legend()

		plt.show()


	return




#################################################################################
####TESTING
#################################################################################

def plotInverted2DSection_meshGrid(file, vLimits=None, zLimits=None, ax=None, cBar='viridis'):
	
	#######
	#READ AND MANIPULATE DATA
	#######
	#read file to get number of data lines (rows)
	with open(file, 'r') as textFile:
		lines = textFile.readlines()
		dataLinesNum = len([l for l in lines if l.startswith(' ')]) #list comp
		
	#read file into arrays using genfromtext
	x, z, vs = np.genfromtxt(file, usecols=(0,1,2), max_rows=dataLinesNum, unpack=True)
	z = -1*z #get rid of negative sign
	
	#get some basic info
	uniqueXpos = np.unique(x).size #number of x locations (i.e. shots)
	numLay = dataLinesNum/uniqueXpos #number of layers in inverted model
	
	#create new arrays which also contain 'top layer' measurements for plotting
	epsilon = 0.001 #depth offset of top and bottom layer of adjacent rows
	
	xLay, zLay, vsLay = np.array([]), np.array([]), np.array([])
	
	for row in range(dataLinesNum): #loop through and append
		if row%numLay == 0: #for first data point at each new x location
			xLay, zLay, vsLay = np.append(xLay, x[row]), np.append(zLay, 0), np.append(vsLay, vs[row])
		else:
			xLay, zLay, vsLay = np.append(xLay, x[row]), np.append(zLay, z[row-1]+epsilon), np.append(vsLay, vs[row])
		
		xLay, zLay, vsLay = np.append(xLay, x[row]), np.append(zLay, z[row]), np.append(vsLay, vs[row])
		
	#######
	#PLOT
	#######
	
	if ax != None:
		ax = ax
		msg = 'Master Plot'
	else:
		fig, ax = plt.subplots(1,1,figsize=(15,5))
		msg = 'No Master Plot'
	
	if vLimits != None: #allow control of colorbar extents
		if np.nanmin(vsLay) < vLimits[0] and np.nanmax(vsLay) > vLimits[1]: ext = 'both'
		if np.nanmin(vsLay) < vLimits[0] and np.nanmax(vsLay) < vLimits[1]: ext = 'min'
		if np.nanmin(vsLay) > vLimits[0] and np.nanmax(vsLay) > vLimits[1]: ext = 'max'
		if np.nanmin(vsLay) > vLimits[0] and np.nanmax(vsLay) < vLimits[1]: ext = 'neither'

		#modify color bar accordingly
		X, Y = np.meshgrid(xLay, zLay)
		vsLay_i = griddata((xLay,zLay), vsLay, (X,Y), method='linear')
		levs = np.linspace(vLimits[0],vLimits[1], 100) #get levels for cbar
		ctr = ax.contourf(xLay, zLay, vsLay_i, levels=levs, extend=ext, cmap=cBar)
		#ctr = ax.tricontourf(xLay, zLay, vsLay, levels=levs, extend=ext, cmap=cBar)
		ctr.set_clim(vLimits[0],vLimits[1])

	else:
		X, Y = np.meshgrid(xLay, zLay)
		vsLay_i = griddata((xLay,zLay), vsLay, (X,Y), method='linear')
		levs = np.linspace(min(vsLay), max(vsLay), 50)
		ctr = ax.contourf(xLay,zLay,vsLay_i, levels=levs, cmap=cBar)
		#ctr = ax.tricontourf(xLay,zLay,vsLay, levels=levs, cmap=cBar)
	
	#plot original data location
	ax.scatter(xLay, zLay, c='k', marker='+', s=35, alpha = 0.55)
	
	#formatting
	if zLimits != None:
		ax.set_ylim(zLimits[0], zLimits[1])
	else:
		ax.set_ylim(min(zLay), max(zLay))
	ax.invert_yaxis()
	ax.set_ylabel('Depth [m]')
	
	ax.set_xlim(min(xLay), max(xLay))
	ax.set_xlabel('Position Along Line [m]')

	ax.set_title('Inverted Shear Velocity Profile')
	
	
	#this is a super stanky way to do this, no time to clean it up though
	if msg == 'No Master Plot':
		cbar = fig.colorbar(ctr)
		cbar.set_label('Inverted Shear Velocity [m/s]')
		plt.show()

		return
	
	else:
		return ctr