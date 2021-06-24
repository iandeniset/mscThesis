import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import gridspec
from matplotlib import cm
from scipy.interpolate import griddata


##################
#FILE LOADING CODE
##################

##################
#Load Raw Field Data

def loadProSysData(fname):
	''' 
	Takes in a file path to a csv file exported from ProSys
	Returns:
	Tx1: Transmitter 1 location
	Tx2: Transmitter 2 location
	Rx1: Rec. location 1
	Rx2: Rec. location 2
	a: base a spacing
	n: all n values for observations
	xLoc: lateral location for each measurement 
	rho: apparent resistivity for each measurement 
	'''

	Tx1, Tx2, Rx1, Rx2, rho = np.genfromtxt(fname, delimiter=',', skip_header=1,
		usecols=(2,3,4,5,6), unpack=True)

	#find base 'a' spacing
	aSpacing = Tx2[0] - Tx1[0]

	#calculate lateral location of each measurement
	xLoc = ((Rx1 - Tx2) / 2) + Tx2

	#calculate 'n' value for each measurement
	n = np.zeros_like(Tx1)

	for obs in range(Tx1.size):
		n[obs] = ((Rx1[obs] - Tx1[obs]) / aSpacing) - 1

	return Tx1, Tx2, Rx1, Rx2, rho, xLoc, n, aSpacing 


##################
#Load inverted Data

def loadRes2DINVResults(fname):
	''' 
	Takes in a file path to a *.xyz file from res2dinv
	Returns:
	x: central location of model blocks
	z: this is the depth value for the center of the blocks
	rho: this is the inverted resistivity
	'''

	#get the number of inversion model blocks
	with open(fname, 'r') as xyzINVFile:
		lines = xyzINVFile.readlines()
		invBlockNum = int(lines[1][21:])	#info we want is on line 2 and is from location 20 to the end

	#extrace info from file
	x, z, rho = np.genfromtxt(fname, skip_header=5, usecols=(0,1,2), max_rows=invBlockNum, unpack=True)

	return x, z, rho

##################
#Load Modeled *.dat Pseudosection Data

def loadRES2DModData(fname):
	'''
	Takes in a *.dat file created from resd2dmod and returns
	xLoc: lateral position of measurement
	n: n value for measurement 
	rho: apparent resistivity for measurement
	aSpac: unit electrode spacing
	'''

	#get the number of inversion model blocks
	with open(fname, 'r') as datModFile:
		lines = datModFile.readlines()
		numObs = int(lines[3])	#info we want is on line 4 and is from location 20 to the end
		aSpac = int(lines[1][3]) #a spacing on line 2

	xLoc, n, rho = np.genfromtxt(fname, skip_header=6, usecols=(0,2,3), max_rows=numObs, unpack=True)

	return xLoc, n, rho, aSpac



##################
#PLOTTING CODE
##################

##################
#Plot Raw Field Data

def plotRawData(fname, sliceLoc, nSliceVal, plotTitle, gridDimension=500, limits=None, 
	showPoints=False, save=False, fs=12):
	'''
	Takes in a file path to a csv (or modeled .dat file) file exported from ProSys
	Plots raw data as a pseudosection
	'''

	#####################
	#Load the file
	if fname.endswith('.csv'):
		Tx1, Tx2, Rx1, Rx2, rho, xLoc, n, aSpacing = loadProSysData(fname)
	elif fname.lower().endswith('.dat'):
		xLoc, n, rho, aSpacing = loadRES2DModData(fname)


	#####################
	#Grid the Data

	#get max values
	max_x0, max_y0, max_z0 = np.amax(xLoc), np.amax(n), np.amax(rho)

	#get lateral step 
	dx = (max_x0 + aSpacing + 1) / gridDimension

	#establish grid
	xi = np.linspace(0, max_x0 + aSpacing + 1, gridDimension)
	yi = np.linspace(0, max_y0, gridDimension)

	#finally, grid the data
	zi = griddata((xLoc, n), rho, (xi[None,:], yi[:,None]), method='cubic')


	#####################
	#Plot the gridded data

	#get depth slice
	numNLevels = int(np.amax(yi)) #number of n levels
	dy = numNLevels / yi.size #y step of grid
	nSliceLoc = int(nSliceVal/dy) #array index for desired n level
	nSlice = zi[nSliceLoc,:] #get values at index

	#get 1D profile
	oneDProf = zi[:, int(sliceLoc/dx)] #get values at xloc

	#init figure
	fig = plt.figure(figsize=(17,6))
	gs = gridspec.GridSpec(2,3, width_ratios=[1,5,0.15], height_ratios=[3,1], hspace=.42, wspace=0.28)

	#main pseudosection
	ax0 = plt.subplot(gs[0,1])
	if limits != None: #allow control of colorbar extents
		if np.nanmin(zi) < limits[0] and np.nanmax(zi) > limits[1]: ext = 'both'
		if np.nanmin(zi) < limits[0] and np.nanmax(zi) < limits[1]: ext = 'min'
		if np.nanmin(zi) > limits[0] and np.nanmax(zi) > limits[1]: ext = 'max'
		if np.nanmin(zi) > limits[0] and np.nanmax(zi) < limits[1]: ext = 'neither'

		#modify color bar accordingly
		ctr = ax0.contourf(xi,yi,zi,np.arange(limits[0],limits[1],0.5), extend=ext)
		ctr.set_clim(limits[0],limits[1])
	else:
		ctr = ax0.contourf(xi,yi,zi)
	#draw contours
	ax0.contour(xi,yi,zi, 15, colors='gray', linewidths=0.75)

	if showPoints == True:
		ax0.scatter(xLoc, n, c='k', marker='+', s=35, alpha = 0.55)

	#main plot formatting 
	ax0.invert_yaxis()
	ax0.axvline(x=sliceLoc)
	ax0.axhline(y=nSliceVal)
	ax0.set_ylabel('n', fontsize=fs)
	#ax0.set_xlabel('Profile Distance [m]', fontsize=fs)
	ax0.set_title('Apparent Resistivity Pseudosection', fontsize=fs)
	ax0.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax0.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	#plot cBar
	ax3 = plt.subplot(gs[:,2])
	cbar = fig.colorbar(ctr, cax=ax3)
	cbar.set_label('Apparent Resistivity [$\Omega .m$]', fontsize=fs)
	ax3.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax3.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	#plot 1D profile at specified x location
	ax1 = plt.subplot(gs[:,0])
	ax1.plot(oneDProf, yi)
	ax1.scatter(rho[xLoc == sliceLoc], n[xLoc == sliceLoc])
	ax1.invert_yaxis()
	ax1.grid(which='both')
	ax1.set_ylabel('n', fontsize=fs)
	ax1.set_xlabel('$\\rho _a$ [$\Omega .m$]', fontsize=fs)
	ax1.set_title('$\\rho _a$ at %d m' %sliceLoc, fontsize=fs)
	ax1.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax1.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	#plot n slice at specified level
	ax2 = plt.subplot(gs[1,1])
	ax2.plot(xi, nSlice)
	ax2.scatter(xLoc[n == nSliceVal], rho[n == nSliceVal])
	ax2.set_ylim(np.nanmin(nSlice)*0.8, np.nanmax(nSlice)*1.2)
	ax2.set_xlim(np.nanmin(xi), np.nanmax(xi))
	ax2.grid(which='both')
	ax2.set_ylabel('$\\rho _a$ [$\Omega .m$]', fontsize=fs)
	ax2.set_xlabel('Profile Distance [m]', fontsize=fs)
	ax2.set_title('$\\rho _a$ at n = %d' %nSliceVal, fontsize=fs)
	ax2.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax2.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	if save ==False:
		supT = fig.suptitle(plotTitle, x=0.445, y=1.04, fontsize=15)

	figTitle = plotTitle.replace('.dat','')
	figTitle = plotTitle.replace('.csv','')

	if save:
		plt.savefig(figTitle, dpi=100, bbox_inches='tight')


	return


##################
#Plot Inverted Data

def plotINVResults(fname, sliceLoc, depthSlice, plotTitle, limits=None, AOI=None, 
	bounds=None, bCols=None, save=False, fs=12):
	'''
	Takes raw output from res2dinv and plots using tricontour (Delaunay triangulation)
	Note: this really only works for inversions with a lot of model blocks and 
	extended models.  Use interpolation for more sparse data
	limits - specify the min and max resistivity value to allow in the colorbar
	AOI - this controls the plot extent and allows 'zooming' into an area of interest (x1,x2,z1,z2)
	bounds - list of n values between which resistivity values will be grouped
	bCols - list of n-1 colors which will represent the bounded values specified in bounds
	'''

	#####################
	#Load the file
	x, z, rho = loadRes2DINVResults(fname)


	#####################
	#Plot the inverted data

	fig = plt.figure(figsize=(17,6))

	gs = gridspec.GridSpec(2,3, width_ratios=[1,5,0.15], height_ratios=[3,1], hspace=.42,wspace=0.28)

	#main plot
	ax0 = plt.subplot(gs[0,1])

	#plot limited countours if specified
	if limits != None: #allow control of colorbar extents
		if np.nanmin(rho) < limits[0] and np.nanmax(rho) > limits[1]: ext = 'both'
		if np.nanmin(rho) < limits[0] and np.nanmax(rho) < limits[1]: ext = 'min'
		if np.nanmin(rho) > limits[0] and np.nanmax(rho) > limits[1]: ext = 'max'
		if np.nanmin(rho) > limits[0] and np.nanmax(rho) < limits[1]: ext = 'neither'

		#modify color bar accordingly
		levs = np.linspace(limits[0],limits[1], 20) #get levels for cbar
		ctr = ax0.tricontourf(x, z, rho, levels=levs, extend=ext)
		ctr.set_clim(limits[0],limits[1])
	
	#bounded resistivity range plot
	elif (bounds != None) and (bCols != None):
		cmap = matplotlib.colors.ListedColormap(bCols)
		norm = matplotlib.colors.BoundaryNorm(bounds, len(bCols), clip=True)

		levs = bounds #set levels for cbar
		ctr = ax0.tricontourf(x, z, rho, norm=norm, cmap=cmap, levels=levs, alpha=0.9)

	#plot regular contours if not specified
	else:
		levs = 10 #set levels for cbar
		ctr = ax0.tricontourf(x, z, rho, levels=levs)

	#draw contours
	ax0.tricontour(x, z, rho, colors='gray', linewidths=0.75, levels=levs)


	#if area of interest if specified
	if AOI:
		aoi_x0, aoi_x1, aoi_z0, aoi_z1 = AOI[0], AOI[1], AOI[2], AOI[3]

	if AOI:
		ax0.set_xlim(aoi_x0, aoi_x1)
		ax0.set_ylim(aoi_z0, aoi_z1)
	ax0.invert_yaxis()
	ax0.axvline(x=sliceLoc)
	ax0.axhline(y=depthSlice)
	ax0.set_ylabel('Depth [m]', fontsize=fs)
	#ax0.set_xlabel('Profile Distance [m]', fontsize=fs)
	ax0.set_title('Inverted Resistivity Profile', fontsize=fs)
	ax0.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax0.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	#color bar for main plot
	ax3 = plt.subplot(gs[:,2])
	cbar = fig.colorbar(ctr, cax=ax3)
	cbar.set_label('Resistivity [$\Omega .m$]', fontsize=fs)
	ax3.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax3.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	#depth profile
	ax1 = plt.subplot(gs[:,0])
	profileX, profileY = rho[x == find_nearest(x, sliceLoc)], np.unique(z)
	ax1.scatter(profileX, profileY)
	ax1.plot(profileX, profileY)
	if AOI:
		ax1.set_ylim(aoi_z0, aoi_z1)
	ax1.invert_yaxis()
	ax1.grid(which='both')
	ax1.set_ylabel('Depth [m]', fontsize=fs)
	ax1.set_xlabel('$\\rho$ [$\Omega .m$]', fontsize=fs)
	ax1.set_title('$\\rho$ at %d m' %sliceLoc, fontsize=fs)
	if limits != None:
		ax1.set_xlim(limits[0], limits[1])
	ax1.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax1.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	#depth slice
	ax2 = plt.subplot(gs[1,1])
	depthX, depthY = np.unique(x), rho[z == find_nearest(z, depthSlice)]
	ax2.scatter(depthX, depthY)
	ax2.plot(depthX, depthY)
	if AOI:
		ax2.set_xlim(aoi_x0, aoi_x1)
	else:
		ax2.set_xlim([np.amin(x), np.amax(x)])
	ax2.grid(which='both')
	ax2.set_ylabel('$\\rho$ [$\Omega .m$]', fontsize=fs)
	ax2.set_xlabel('Profile Distance [m]', fontsize=fs)
	ax2.set_title('Inverted Resistivity at z = %d' %depthSlice, fontsize=fs)
	ax2.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax2.tick_params(axis='both', which='minor', labelsize=fs*0.75)
	

	if save ==False:
		supT = fig.suptitle(plotTitle, x=0.445, y=1.04, fontsize=15)

	figTitle = plotTitle.replace('.dat','')
	figTitle = plotTitle.replace('.xyz','')

	if save:
		plt.savefig(figTitle, dpi=100, bbox_inches='tight')

	return ax0


##################
#Plot Percent Change in Inverted Data


def plotINVPercentChange(baseFile, secondFile, sliceLoc, depthSlice, plotTitle, method='absolute',
	save=False, fs=12):
	'''
	Takes raw output from 2 resd2dinv *.xyz files, calculated the difference, and plots the percent
	change from the base survey

	NOTE: can also pass a float value as second file to subtract out a background value

	baseFile: this is the initial or base survey
	secondFile: this is the second or follow up survey
	method: 'absolute' (default) or 'signed' or 'actual-signed'
	'''

	#####################
	#Load base survey file
	x_B, z_B, rho_B = loadRes2DINVResults(baseFile)

	#test if second file is background value or *.xyz file
	if isinstance(secondFile, str):
		#####################
		#Load second survey file
		x_2, z_2, rho_2 = loadRes2DINVResults(secondFile)
	else:
		x_2, z_2 = x_B, z_B
		rho_2 = np.full_like(rho_B, secondFile) #creates array like the base file and fills with specified value


	#####################
	#Calculate difference between two surveys
	if method == 'absolute':
		diffAbs = np.absolute(rho_B - rho_2)
		percentChange = np.absolute((diffAbs/rho_B)*100)
		minVal, maxVal = 0.0, np.nanmax(percentChange)
		cMap = 'viridis'
	if method == 'signed':
		diff_signed = rho_B - rho_2
		percentChange = ((diff_signed/rho_B)*100) * (-1.0) #negative one so that positive percents are increases
		pChangeMax = np.nanmax(np.absolute(percentChange)) #get max percent change value for setting cbar limits (i.e centered)
		minVal, maxVal = (-1.0*pChangeMax), pChangeMax
		cMap = 'seismic'
	if method == 'actual-signed':
		diff_signed = (rho_B - rho_2)*(-1.0) #negative one so that positive percents are increases
		percentChange = diff_signed
		pChangeMax = np.nanmax(np.absolute(percentChange)) #get max percent change value for setting cbar limits (i.e centered)
		minVal, maxVal = (-1.0*pChangeMax), pChangeMax
		cMap = 'seismic'


	#####################
	#Plot the percent change

	fig = plt.figure(figsize=(17,6))

	gs = gridspec.GridSpec(2,3, width_ratios=[1,5,0.15], height_ratios=[3,1], hspace=.42,wspace=0.28)

	#main plot
	ax0 = plt.subplot(gs[0,1])
	ctr = ax0.tricontourf(x_2, z_2, percentChange, levels=np.linspace(minVal, maxVal, 50), cmap=cMap) #levels to control 0 point
	#ax0.tricontour(x0,y0,z0, colors='gray', linewidths=0.75)
	ax0.invert_yaxis()
	ax0.axvline(x=sliceLoc)
	ax0.axhline(y=depthSlice)
	ax0.set_ylabel('Depth [m]', fontsize=fs)
	#ax0.set_xlabel('Profile Distance [m]', fontsize=fs)
	ax0.set_title('Percent Change in Inverted Resistivity', fontsize=fs)
	ax0.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax0.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	#color bar for main plot
	ax3 = plt.subplot(gs[:,2])
	cbar = fig.colorbar(ctr, cax=ax3, ticks=np.linspace(minVal, maxVal, 11)) #11 labels on color bar
	cbar.set_label('% Change in Resistivity', fontsize=fs)
	ax3.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax3.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	#depth profile
	ax1 = plt.subplot(gs[:,0])
	profileX, profileY = percentChange[x_2 == find_nearest(x_2, sliceLoc)], np.unique(z_2)
	ax1.scatter(profileX, profileY)
	ax1.plot(profileX, profileY)
	ax1.invert_yaxis()
	ax1.grid(which='both')
	ax1.set_ylabel('Depth [m]', fontsize=fs)
	ax1.set_xlabel('% Change', fontsize=fs)
	ax1.set_title('%% Change at %d m' %sliceLoc, fontsize=fs)
	ax1.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax1.tick_params(axis='both', which='minor', labelsize=fs*0.75)
	#if np.nanmax(z0[x0 == find_nearest(x0, sliceLoc)]) > 400:
	    #ax1.set_xlim([0,400])

	#depth slice
	ax2 = plt.subplot(gs[1,1])
	depthX, depthY = np.unique(x_2), percentChange[z_2 == find_nearest(z_2, depthSlice)]
	ax2.scatter(depthX, depthY)
	ax2.plot(depthX, depthY)
	ax2.grid(which='both')
	ax2.set_ylabel('% Change', fontsize=fs)
	ax2.set_xlabel('Profile Distance [m]', fontsize=fs)
	ax2.set_title('%% Change at z = %d' %depthSlice, fontsize=fs)
	ax2.set_xlim([np.amin(x_2), np.amax(x_2)])
	ax2.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax2.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	if save ==False:
		supT = fig.suptitle(plotTitle, x=0.445, y=1.04, fontsize=14)

	figTitle = plotTitle.replace('.xyz','')

	if save:
		plt.savefig(figTitle, dpi=150, bbox_inches='tight')

	return




##################
#Plot Percent Change in Raw Data


def plotRAWDataPercentChange(baseFile, secondFile, sliceLoc, nSliceVal, plotTitle, gridDimension=500, 
	showPoints=False, method='absolute', save=False, fs=12):
	'''
	Takes raw output from either 2 proSys *.csv files or 2 *.dat files, 
	calculates the difference, and plots the percent
	change from the base survey

	baseFile: this is the initial or base survey
	secondFile: this is the second or follow up survey
	method: 'absolute' (default) or 'signed'
	'''

	#####################
	#Load base survey file
	if baseFile.endswith('.csv'):
		Tx1_B, Tx2_B, Rx1_B, Rx2_B, rho_B, xLoc_B, n_B, aSpacing_B = loadProSysData(baseFile)
	elif baseFile.endswith('.dat'):
		xLoc_B, n_B, rho_B, aSpacing_B = loadRES2DModData(baseFile)

	#####################
	#Load second survey file
	if secondFile.endswith('.csv'):
		Tx1_2, Tx2_2, Rx1_2, Rx2_2, rho_2, xLoc_2, n_2, aSpacing_2 = loadProSysData(secondFile)
	elif secondFile.endswith('.dat'):
		xLoc_2, n_2, rho_2, aSpacing_2 = loadRES2DModData(secondFile)


	#####################
	#Calculate difference between two surveys and set color map
	if method == 'absolute':
		diffAbs = np.absolute(rho_B - rho_2)
		percentChange = np.absolute((diffAbs/rho_B)*100)
		cMap = 'viridis'
	if method == 'signed':
		diff_signed = rho_B - rho_2
		percentChange = ((diff_signed/rho_B)*100) * (-1.0) #negative one so that positive percents are increases
		cMap = 'seismic'


	#####################
	#Grid the Data

	#get max values
	max_x0, max_y0, max_z0 = np.amax(xLoc_B), np.amax(n_B), np.amax(percentChange)

	#get lateral step 
	dx = (max_x0 + aSpacing_B + 1) / gridDimension

	#establish grid
	xi = np.linspace(0, max_x0 + aSpacing_B + 1, gridDimension)
	yi = np.linspace(0, max_y0, gridDimension)

	#finally, grid the data
	zi = griddata((xLoc_B, n_B), percentChange, (xi[None,:], yi[:,None]), method='cubic')
	#get min and max values for plotting color bars etc.
	if method == 'absolute':
		minVal, maxVal = 0.0, np.nanmax(zi)
		zi[(zi != np.nan) & (zi < 0)] = 0 #set interpolated values below 0 to 0
	if method == 'signed':
		pChangeMax = np.nanmax(np.absolute(zi)) #get max percent change value for setting cbar limits (i.e centered)
		minVal, maxVal = (-1.0*pChangeMax), pChangeMax


	#####################
	#Plot the gridded data

	#get depth slice
	numNLevels = int(np.amax(yi)) #number of n levels
	dy = numNLevels / yi.size #y step of grid
	nSliceLoc = int(nSliceVal/dy) #array index for desired n level
	nSlice = zi[nSliceLoc,:] #get values at index

	#get 1D profile
	oneDProf = zi[:, int(sliceLoc/dx)] #get values at xloc

	#init figure
	fig = plt.figure(figsize=(17,6))
	gs = gridspec.GridSpec(2,3, width_ratios=[1,5,0.15], height_ratios=[3,1], hspace=.42, wspace=0.28)

	#main pseudosection
	ax0 = plt.subplot(gs[0,1])
	ctr = ax0.contourf(xi,yi,zi, cmap=cMap, levels=np.linspace(minVal, maxVal, 50))
	
	#draw contours
	ax0.contour(xi,yi,zi, 15, colors='gray', linewidths=0.75)

	if showPoints == True:
		ax0.scatter(xLoc_B, n_B, c='k', marker='+', s=35, alpha = 0.55)

	#main plot formatting 
	ax0.invert_yaxis()
	ax0.axvline(x=sliceLoc)
	ax0.axhline(y=nSliceVal)
	ax0.set_ylabel('n', fontsize=fs)
	#ax0.set_xlabel('Profile Distance [m]')
	ax0.set_title('Percent Change in Apparent Resistivity', fontsize=fs)
	ax0.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax0.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	#plot cBar
	ax3 = plt.subplot(gs[:,2])
	cbar = fig.colorbar(ctr, cax=ax3, ticks=np.linspace(minVal, maxVal, 11), format='%.1f') #11 labels on color bar
	cbar.set_label('% Change in Apparent Resistivity', fontsize=fs)
	ax3.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax3.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	#plot 1D profile at specified x location
	ax1 = plt.subplot(gs[:,0])
	ax1.plot(oneDProf, yi)
	ax1.scatter(percentChange[xLoc_B == sliceLoc], n_B[xLoc_B == sliceLoc])
	ax1.invert_yaxis()
	ax1.grid(which='both')
	ax1.set_ylabel('n', fontsize=fs)
	ax1.set_xlabel('% Change', fontsize=fs)
	ax1.set_title('%% Change at %d' %sliceLoc, fontsize=fs)
	ax1.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax1.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	#plot n slice at specified level
	ax2 = plt.subplot(gs[1,1])
	ax2.plot(xi, nSlice)
	ax2.scatter(xLoc_B[n_B == nSliceVal], percentChange[n_B == nSliceVal])
	ax2.set_ylim(np.nanmin(nSlice)*0.8, np.nanmax(nSlice)*1.2)
	ax2.set_xlim(np.nanmin(xi), np.nanmax(xi))
	ax2.grid(which='both')
	ax2.set_ylabel('% Change', fontsize=fs)
	ax2.set_xlabel('Profile Distance [m]', fontsize=fs)
	ax2.set_title('%% Change at n = %d' %nSliceVal, fontsize=fs)
	ax2.tick_params(axis='both', which='major', labelsize=fs*0.75)
	ax2.tick_params(axis='both', which='minor', labelsize=fs*0.75)

	if save == False:
		supT = fig.suptitle(plotTitle, x=0.445, y=1.04, fontsize=15)

	figTitle = plotTitle.replace('.dat','')
	figTitle = plotTitle.replace('.csv','')

	if save:
		plt.savefig(figTitle, dpi=150, bbox_inches='tight')

	return



##################
#UTILITY CODE
##################


def find_nearest(array, value):
	#function for finding nearest value to depth and slice locations
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx]