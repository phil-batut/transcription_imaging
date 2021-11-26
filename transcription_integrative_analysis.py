###########################################
## transcription_integrative_analysis.py ##
###########################################
##
##
##
## Philippe BATUT 
## Levine lab
## Lewis-Sigler Institute
## Princeton University 
## pbatut@princeton.edu 
## 
## November 2019 
##
##
##
##
## USAGE: 
##
##			python transcription_integrative_analysis.py [OPTIONS] Expression_data_directory Control_dataset_name
##
##
##
##



import os 
import cv2 
import sys 
import scipy 
import numpy 
import random 
import pandas as pd 
import seaborn as sns 
import scipy.stats as stats 
import matplotlib.pyplot as plt
from optparse import OptionParser 





##########################################################################################################################################################################
##########################################################################################################################################################################



## DEFINE FUNCTION to load expression data from file: 			

def load_expression_data(fileName): 
	file = open(fileName) 
	dataSet = file.readlines() 
	file.close() 
	Ntimes = len(dataSet[0].rstrip().split()) - 1 
	Nnuclei = len(dataSet) - 1 
	exprArray = numpy.zeros((Nnuclei, Ntimes), dtype = 'float64') 
	for line in dataSet[1:]: 
		line = line.rstrip().split() 
		nucleus = int(line[0]) 
		data = numpy.array(line[1:], dtype = 'float64') 
		exprArray[nucleus - 1, :] = data 
	return exprArray 




## DEFINE FUNCTION to load embryo orientations from file: 

def load_embryo_orientations(file_name): 
	orientations = {} 
	file = open(file_name) 
	for line in file: 
		line = line.rstrip().split() 
		embryo_name, embryo_orientation = line 
		orientations[embryo_name] = embryo_orientation 
	file.close() 
	return orientations 




## DEFINE FUNCTION to load expression & background data from database: 

def load_expression_datasets(inDir): 
	datasetNames = {} 
	inDir = inDir.rstrip('/') + '/' 
	allOrientations = {} 
	allCentroidsX = {} 
	allCentroidsY = {} 
	allSignal = {'MS2': {}, 'PP7': {}} 
	allBackground = {'MS2': {}, 'PP7': {}} 
	# Go through all data subdirectories: 
	for subDir in os.listdir(inDir): 
		if subDir.startswith('.'): continue 
		datasetNames[subDir] = [] 
		allCentroidsX[subDir] = [] 
		allCentroidsY[subDir] = []
		# Go through all data files: 
		for fileName in sorted(os.listdir(inDir + subDir)): 
			channel = None 
			# Skip irrelevant files: 
			if fileName.startswith('.'): continue
			if not fileName.endswith('.txt'): continue 
			# Get embryo orientation file: 
			if fileName.endswith('_Orientations.txt'): 
				allOrientations[subDir] = load_embryo_orientations(inDir + subDir + '/' + fileName) 
				continue 
			# Get dataset name: 
			datasetName = fileName.rsplit('_', 2)[0] 
			if datasetName not in datasetNames[subDir]: 
				datasetNames[subDir].append(datasetName) 
			# Load data from file: 
			exprArray = load_expression_data(inDir + subDir + '/' + fileName) 
			# Determine fluorescence channel: 
			if '_GFP_' in fileName: channel = 'MS2' 
			elif '_RFP_' in fileName: channel = 'PP7' 
			# Save to appropriate data structures: 
			if channel: 
				if subDir not in allSignal[channel]: 
					allSignal[channel][subDir] = [] 
					allBackground[channel][subDir] = [] 
				if fileName.endswith('_Signal.txt'): allSignal[channel][subDir].append(exprArray) 
				elif fileName.endswith('_Background.txt'): allBackground[channel][subDir].append(exprArray) 
			else: 
				if fileName.endswith('_Nucleus_X.txt'): allCentroidsX[subDir].append(exprArray) 
				elif fileName.endswith('_Nucleus_Y.txt'): allCentroidsY[subDir].append(exprArray) 
	# Return background & expression data: 
	return datasetNames, allSignal, allBackground, allCentroidsX, allCentroidsY, allOrientations 




## DEFINE FUNCTION to correct nucleus coordinates for embryo orientation: 

def correct_embryo_orientation(centroids_X, centroids_Y, orientations, image_width, image_height): 
	corr_centroids_X = {} 
	corr_centroids_Y = {} 
	# Go through all samples: 
	for sample_name in orientations: 
		corr_centroids_X[sample_name] = [] 
		corr_centroids_Y[sample_name] = [] 
		rep_list = sorted(orientations[sample_name].keys()) 
		rep_index = 0 
		# Go through all replicates: 
		for rep_name in rep_list: 
			orientation = orientations[sample_name][rep_name] 
			# If orientation is correct: 
			if orientation == '.': 
				corr_centroids_X[sample_name].append(centroids_X[sample_name][rep_index]) 
				corr_centroids_Y[sample_name].append(centroids_Y[sample_name][rep_index])
			# Horizontal flip: 
			elif orientation == 'H': 
				rep_centroids_X = centroids_X[sample_name][rep_index] 
				Nnuclei, Ntimes = rep_centroids_X.shape 
				new_centroids_X = numpy.zeros((Nnuclei, Ntimes)) 
				for n in range(Nnuclei): 
					for t in range(Ntimes): 
						old_X = rep_centroids_X[n, t] 
						new_X = image_width - old_X - 1 
						new_centroids_X[n, t] = new_X 
				corr_centroids_X[sample_name].append(new_centroids_X) 
				corr_centroids_Y[sample_name].append(centroids_Y[sample_name][rep_index]) 
			# Vertical flip: 
			elif orientation == 'V': 
				rep_centroids_Y = centroids_Y[sample_name][rep_index]  
				Nnuclei, Ntimes = rep_centroids_Y.shape 
				new_centroids_Y = numpy.zeros((Nnuclei, Ntimes)) 
				for n in range(Nnuclei):
					for t in range(Ntimes): 
						old_Y = rep_centroids_Y[n, t] 
						new_Y = image_height - old_Y - 1 
						new_centroids_Y[n, t] = new_Y 
				corr_centroids_Y[sample_name].append(new_centroids_Y) 
				corr_centroids_X[sample_name].append(centroids_X[sample_name][rep_index]) 
			# Horizontal & Vertical flips: 
			elif orientation == 'HV': 
				rep_centroids_X = centroids_X[sample_name][rep_index] 
				rep_centroids_Y = centroids_Y[sample_name][rep_index]  
				Nnuclei, Ntimes = rep_centroids_X.shape 
				new_centroids_X = numpy.zeros((Nnuclei, Ntimes)) 
				new_centroids_Y = numpy.zeros((Nnuclei, Ntimes)) 
				for n in range(Nnuclei): 
					for t in range(Ntimes): 
						old_X = rep_centroids_X[n, t] 
						old_Y = rep_centroids_Y[n, t] 
						new_X = image_width - old_X - 1 
						new_Y = image_height - old_Y - 1 
						new_centroids_X[n, t] = new_X 
						new_centroids_Y[n, t] = new_Y 
				corr_centroids_X[sample_name].append(new_centroids_X) 
				corr_centroids_Y[sample_name].append(new_centroids_Y) 
			rep_index += 1 
	# Return corrected nucleus coordinates: 
	return corr_centroids_X, corr_centroids_Y 




## DEFINE FUNCTION to fit Gaussian to background distribution: 

def fit_gaussian(background, backStart, backStop, Nbins, histThresh):
	backgroundDist = background[ : , backStart : backStop].flatten()
	minSignal = numpy.amin(backgroundDist)
	maxSignal = numpy.amax(backgroundDist)
	# Compute histogram: 
	distHist = numpy.histogram(
						backgroundDist,
						bins = Nbins,
						range = (minSignal, maxSignal),
						density = True
						)
	distDensities, distEdges = distHist
	# Threshold histogram:
	maxDensity = numpy.amax(distDensities)
	densityThresh = histThresh * maxDensity
	threshDensities = numpy.where(distDensities >= densityThresh, distDensities, 0)
	distMask = numpy.ones(backgroundDist.shape)
	for edgeIndex in range(Nbins):
		if threshDensities[edgeIndex] > 0: continue
		edgeLow, edgeHigh = distEdges[edgeIndex], distEdges[edgeIndex + 1]
		lowFilter = numpy.where(backgroundDist >= edgeLow, 1, 0)
		highFilter = numpy.where(backgroundDist < edgeHigh, 1, 0)
		combinedFilter = lowFilter * highFilter
		combinedFilter = numpy.where(combinedFilter == 1, 0, 1)
		distMask = distMask * combinedFilter
	if threshDensities[edgeIndex] == 0:
		highFilter = numpy.where(backgroundDist <= edgeHigh + 1, 1, 0)
		combinedFilter = lowFilter * highFilter
		combinedFilter = numpy.where(combinedFilter == 1, 0, 1)
		distMask = distMask * combinedFilter
	threshProfiles = backgroundDist[numpy.where(distMask > 0)]
	# Estimate Gaussian distribution parameters: 
	distMean = numpy.mean(threshProfiles)
	distStdDev = numpy.std(threshProfiles)
	# Return ditribution parameters: 
	return distMean, distStdDev





## DEFINE FUNCTION to normalize all samples to mean nuclear background: 

def background_normalization(allSignal, allBackground, ms2BackStart, ms2BackStop, pp7BackStart, pp7BackStop, Nbins, ms2_histThresh, pp7_histThresh, ms2_new_background, pp7_new_background): 
	
	backStart = {'MS2': ms2BackStart, 'PP7': pp7BackStart} 
	backStop = {'MS2': ms2BackStop, 'PP7': pp7BackStop} 
	histThresh = {'MS2': ms2_histThresh, 'PP7': pp7_histThresh} 
	newBack = {'MS2': ms2_new_background, 'PP7': pp7_new_background} 
	## Compute background means per replicate: 
	background_means = {} 
	# Go through both channels: 
	for channel in allBackground: 
		background_means[channel] = {} 
		# Go through all samples: 
		for sampleName in allBackground[channel]: 
			# Go through all replicates: 
			Nreps = len(allBackground[channel][sampleName]) 
			background_means[channel][sampleName] = numpy.zeros(Nreps, dtype = 'float64')  
			for repIndex in range(Nreps): 
				distMean, distStdDev = fit_gaussian( 
												allBackground[channel][sampleName][repIndex], 
												backStart[channel], 
												backStop[channel], 
												Nbins, 
												histThresh[channel] 
												) 
				background_means[channel][sampleName][repIndex] = distMean 
	## Compute target mean background values (if not specified): : 
	if not ms2_new_background: 
		allBackMeans = numpy.array([])  
		for sampleName in background_means['MS2']: allBackMeans = numpy.hstack(allBackMeans, background_means['MS2'][sampleName]) 
		ms2_new_background = numpy.mean(allBackMeans) 
	if not pp7_new_background: 
		allBackMeans = numpy.array([]) 
		for sampleName in background_means['PP7']: allBackMeans = numpy.hstack(allBackMeans, background_means['PP7'][sampleName]) 
		pp7_new_background = numpy.mean(allBackMeans) 
	## Compute normalization factors
	norm_factors = {} 
	for channel in background_means: 
		norm_factors[channel] = {} 
		for sampleName in background_means[channel]: 
			norm_factors[channel][sampleName] = newBack[channel] / background_means[channel][sampleName] 
	## Normalize all data: 
	newSignal = {} 
	newBackground = {}
	for channel in norm_factors: 
		newSignal[channel] = {} 
		newBackground[channel] = {} 
		for sampleName in norm_factors[channel]: 
			newSignal[channel][sampleName] = [] 
			newBackground[channel][sampleName] = [] 
			Nreps = len(allBackground[channel][sampleName]) 
			for repIndex in range(Nreps): 
				norm_factor = norm_factors[channel][sampleName][repIndex] 
				newBackground[channel][sampleName].append(allBackground[channel][sampleName][repIndex] * norm_factor) 
				newSignal[channel][sampleName].append(allSignal[channel][sampleName][repIndex] * norm_factor) 
	## Return normalized data: 
	return newSignal, newBackground, norm_factors 




## DEFINE FUNCTION to define signal threshold based on desired FDR: 

def set_signal_threshold(distMean, distStdDev, fdr):
	percentile = 1 - fdr
	signalThresh = stats.norm.ppf(percentile, distMean, distStdDev)
	return signalThresh




## DEFINE FUNCTION to plot background histogram (with fitted Gaussian): 

def plot_background_histogram(background, plotName, title, backMean, backStd, signalThresh, Nbins, histColor):
	# Get axis dimensions: 
	minSignal = numpy.amin(background)
	maxSignal = numpy.amax(background)
	# Create histogram: 
	plt.figure(1, (10.5, 7))
	plt.title(title)
	backValues = background.flatten()
	hDens, hBins, hPat = plt.hist(
							backValues,
							bins = Nbins,
							range = (minSignal, maxSignal),
							density = True,
							histtype = 'stepfilled',
							color = histColor
							)
	# Plot fitted Gaussian: 
	signalScale = numpy.linspace(
							minSignal,
							maxSignal,
							Nbins + 1
							)
	gaussianFit = stats.norm.pdf(
							signalScale,
							backMean,
							backStd
							)
	plt.plot(
							signalScale,
							gaussianFit,
							linestyle = 'dashed',
							linewidth = 1,
							color = 'black'
							)
	# Add signal threshold line: 
	maxDensity = numpy.amax(hDens)
	plt.vlines(
							signalThresh,
							0,
							maxDensity,
							linestyle = 'dashed',
							linewidth = 1,
							color = 'black'
							)
	# Save figure to file: 
	plt.savefig(plotName, format = 'pdf', dpi = 300)
	plt.close()
	return




## DEFINE FUNCTION to subtract background from expression signal: 

def subtract_background(all_profiles, expr_thresh): 
	# Create new data structure: 
	subtracted_profiles = {} 
	# Process both channels: 
	for channel in ['MS2', 'PP7']: 
		subtracted_profiles[channel] = {} 
		# Process all samples: 
		for sampleName in all_profiles[channel].keys(): 
			subtracted_profiles[channel][sampleName] = [] 
			# Go through all replicates: 
			for replicate in range(len(all_profiles[channel][sampleName])): 
				# Subtract background: 
				new_profiles = all_profiles[channel][sampleName][replicate] - expr_thresh[channel] 
				new_profiles = numpy.where(new_profiles <= 0, 0, new_profiles) 
				subtracted_profiles[channel][sampleName].append(new_profiles) 
	# Return subtracted profiles: 
	return subtracted_profiles 




## DEFINE FUNCTION to compute the center of gravity of an expression domain:

def center_of_gravity(centroids, masses):
    center_gravity = numpy.sum(centroids * masses) / numpy.sum(masses)
    return center_gravity 




## DEFINE FUNCTION to compute center of gravity for all expression domains: 

def all_centers_of_gravity(centroids_X, time_point, all_profiles, ref_channel): 
	centers_of_gravity = {} 
	for sampleName in centroids_X.keys(): 
		centers_of_gravity[sampleName] = [] 
		Nreps = len(centroids_X[sampleName]) 
		for repIndex in range(Nreps): 
			centroids = centroids_X[sampleName][repIndex][ : , time_point] 
			masses = numpy.mean(all_profiles[ref_channel][sampleName][repIndex], 1) 
			CoG = center_of_gravity(centroids, masses) 
			centers_of_gravity[sampleName].append(CoG) 
	return centers_of_gravity 




## DEFINE FUNCTION to select nuclei within domain of interest: 

def nuclei_in_domains(centroids_X, time_point, centers_of_gravity, domain_width, domain_offset): 
	half_domain_width = int(domain_width / 2) 
	selected_nuclei = {} 
	for sampleName in centroids_X.keys(): 
		selected_nuclei[sampleName] = [] 
		Nreps = len(centroids_X[sampleName]) 
		for repIndex in range(Nreps): 
			centroids = centroids_X[sampleName][repIndex][ : , time_point] 
			CoG = centers_of_gravity[sampleName][repIndex] + domain_offset 
			left_boundary = CoG - half_domain_width  
			right_boundary = CoG + half_domain_width 
			select_left = numpy.where(centroids >= left_boundary, 1, 0) 
			select_right = numpy.where(centroids <= right_boundary, 1, 0) 
			select_both = select_left * select_right 
			selection = numpy.where(select_both == 1)[0] 
			selected_nuclei[sampleName].append(selection) 
	return selected_nuclei 




## DEFINE FUNCTION to extract a defined subset of nuclei: 

def filter_nuclei(all_profiles, selected_nuclei): 
	filtered_set = {} 
	for channel in all_profiles.keys(): 
		filtered_set[channel] = {} 
		for sampleName in all_profiles[channel].keys(): 
			filtered_set[channel][sampleName] = [] 
			Nreps = len(all_profiles[channel][sampleName]) 
			for repIndex in range(Nreps): 
				selection = selected_nuclei[sampleName][repIndex] 
				data_in = all_profiles[channel][sampleName][repIndex] 
				data_out = data_in[selection] 
				filtered_set[channel][sampleName].append(data_out) 
	return filtered_set 




## DEFINE FUNCTION to select nuclei active in reference channel: 

def select_active_nuclei(all_profiles, refChannel, refMin): 
	# Identify non-reference channel: 
	if refChannel == 'PP7': nonRefChannel = 'MS2' 
	elif refChannel == 'MS2': nonRefChannel = 'PP7' 
	else: print('\n\t(!) Ref channel?? \n'); sys.exit() 
	# Filter nuclei on reference channel: 
	filter_profiles = {'MS2': {}, 'PP7': {}} 
	# Go through all samples: 
	for sampleName in all_profiles[refChannel]: 
		filter_profiles[refChannel][sampleName] = [] 
		filter_profiles[nonRefChannel][sampleName] = [] 
		# Go through all replicates: 
		for replicate in range(len(all_profiles[refChannel][sampleName])): 
			meanExpr = numpy.mean(all_profiles[refChannel][sampleName][replicate], 1) 
			activeNucleiIndexes = numpy.where(meanExpr >= refMin)[0] 
			filter_profiles[refChannel][sampleName].append(all_profiles[refChannel][sampleName][replicate][activeNucleiIndexes, :]) 
			filter_profiles[nonRefChannel][sampleName].append(all_profiles[nonRefChannel][sampleName][replicate][activeNucleiIndexes, :])
	# Return filtered profiles: 
	return filter_profiles 




## DEFINE FUNCTION to determine activity state of individual nucleus: 

def nucleus_activity_states(profiles, window): 
	Nnuclei, Ntimes = profiles.shape 
	state_profiles = numpy.where(profiles > 0, 1, 0) 
	window_profiles = numpy.zeros((Nnuclei, (Ntimes - window + 1)), dtype = 'int64') 
	for k in range(window, Ntimes+1): window_profiles[:, (k - window)] = numpy.sum(state_profiles[: , (k - window) : k], 1) 
	window_profiles = numpy.where(window_profiles >= 1, 1, 0) 
	return window_profiles 




## DEFINE FUNCTION to compute nucleus activity status over time window for all datasets: 

def nucleus_activity_states_alldata(all_profiles, window): 
	all_states = {} 
	# Go through both channels: 
	for channel in ['MS2', 'PP7']: 
		all_states[channel] = {} 
		# Go through all samples: 
		for sampleName in all_profiles[channel]: 
			all_states[channel][sampleName] = [] 
			# Go through all replicates: 
			for replicate in range(len(all_profiles[channel][sampleName])): 
				# Determine activity state of all nuclei: 
				nucleus_states = nucleus_activity_states(all_profiles[channel][sampleName][replicate], window) 
				all_states[channel][sampleName].append(nucleus_states) 
	# Return all nucleus activity states: 
	return all_states 




## DEFINE FUNCTION to determine activity state of individual nucleus (Cumulative): 

def nucleus_activity_states_cumulative(profiles): 
	Nnuclei, Ntimes = profiles.shape 
	state_profiles = numpy.where(profiles > 0, 1, 0) 
	cumulative_profiles = numpy.zeros(profiles.shape) 
	for N in range(Nnuclei): 
		active_times = numpy.where(state_profiles[N, : ] == 1)[0] 
		if active_times.shape[0] == 0: continue 
		first_activation = active_times[0] 
		for t in range(first_activation, Ntimes): cumulative_profiles[N, t] = 1 
	return cumulative_profiles 




## DEFINE FUNCTION to compute nucleus activity status for all datasets (Cumulative): 

def nucleus_activity_states_cumulative_alldata(all_profiles): 
	all_states = {} 
	# Go through both channels: 
	for channel in ['MS2', 'PP7']: 
		all_states[channel] = {} 
		# Go through all samples: 
		for sampleName in all_profiles[channel]: 
			all_states[channel][sampleName] = [] 
			# Go through all replicates: 
			for replicate in range(len(all_profiles[channel][sampleName])): 
				# Determine activity state of all nuclei: 
				nucleus_states = nucleus_activity_states_cumulative(all_profiles[channel][sampleName][replicate]) 
				all_states[channel][sampleName].append(nucleus_states) 
	# Return all nucleus activity states: 
	return all_states 




## DEFINE FUNCTION to plot nucleus activity profiles for all individual replicates, per sample: 

def plot_nucleus_activity_replicates_per_sample(datasetNames, nucleus_states, stack_time, window, offset, fdr, plotName, title): 
	# Get dataset dimensions: 
	Nembryos = len(nucleus_states) 
	Ntimes = nucleus_states[0].shape[1] 
	timeScale = numpy.array(range(Ntimes)) * stack_time / 60
	timePadding = (window + offset) * stack_time / 60
	timeScale += timePadding 
	# Define color map: 
	color_palette_name = 'hsv' 	# 'Greys' 
	color_palette = plt.cm.get_cmap(color_palette_name, 200) 
	colors = range(20, 190, round(170 / Nembryos)) 
	# Generate figure:
	plt.figure(1,
			(10, 6),
			frameon = False
			)
	plt.gca().spines['top'].set_visible(False) 
	plt.gca().spines['right'].set_visible(False) 
	# Go through all replicates & plot active nuclei over time: 
	for repIndex in range(Nembryos): 
		datasetName = datasetNames[repIndex].replace ('_', ' ')  
		embryo_states = nucleus_states[repIndex] 
		Nnuclei = embryo_states.shape[0] 
		nuclei_counts = numpy.sum(embryo_states, 0) 
		nuclei_fractions = nuclei_counts / Nnuclei * 100 
		plt.plot( 
				timeScale, 
				nuclei_fractions, 
				color = tuple(color_palette(colors[repIndex])[:3]), 
				linewidth = 1, 
				label = '%s (n=%s)' %(datasetName, Nnuclei) 
				) 
	# Add labels & legend:  
	plt.title(title)
	plt.xlabel('Time after Metaphase 13 (min)')
	plt.ylabel('Percentage Active Nuclei (FDR = %.2e)' % fdr) 
	plt.legend(loc = 'upper left')
	plt.savefig(plotName, format = 'pdf', dpi = 300)
	plt.close()
	return





## DEFINE FUNCTION to compute active fraction of nuclei across all samples: 

def active_nuclei_counts(all_states): 
	Nembryos = {} 
	Nnuclei = {} 
	active_nuclei = {} 
	# Go through all samples & compute active fraction: 
	for sampleName in all_states: 
		Ntimes = all_states[sampleName][0].shape[1] 
		Nembryos[sampleName] = len(all_states[sampleName]) 
		Nnuclei[sampleName] = [] 
		active_nuclei[sampleName] = [] 
		# Go through all replicates: 
		for replicate in range(Nembryos[sampleName]): 
			Nnuclei[sampleName].append(all_states[sampleName][replicate].shape[0]) 
			active_nuclei[sampleName].append(numpy.sum(all_states[sampleName][replicate], 0)) 
		active_nuclei[sampleName] = numpy.vstack(active_nuclei[sampleName]) 
	# Return statistics: 
	return Nembryos, Nnuclei, active_nuclei 




## DEFINE FUNCTION to plot nucleus activity profiles for all individual replicates: 

def plot_nucleus_activity_all_replicates(all_states, stack_time, window, offset, fdr, plotName, col, title): 
	# Compute active fraction of nuclei across all samples: 
	Nembryos, Nnuclei, active_nuclei = active_nuclei_counts(all_states) 
	# Gather samples info & stats: 
	samples_list = sorted(active_nuclei.keys()) 
	Ntimes = active_nuclei[samples_list[0]].shape[1] 
	timeScale = numpy.array(range(Ntimes)) * stack_time / 60 
	timePadding = (window + offset) * stack_time / 60
	timeScale += timePadding
	colors = col.split(',')
	colorIndex = 0
	# Generate figure: 
	plt.figure(1, 
			(10, 6), 
			frameon = False 
			) 
	plt.gca().spines['top'].set_visible(False) 
	plt.gca().spines['right'].set_visible(False) 
	# Plot data for each sample: 
	for sampleName in samples_list: 
		first_rep = True 
		nucleus_totals = numpy.array([Nnuclei[sampleName]] * Ntimes).swapaxes(0, 1) 
		active_fraction = active_nuclei[sampleName] / nucleus_totals * 100 
		number_reps = active_fraction.shape[0] 
		for rep_index in range(number_reps): 
			fraction = active_fraction[rep_index, : ] 
			if first_rep: 
				plt.plot( 
						timeScale, 
						fraction, 
						color = colors[colorIndex], 
						linewidth = 1,
						label = '%s (N = %s)' %(sampleName, Nembryos[sampleName]) 
						) 
			else: 
				plt.plot( 
						timeScale, 
						fraction, 
						color = colors[colorIndex], 
						linewidth = 1,
						) 
			first_rep = False 
		colorIndex += 1 
	# Add labels & legend:  
	plt.title(title)
	plt.xlabel('Time after Metaphase 13 (min)')
	plt.ylabel('Percentage Active Nuclei (FDR = %.2e)' % fdr) 
	plt.legend(loc = 'upper left')
	plt.savefig( 
			plotName, 
			format = 'pdf', 
			dpi = 300 
			)
	plt.close()
	return
	
	
	

## DEFINE FUNCTION to plot nucleus activity profiles for all genes: 

def plot_nucleus_activity_profiles(all_states, stack_time, window, offset, fdr, critic_fract, plotName, col, title): 
	# Compute active fraction of nuclei across all samples: 
	Nembryos, Nnuclei, active_nuclei = active_nuclei_counts(all_states) 
	# Gather samples info & stats: 
	samples_list = sorted(active_nuclei.keys()) 
	Ntimes = active_nuclei[samples_list[0]].shape[1] 
	timeScale = numpy.array(range(Ntimes)) * stack_time / 60 
	timePadding = (window + offset) * stack_time / 60
	timeScale += timePadding
	colors = col.split(',')
	colorIndex = 0
	critic_fract_value = [] 
	# Generate figure: 
	plt.figure(1, 
			(10, 6), 
			frameon = False 
			) 
	plt.gca().spines['top'].set_visible(False) 
	plt.gca().spines['right'].set_visible(False) 
	# Plot data for each sample: 
	for sampleName in samples_list: 
		nucleus_totals = numpy.array([Nnuclei[sampleName]] * Ntimes).swapaxes(0, 1) 
		active_fraction = active_nuclei[sampleName] / nucleus_totals * 100 
		mean_fraction = numpy.mean(active_fraction, 0) 
		sem_fraction = stats.sem(active_fraction, 0) 
		total_nuclei = sum(Nnuclei[sampleName]) 
		# Compute critical fraction value(Optional): 
		if critic_fract: critic_fract_value.append(numpy.amax(mean_fraction)) 
		# Plot data: 
		plt.plot( 
				timeScale, 
				mean_fraction, 
				color = colors[colorIndex], 
				linewidth = 1,
				label = '%s (N=%s, n=%s)' %(sampleName.replace('_', ' '), Nembryos[sampleName], total_nuclei) 
				) 
		plt.fill_between( 
				timeScale, 
				mean_fraction - sem_fraction, 
				mean_fraction + sem_fraction, 
				alpha = 0.15, 
				edgecolor = 'none', 
				facecolor = colors[colorIndex] 
				) 
		colorIndex += 1 
	if critic_fract: 
		critic_fract_value = max(critic_fract_value) * critic_fract 
		plt.plot(
				[timeScale[0], timeScale[-1]], 
				[critic_fract_value, critic_fract_value], 
				color = 'grey', 
				linewidth = 0.75, 
				linestyle = 'dashed' 
				) 
	plt.title(title)
	plt.xlabel('Time after Metaphase 13 (min)')
	plt.ylabel('Percentage Active Nuclei (FDR = %.2e)' % fdr) 
	plt.legend(loc = 'upper left') 
	plt.savefig(plotName, format = 'pdf', dpi = 300) 
	plt.close()
	return




## DEFINE FUNCTION to compute running mean over time window:

def running_average(exprArray, window):
    Nreps, Ntimes = exprArray.shape
    runningMean = numpy.zeros((Nreps, (Ntimes - window + 1)), dtype = 'float64')
    for k in range(window, Ntimes + 1): runningMean[ : , (k - window)] = numpy.mean(exprArray[ : , (k - window) : k], 1)
    return runningMean




## DEFINE FUNCTION to compute mean intensities per embryo over active nuclei: 

def mean_intensities_per_embryo(all_profiles, intensity_thresh, window): 
	Nembryos = {} 
	Nnuclei = {} 
	mean_intensities = {} 
	# Go through all samples & compute active fraction: 
	for sampleName in all_profiles: 
		Ntimes = all_profiles[sampleName][0].shape[1] 
		Nembryos[sampleName] = len(all_profiles[sampleName]) 
		Nnuclei[sampleName] = [] 
		mean_intensities[sampleName] = [] 
		# Go through all replicates:
		for replicate in range(Nembryos[sampleName]): 
			Nnuclei[sampleName].append(all_profiles[sampleName][replicate].shape[0]) 
			rep_means = numpy.zeros(Ntimes, dtype = 'float') 
			for t in range(Ntimes): 
				intensities = all_profiles[sampleName][replicate][ : , t]  
				active_nuclei = numpy.where(intensities > intensity_thresh)[0] 
				active_intensities = intensities[active_nuclei] 
				number_active = active_intensities.shape[0] 
				if number_active == 0: mean_intensity = 0 
				else: mean_intensity = numpy.mean(active_intensities) 
				rep_means[t] = mean_intensity 	
			mean_intensities[sampleName].append(rep_means) 
		# Convert to Numpy array: 
		mean_intensities[sampleName] = numpy.vstack(mean_intensities[sampleName]) 
		# compute running mean across window: 
		mean_intensities[sampleName] = running_average(mean_intensities[sampleName], window) 
	# Return statistics:
	return Nembryos, Nnuclei, mean_intensities 




## DEFINE FUNCTION to plot transcription intensity profiles over time: 

def plot_transcription_intensities_profiles(all_profiles, intensity_thresh, window, normalize, stack_time, offset, plotName, col, title): 
	# Compute mean intensities per embryo over active nuclei: 
	intensity_data = mean_intensities_per_embryo( 
										all_profiles, 
										intensity_thresh, 
										window 
										) 
	Nembryos, Nnuclei, mean_intensities = intensity_data 
	# Gather samples info & stats: 
	samples_list = sorted(mean_intensities.keys()) 
	Ntimes = mean_intensities[samples_list[0]].shape[1] 
	timeScale = numpy.array(range(Ntimes)) * stack_time / 60 
	timePadding = (window + offset) * stack_time / 60
	timeScale += timePadding
	colors = col.split(',')
	colorIndex = 0
	# Generate figure: 
	plt.figure(1, 
			(10, 6), 
			frameon = False 
			) 
	plt.gca().spines['top'].set_visible(False) 
	plt.gca().spines['right'].set_visible(False) 
	# Plot data for each sample: 
	for sampleName in samples_list: 
		mean_intensity = numpy.mean(mean_intensities[sampleName], 0) 
		if normalize: norm_mean_intensities = mean_intensities[sampleName] / numpy.amax(mean_intensity) * 100 
		else: norm_mean_intensities = mean_intensities[sampleName] 
		mean_intensity = numpy.mean(norm_mean_intensities, 0) 
		sem_intensity = stats.sem(norm_mean_intensities, 0) 
		total_nuclei = sum(Nnuclei[sampleName]) 
		plt.plot( 
				timeScale, 
				mean_intensity, 
				color = colors[colorIndex], 
				linewidth = 1, 
				label = '%s (N=%s, n=%s)' %(sampleName.replace('_', ' '), Nembryos[sampleName], total_nuclei) 
				) 
		plt.fill_between( 
				timeScale, 
				mean_intensity - sem_intensity, 
				mean_intensity + sem_intensity, 
				alpha = 0.15, 
				edgecolor = 'none', 
				facecolor = colors[colorIndex] 
				) 
		colorIndex += 1 
	plt.title(title)
	plt.xlabel('Time after Metaphase 13 (min)')
	if normalize: plt.ylabel('Mean Intensity over Active Nuclei (% Max)') 
	else: plt.ylabel('Mean Intensity over Active Nuclei (FU)') 
	plt.legend(loc = 'upper left') 
	plt.savefig(plotName, format = 'pdf', dpi = 300) 
	plt.close()
	return




## DEFINE FUNCTION to compute total transcriptional output per nucleus: 

def total_output_per_nucleus(all_profiles, profiles_start, profiles_stop): 
	total_output = {} 
	# Go through both channels: 
	for channel in all_profiles: 
		if channel not in total_output: total_output[channel] = {} 
		# Go through all samples: 
		for sampleName in all_profiles[channel]: 
			total_output[channel][sampleName] = [] 
			# Go through all replicates 
			for replicate in range(len(all_profiles[channel][sampleName])): 
				# Compute total transcriptional output: 
				rep_data = all_profiles[channel][sampleName][replicate][ : , profiles_start : profiles_stop] 
				output = numpy.sum(rep_data, 1) 
				total_output[channel][sampleName].append(output) 
	# Return statistics: 
	return total_output 




## DEFINE FUNCTION to compute total transcriptional output per embryo: 

def total_output_per_embryo(total_output): 
	embryo_output = {} 
	# Go through both channels:
	for channel in total_output: 
		if channel not in embryo_output: embryo_output[channel] = {} 
		# Go through all samples: 
		for sampleName in total_output[channel]: 
			embryo_output[channel][sampleName] = [] 
			# Go through all replicates: 
			for replicate in range(len(total_output[channel][sampleName])): 
				# Compute total transcriptional output per embryo: 
				rep_data = total_output[channel][sampleName][replicate] 
				mean_output = numpy.mean(rep_data) 
				embryo_output[channel][sampleName].append(mean_output) 
	# Return statistics: 
	return embryo_output 




## DEFINE FUNCTION to write total output to file: 

def write_total_output_per_embryo(embryo_output): 
	# Create output territory: 
	out_dir_name = 'Total_Output' 
	if out_dir_name not in os.listdir(os.getcwd()): out_dir = os.mkdir(out_dir_name) 
	# Go through all samples: 
	out_dir_name += '/' 
	for sampleName in embryo_output['MS2']: 
		# Create file: 
		file_name = out_dir_name + sampleName + '_TotalOutput.txt' 
		out_file = open(file_name, 'w') 
		header = '\t'.join(['Replicate', 'MS2', 'PP7']) 
		out_file.write(header + '\n') 
		# Go through all replicates:
		for replicate in range(len(embryo_output['MS2'][sampleName])): 
			rep_index = str(replicate + 1).rjust(3, '0') 
			ms2_output = str(embryo_output['MS2'][sampleName][replicate]) 
			pp7_output = str(embryo_output['PP7'][sampleName][replicate]) 
			rep_name = 'Rep_' + rep_index 
			rep_line = '\t'.join([rep_name, ms2_output, pp7_output]) + '\n' 
			out_file.write(rep_line) 
		out_file.close() 
	return 




## DEFINE FUNCTION to compute spatial activity profiles along A-P axis: 

def compute_spatial_activity_profiles(total_output, centroids_X, time_point, selected_nuclei, centers_of_gravity, domain_width, domain_offset, spatial_bins, overlapping_bins): 
	## Compute spatial bin boundaries (relative to center of gravity): 
	bin_mids = [] 
	bin_egdes = [] 
	half_domain_width = int(domain_width / 2) 
	left_edge = - half_domain_width + domain_offset 
	if overlapping_bins: 
		bin_width = spatial_bins 
		right_edge = -10000
		while right_edge <= half_domain_width: 
			right_edge = left_edge + bin_width 
			midpoint = int((left_edge + right_edge) / 2) 
			bin_mids.append(midpoint) 
			bin_egdes.append([left_edge, right_edge]) 
			left_edge += 1  
	else: 
		bin_width = int(domain_width / spatial_bins) 
		for bin_i in range(spatial_bins - 1): 
			right_edge = left_edge + bin_width 
			midpoint = int((left_edge + right_edge) / 2) 
			bin_mids.append(midpoint) 
			bin_egdes.append([left_edge, right_edge]) 
			left_edge = right_edge 
	last_mid = int((left_edge + half_domain_width) / 2) 
	bin_mids.append(last_mid)
	bin_egdes.append([left_edge, half_domain_width + 1]) 
	bin_mids = numpy.array(bin_mids) 
	bin_egdes = numpy.array(bin_egdes) 
	number_bins = bin_mids.shape[0] 
	## Compute spatial transcription activity profiles for each sample/replicate: 
	spatial_profiles = {} 
	# Go through both channels: 
	for channel in total_output: 
		spatial_profiles[channel] = {} 
		# Go through all samples: 
		for sample_name in total_output[channel]:
			spatial_profiles[channel][sample_name] = [] 
			# Go through all replicates: 
			number_reps = len(total_output[channel][sample_name]) 
			for rep_index in range(number_reps): 
				spatial_profile = numpy.zeros(number_bins, dtype = 'float') 
				CoG = centers_of_gravity[sample_name][rep_index] 
				nuc_subset = selected_nuclei[sample_name][rep_index] 
				nuc_centroids = centroids_X[sample_name][rep_index][nuc_subset , time_point] 
				nuc_outputs = total_output[channel][sample_name][rep_index] 
				# Compute mean output for each bin: 
				for bin_i in range(number_bins): 
					bin_start = CoG + bin_egdes[bin_i, 0] 
					bin_stop = CoG + bin_egdes[bin_i, 1] 
					select_left = numpy.where(nuc_centroids >= bin_start, 1, 0) 
					select_right = numpy.where(nuc_centroids < bin_stop, 1, 0) 
					select_both = select_left * select_right 
					selection = numpy.where(select_both == 1)[0] 
					number_nuclei = selection.shape[0] 
					if number_nuclei == 0: spatial_profile[bin_i] = 0 
					else: 
						nuclei_in_bin = nuc_outputs[selection] 
						integrated_output = numpy.mean(nuclei_in_bin) 
						spatial_profile[bin_i] = integrated_output 
				# Append replicate profile to data structure: 
				spatial_profiles[channel][sample_name].append(spatial_profile) 
			# Convert list to array: 
			spatial_profiles[channel][sample_name] = numpy.array(spatial_profiles[channel][sample_name]) 
	# Return spatial profiles for all datasets: 
	return spatial_profiles, bin_mids 




## DEFINE FUNCTION to plot spatial activity profiles along A-P axis:

def plot_spatial_activity_profiles(spatial_profiles, bin_mids, pixel_size, normalize_profiles, plot_name, col, plot_title): 

	# Generate figure:
	plt.figure(1, 
			#(10, 10), 
			(10, 6), 
			frameon = False 
			) 
	plt.gca().spines['top'].set_visible(False) 
	plt.gca().spines['right'].set_visible(False) 
	colors = col.split(',')
	colorIndex = 0 
	# Plot data for each sample: 
	bin_mids = bin_mids * pixel_size 
	samples_list = sorted(spatial_profiles.keys()) 
	for sample_name in samples_list: 
		rep_outputs = spatial_profiles[sample_name] 
		number_reps = rep_outputs.shape[0] 
		mean_output = numpy.mean(rep_outputs, 0) 
		# Normalize spatial profiles to maximum (Optional): 
		if normalize_profiles: 
			profile_max = numpy.amax(mean_output) 
			rep_outputs = rep_outputs / profile_max 
			mean_output = numpy.mean(rep_outputs, 0)
		sem_output = stats.sem(rep_outputs, 0)
		plt.plot( 
				bin_mids, 
				mean_output, 
				color = colors[colorIndex], 
				linewidth = 1, 
				label = '%s (N=%s)' %(sample_name.replace('_', ' '), number_reps) 
				) 
		plt.fill_between(
				bin_mids, 
				mean_output - sem_output, 
				mean_output + sem_output, 
				alpha = 0.15,
				edgecolor = 'none', 
				facecolor = colors[colorIndex]
				) 
		colorIndex += 1 
	# Add axes, title, legend: 
	plt.title(plot_title) 
	plt.xlabel('Position along A-P axis (microns)') 
	plt.ylabel('Total output per nucleus') 
	plt.legend(loc = 'upper left')
	plt.savefig( 
			plot_name, 
			format = 'pdf',
			dpi = 300 
			) 
	plt.close() 
	return 




## DEFINE FUNCTION to plot total output scatterplots for each sample: 

def plot_total_output_scatterplot(total_output, plot_name, plot_title): 
	
	# Generate figure:
	plt.figure(1,
			(10, 10),
			frameon = False
			)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['right'].set_visible(False)
	# Generate scatterplot: 
	total_output['MS2'] = numpy.hstack(total_output['MS2']) 
	total_output['PP7'] = numpy.hstack(total_output['PP7']) 
	Pearson = stats.pearsonr(total_output['MS2'], total_output['PP7']) 
	plt.scatter( 
			total_output['PP7'], 
			total_output['MS2'],
			s = 25, 
			c = 'darkred', 
			alpha = 0.6 
			) 
	# Add axes & title: 
	plt.title(plot_title) 
	plt.xlabel('Total output per nucleus (PP7)') 
	plt.ylabel('Total output per nucleus (MS2)') 
	plt.savefig( 
			plot_name, 
			format = 'pdf',
			dpi = 300 
			) 
	plt.close() 
	return 




## DEFINE FUNCTION to compute fraction of time each nucleus is active: 

def fraction_time_active_per_nucleus(all_profiles): 
	fraction_time_active = {} 
	# Go through both channels: 
	for channel in all_profiles: 
		if channel not in fraction_time_active: fraction_time_active[channel] = {} 
		# Go through all samples: 
		for sampleName in all_profiles[channel]: 
			fraction_time_active[channel][sampleName] = [] 
			# Go through all replicates: 
			for replicate in range(len(all_profiles[channel][sampleName])): 
				# Compute fraction of time active: 
				fraction_time = numpy.where(all_profiles[channel][sampleName][replicate] > 0, 1, 0) 
				fraction_time = numpy.mean(fraction_time, 1) * 100 
				fraction_time_active[channel][sampleName].append(fraction_time) 
	# Return statistics: 
	return fraction_time_active 




## DEFINE FUNCTION to compute critical activation time for each embryo: 

def compute_critical_activation_time(all_states, control_set, critical_fraction): 
	activation_times = {} 
	for channel in all_states.keys(): 
		Nembryos, Nnuclei, active_nuclei = active_nuclei_counts(all_states[channel]) 
		Nnuclei_control = Nnuclei[control_set] 
		active_nuclei_control = active_nuclei[control_set] 
		active_fraction_control = [] 
		for rep_index in range(active_nuclei_control.shape[0]): 
			active_fraction_control.append(active_nuclei_control[rep_index] / Nnuclei_control[rep_index]) 
		active_fraction_control = numpy.array(active_fraction_control) 
		max_fraction = numpy.max(numpy.mean(active_fraction_control, 0)) 
		threshold_fraction = critical_fraction * max_fraction 
		N_timepoints = active_fraction_control.shape[1] 
		activation_times[channel] = {} 
		for sampleName in active_nuclei.keys(): 
			activation_times[channel][sampleName] = [] 
			number_reps = active_nuclei[sampleName].shape[0] 
			for rep_index in range(number_reps): 
				active_counts = active_nuclei[sampleName][rep_index] 
				active_fraction = active_counts / Nnuclei[sampleName][rep_index] 
				if numpy.max(active_fraction) < threshold_fraction: critical_time = N_timepoints 
				else: critical_time = numpy.where(active_fraction >= threshold_fraction)[0][0] 
				activation_times[channel][sampleName].append(critical_time) 
	return activation_times 




## DEFINE FUNCTION to generate categorical scatterplot of gene activation times per embryo: 

def generate_activation_time_scatterplot(plot_data, channel, plot_name, col, horizontal, reverse, stack_time, offset, title, y_label):
	# Create formatted data structure: 
	data_frame = {
			'Genotype': [], 
			'ActivTime': [], 
			'Color': [] 
			} 
	colorIndex = 0 
	colors = col.split(',') 
	colors = colors[ : int(len(colors) / 2)] 
	channels_list = sorted(plot_data.keys()) 
	if reverse: 
		colors = colors[::-1] 
		channels_list = channels_list[::-1] 
	sample_labels = [] 
	plotting_order = [] 
	samples_list = sorted(plot_data[channel].keys()) 
	if reverse: samples_list = samples_list[::-1] 
	for genotype in samples_list: 
		plotting_order.append(genotype) 
		sample_labels.append(genotype.rsplit('_', 1)[-1]) 
		number_reps = len(plot_data[channel][genotype]) 
		for rep_index in range(number_reps): 
			activ_time = plot_data[channel][genotype][rep_index] 
			activ_time = (offset + activ_time) * stack_time / 60.0 
			data_frame['Genotype'].append(genotype) 
			data_frame['ActivTime'].append(activ_time) 
			data_frame['Color'].append(colors[colorIndex]) 
		colorIndex += 1 
	data_frame['Genotype'] = numpy.array(data_frame['Genotype'], dtype = 'str') 
	data_frame['ActivTime'] = numpy.array(data_frame['ActivTime'], dtype = 'float64') 
	data_frame['Color'] = numpy.array(data_frame['Color'], dtype = 'str') 
	data_frame = pd.DataFrame(data_frame) 
	# Generate scatterplot: 
	if horizontal: 
		orientation = "h" 
		rotation = 0 
	else: 
		orientation = "v" 
		rotation = 45 
	swarm = sns.catplot(
					x = 'Genotype',
					y = 'ActivTime',
					hue = 'Color', 
					palette = colors, 
					data = data_frame, 
					orient = orientation,
					order = plotting_order, 
					kind = 'swarm', 
					height = 5.00,
					aspect = 0.575, 
					legend = False 
					)  
	swarm.set_xticklabels( 
					sample_labels, 
					rotation = rotation 
					)  
	plt.subplots_adjust( 
					left = 0.30, 
					bottom = 0.175, 
					right = 0.925, 
					top = 0.900 
					) 
	max_time = max(data_frame['ActivTime']) 
	plt.ylim(0, 1.3 * max_time) 
	p_value = stats.ttest_ind(
					plot_data[channel][samples_list[0]], 
					plot_data[channel][samples_list[1]], 
					equal_var = 'False' 
					) 
	p_value_one_tail = p_value[1] / 2 
	plt.text( 
			0.075, 
			1.15 * max_time, 
			'p = %.4f' % p_value_one_tail 
			) 
	plt.plot(
			[0, 1], 
			[1.075 * max_time, 1.075 * max_time], 
			linewidth = 0.5, 
			color = 'black' 
			) 
	plt.title(title) 
	plt.xlabel('Embryo Genotype') 
	plt.ylabel(y_label) 
	plt.savefig( 
			plot_name, 
			format = 'pdf', 
			dpi = 300 
			) 
	plt.close()
	return 




## DEFINE FUNCTION to generate categorical scatterplot of total transcriptional output per embryo: 

def generate_embryo_total_output_scatterplot(plot_data, channel, plot_name, col, horizontal, reverse, title, y_label):
	# Create formatted data structure: 
	data_frame = {
			'Genotype': [], 
			'TotalOutput': [], 
			'Color': [] 
			} 
	colorIndex = 0 
	colors = col.split(',') 
	colors = colors[ : int(len(colors) / 2)] 
	channels_list = sorted(plot_data.keys()) 
	if reverse: 
		colors = colors[::-1] 
		channels_list = channels_list[::-1] 
	sample_labels = [] 
	plotting_order = [] 
	samples_list = sorted(plot_data[channel].keys()) 
	if reverse: samples_list = samples_list[::-1] 
	for genotype in samples_list: 
		plotting_order.append(genotype) 
		sample_labels.append(genotype.rsplit('_', 1)[-1]) 
		number_reps = len(plot_data[channel][genotype]) 
		for rep_index in range(number_reps): 
			total_output = plot_data[channel][genotype][rep_index] 
			data_frame['Genotype'].append(genotype) 
			data_frame['TotalOutput'].append(total_output) 
			data_frame['Color'].append(colors[colorIndex]) 
		colorIndex += 1 
	data_frame['Genotype'] = numpy.array(data_frame['Genotype'], dtype = 'str') 
	data_frame['TotalOutput'] = numpy.array(data_frame['TotalOutput'], dtype = 'float64') 
	data_frame['Color'] = numpy.array(data_frame['Color'], dtype = 'str') 
	data_frame = pd.DataFrame(data_frame) 
	# Generate scatterplot: 
	if horizontal: 
		orientation = "h" 
		rotation = 0 
	else: 
		orientation = "v" 
		rotation = 45 
	swarm = sns.catplot(
					x = 'Genotype',
					y = 'TotalOutput',
					hue = 'Color', 
					palette = colors, 
					data = data_frame, 
					orient = orientation,
					order = plotting_order, 
					kind = 'swarm', 
					height = 5.00,
					aspect = 0.575, 
					legend = False 
					)  
	swarm.set_xticklabels( 
					sample_labels, 
					rotation = rotation 
					)  
	plt.subplots_adjust( 
					left = 0.30, 
					bottom = 0.175, 
					right = 0.925, 
					top = 0.900 
					) 
	max_output = max(data_frame['TotalOutput']) 
	plt.ylim(0, 1.3 * max_output) 
	p_value = stats.ttest_ind(
					plot_data[channel][samples_list[0]], 
					plot_data[channel][samples_list[1]], 
					equal_var = 'False' 
					) 
	p_value_one_tail = p_value[1] / 2 
	plt.text( 
			0.075, 
			1.15 * max_output, 
			'p = %.4f' % p_value_one_tail 
			) 
	plt.plot(
			[0, 1], 
			[1.075 * max_output, 1.075 * max_output], 
			linewidth = 0.5, 
			color = 'black' 
			) 
	plt.title(title) 
	plt.xlabel('Embryo Genotype') 
	plt.ylabel(y_label) 
	plt.savefig( 
			plot_name, 
			format = 'pdf', 
			dpi = 300 
			) 
	plt.close()
	return 




## DEFINE FUNCTION to compute max active fraction for each embryo: 

def compute_maximum_active_fraction(all_states): 
	max_fraction = {} 
	for channel in all_states: 
		max_fraction[channel] = {} 
		Nembryos, Nnuclei, active_nuclei = active_nuclei_counts(all_states[channel]) 
		for sample_name in all_states[channel]: 
			max_fraction[channel][sample_name] = [] 
			number_reps = len(all_states[channel][sample_name]) 
			for rep_index in range(number_reps): 
				active_fraction = active_nuclei[sample_name][rep_index] / Nnuclei[sample_name][rep_index] 
				maximum = numpy.max(active_fraction) 
				max_fraction[channel][sample_name].append(maximum) 
			max_fraction[channel][sample_name] = numpy.array(max_fraction[channel][sample_name]) 
	return max_fraction 




## DEFINE FUNCTION to generate categorical scatterplot of max active fraction per embryo: 

def generate_maximum_active_fraction_scatterplot(plot_data, plot_name, col, horizontal, reverse, title, y_label):
	# Create formatted data structure: 
	data_frame = {
			'Genotype': [], 
			'maxFraction': [], 
			'Color': [] 
			} 
	colorIndex = 0 
	colors = col.split(',') 
	channels_list = sorted(plot_data.keys()) 
	if reverse: 
		colors = colors[::-1] 
		channels_list = channels_list[::-1] 
	sample_labels = [] 
	plotting_order = [] 
	for channel in channels_list: 
		samples_list = sorted(plot_data[channel].keys()) 
		if reverse: samples_list = samples_list[::-1] 
		for sampleName in samples_list: 
			genotype = channel + ' ' + sampleName 
			plotting_order.append(genotype) 
			sample_labels.append(sampleName.rsplit('_', 1)[-1]) 
			number_reps = len(plot_data[channel][sampleName]) 
			for rep_index in range(number_reps): 
				max_fraction = plot_data[channel][sampleName][rep_index] * 100 
				data_frame['Genotype'].append(genotype) 
				data_frame['maxFraction'].append(max_fraction) 
				data_frame['Color'].append(colors[colorIndex]) 
			colorIndex += 1 
	data_frame['Genotype'] = numpy.array(data_frame['Genotype'], dtype = 'str') 
	data_frame['maxFraction'] = numpy.array(data_frame['maxFraction'], dtype = 'float64') 
	data_frame['Color'] = numpy.array(data_frame['Color'], dtype = 'str') 
	data_frame = pd.DataFrame(data_frame) 
	# Generate scatterplot: 
	if horizontal: 
		orientation = "h" 
		rotation = 0 
	else: 
		orientation = "v" 
		rotation = 45 
	swarm = sns.catplot(
					x = 'Genotype',
					y = 'maxFraction',
					hue = 'Color', 
					palette = colors, 
					data = data_frame, 
					orient = orientation,
					order = plotting_order, 
					kind = 'swarm', 
					height = 4.25,
					aspect = 1.667, 
					legend = False 
					)  
	swarm.set_xticklabels( 
					sample_labels, 
					rotation = rotation 
					)  
	plt.subplots_adjust( 
					left = 0.1, 
					bottom = 0.225, 
					right = 0.975, 
					top = 0.85 
					) 
	plt.title(title) 
	plt.xlabel('Embryo Genotype') 
	plt.ylabel(y_label) 
	plt.savefig( 
			plot_name, 
			format = 'pdf', 
			dpi = 300 
			) 
	plt.close()
	return 




## DEFINE FUNCTION to compute cumulative outputs for individual nuclei: 

def compute_cumulative_outputs(nucleus_profiles): 
	Nnuclei, Ntimes = nucleus_profiles.shape 
	cumul_outputs = numpy.zeros((Nnuclei, Ntimes), dtype = 'float') 
	for nucleus in range(Nnuclei): 
		cumul_outputs[nucleus, 0] = nucleus_profiles[nucleus, 0] 
		for time in range(1, Ntimes): 
			previous_total = cumul_outputs[nucleus, time - 1] 
			current_value = nucleus_profiles[nucleus, time] 
			current_total = previous_total + current_value 
			cumul_outputs[nucleus, time] = current_total 
	return cumul_outputs 




## DEFINE FUNCTION to compute activation times for individual nuclei: 

def compute_single_nucleus_activation_times(all_profiles, critical_output): 
	activation_times = {} 
	for channel in all_profiles.keys(): 
		activation_times[channel] = {} 
		for sample_name in all_profiles[channel].keys(): 
			activation_times[channel][sample_name] = [] 
			number_reps = len(all_profiles[channel][sample_name]) 
			for rep_index in range(number_reps): 
				rep_active_times = [] 
				nucleus_profiles = all_profiles[channel][sample_name][rep_index] 
				cumul_outputs = compute_cumulative_outputs(nucleus_profiles) 
				total_outputs = cumul_outputs[ : , -1] 
				active_nuclei = numpy.where(total_outputs >= critical_output)[0] 
				cumul_outputs = cumul_outputs[active_nuclei, : ] 
				Nnuclei = cumul_outputs.shape[0] 
				for nucleus in range(Nnuclei): 
					nucleus_profile = cumul_outputs[nucleus, :] 
					activ_time = numpy.where(nucleus_profile >= critical_output)[0][0] 
					rep_active_times.append(activ_time) 
				rep_active_times = numpy.array(rep_active_times) 
				activation_times[channel][sample_name].append(rep_active_times) 
	return activation_times 




## DEFINE FUNCTION to compute nucleus activation latencies per embryo: 

def compute_nucleus_activation_latencies(activation_times): 
	activation_latencies = {} 
	for channel in activation_times.keys(): 
		activation_latencies[channel] = {} 
		for sample_name in activation_times[channel].keys(): 
			activation_latencies[channel][sample_name] = [] 
			number_reps = len(activation_times[channel][sample_name]) 
			for rep_index in range(number_reps): 
				active_times = activation_times[channel][sample_name][rep_index] 
				Nnuclei = active_times.shape[0] 
				rep_latencies = numpy.zeros(Nnuclei - 1) 
				sorted_times = numpy.sort(active_times) 
				for nucleus in range(1, Nnuclei): 
					latency = sorted_times[nucleus] - sorted_times[nucleus - 1] 
					rep_latencies[nucleus - 1] = latency 
				activation_latencies[channel][sample_name].append(rep_latencies) 
	return activation_latencies 




## DEFINE FUNCTION to compute nucleus activation latency stats: 

def compute_nucleus_activation_latency_stats(activation_latencies): 
	latency_means = {} 
	latency_medians = {} 
	latency_stdevs = {} 
	for channel in activation_latencies.keys(): 
		latency_means[channel] = {} 
		latency_medians[channel] = {} 
		latency_stdevs[channel] = {}
		for sample_name in activation_latencies[channel].keys(): 
			latency_means[channel][sample_name] = [] 
			latency_medians[channel][sample_name] = [] 
			latency_stdevs[channel][sample_name] = [] 
			number_reps = len(activation_latencies[channel][sample_name]) 
			for rep_index in range(number_reps): 
				rep_latencies = activation_latencies[channel][sample_name][rep_index] 
				rep_mean = numpy.mean(rep_latencies) 
				rep_median = numpy.median(rep_latencies) 
				rep_stdev = numpy.std(rep_latencies) 
				latency_means[channel][sample_name].append(rep_mean) 
				latency_medians[channel][sample_name].append(rep_median) 
				latency_stdevs[channel][sample_name].append(rep_stdev) 
	return latency_means, latency_medians, latency_stdevs 




## DEFINE FUNCTION to compute nucleus activation time statistics: 

def compute_activation_time_stats(activation_times): 
	activation_means = {} 
	activation_stdevs = {} 
	activation_CVs = {} 
	for channel in activation_times.keys(): 
		activation_means[channel] = {} 
		activation_stdevs[channel] = {} 
		activation_CVs[channel] = {} 
		for sample_name in activation_times[channel].keys(): 
			activation_means[channel][sample_name] = [] 
			activation_stdevs[channel][sample_name] = [] 
			activation_CVs[channel][sample_name] = [] 
			number_reps = len(activation_times[channel][sample_name]) 
			for rep_index in range(number_reps): 
				nuc_data = activation_times[channel][sample_name][rep_index] 
				nuc_mean = numpy.mean(nuc_data) 
				nuc_stdev = numpy.std(nuc_data) 
				nuc_CV = nuc_stdev / nuc_mean 
				activation_means[channel][sample_name].append(nuc_mean) 
				activation_stdevs[channel][sample_name].append(nuc_stdev) 
				activation_CVs[channel][sample_name].append(nuc_CV) 
	return activation_means, activation_stdevs 




## DEFINE FUNCTION to generate categorical scatterplot: 

def generate_categorical_scatterplot_OLD(all_data, plotName, col, title, y_label): 
	colors = col.split(',')
	colorIndex = 0 
	# Create formatted data structure: 
	data_frame = { 
			'Embryo': numpy.array([], dtype = 'str'), 
			'Output': numpy.array([], dtype = 'float64'), 
			'Color': numpy.array([], dtype = 'str') 
			} 
	# Go through all samples: 
	sample_list = sorted(all_data.keys()) 
	for sampleName in sample_list: 
		# Go through all replicates: 
		for replicate in range(len(all_data[sampleName])): 
			suffix = '_Rep%s' %(str(replicate + 1).rjust(2, '0')) 
			repName = sampleName + suffix 
			Nnuclei = all_data[sampleName][replicate].shape[0] 
			data_frame['Embryo'] = numpy.hstack((data_frame['Embryo'], numpy.array([repName] * Nnuclei, dtype = 'str'))) 
			data_frame['Output'] = numpy.hstack((data_frame['Output'], all_data[sampleName][replicate])) 
			data_frame['Color'] = numpy.hstack((data_frame['Color'], numpy.array([colors[colorIndex]] * Nnuclei, dtype = 'str'))) 
		colorIndex += 1 
	# Format data frame: 
	data_frame = pd.DataFrame(data_frame) 
	# Generate scatterplot: 
	sns.catplot(
			x = 'Embryo',
			y = 'Output',
			hue = 'Color', 
			palette = colors, 
			data = data_frame, 
			kind = 'swarm', 
			height = 5,
			aspect = 1.4, 
			) 
	plt.title(title) 
	plt.xlabel('Embryo Genotype') 
	plt.ylabel(y_label) 
	plt.savefig(plotName, format = 'pdf', dpi = 300) 
	plt.close()
	return 




##########################################################################################################################################################################
##########################################################################################################################################################################














#### Arguments & Options: 
parser = OptionParser() 
parser.add_option('--image-width', type = 'int', default = 1304) 
parser.add_option('--image-height', type = 'int', default = 864) 
parser.add_option('--fdr', '-f',  type = 'float', default = 0.0000000001) 
parser.add_option('--colors', type = 'str', default = 'magenta,cyan,grey,darkmagenta,darkcyan,black') 
parser.add_option('--ms2-background-start', type = 'int', default = 0)
parser.add_option('--ms2-background-stop', type = 'int', default = None)
parser.add_option('--pp7-background-start', type = 'int', default = 25)
parser.add_option('--pp7-background-stop', type = 'int', default = None)
parser.add_option('--background-normalization', action = 'store_true', default = False) 
parser.add_option('--ms2-new-background', type = 'int', default = None) 
parser.add_option('--pp7-new-background', type = 'int', default = None) 
parser.add_option('--center-of-gravity', type = 'str', default = 'PP7')
parser.add_option('--fix-center-of-gravity', type = 'str', default = None) 
parser.add_option('--center-coordinate', type = 'int', default = None) 
parser.add_option('--embryo-map', type = 'int', default = 80) 
parser.add_option('--domain-width', type = 'int', default = 600) 
parser.add_option('--domain-offset', type = 'int', default = 0) 
parser.add_option('--pixel-size', type = 'float', default = 0.1) 
parser.add_option('--spatial-profiles-start', type = 'int', default = 0) 
parser.add_option('--spatial-profiles-stop', type = 'int', default = None) 
parser.add_option('--normalize-spatial-profiles', action = 'store_true', default = False) 
parser.add_option('--intensity-threshold', type = 'float', default = 0) 
parser.add_option('--normalize-intensities', action = 'store_true', default = False) 
parser.add_option('--stackTime', '-s', type = 'float', default = 21) 
parser.add_option('--nucleus-counts-window', type = 'int', default = 9) 
parser.add_option('--intensities-window', type = 'int', default = 1) 
parser.add_option('--offset', '-o',  type = 'int', default = 25) 
parser.add_option('--spatial-bins', type = 'int', default = 20) 
parser.add_option('--overlapping-bins', action = 'store_true', default = False) 
parser.add_option('--plot-channels-together', action = 'store_true', default = False) 
parser.add_option('--critical-fraction', type = 'float', default = 0.50) 
parser.add_option('--critical-output', type = 'float', default = 500) 
parser.add_option('--reverse-order', action = 'store_true', default = False) 
parser.add_option('--horizontal-swarmplot', action = 'store_true', default = False) 
parser.add_option('--gene-name', '-g', type = 'str', default = 'Gene') 
parser.add_option('--total-output-channel', type = 'str', default = 'MS2') 
(options, args) = parser.parse_args() 
expressionDataDir = args[0] 
controlDataset = args[1] 



#### Load expression datasets into hierarchical data structure: 
allDataSets = load_expression_datasets(expressionDataDir) 
datasetList = allDataSets[0] 
expressionData = allDataSets[1] 
backgroundData = allDataSets[2] 
nucleusCentroids_X = allDataSets[3] 
nucleusCentroids_Y = allDataSets[4] 
embryoOrientations = allDataSets[5] 
del allDataSets 



#### Correct nucleus coordinates for embryo orientation: 
correctOrientations = correct_embryo_orientation( 
										nucleusCentroids_X, 
										nucleusCentroids_Y, 
										embryoOrientations, 
										options.image_width, 
										options.image_height 
										) 
nucleusCentroids_X, nucleusCentroids_Y = correctOrientations 
del correctOrientations 



#### Sample-wise fluorescence normalization to nuclear background (OPTIONAL): 
if options.background_normalization: 
	numberBins = 250
	MS2densityThreshold = 0.01
	PP7densityThreshold = 0.01 
	normalizedData = background_normalization(
										expressionData, 
										backgroundData, 
										options.ms2_background_start, 
										options.ms2_background_stop, 
										options.pp7_background_start, 
										options.pp7_background_stop, 
										numberBins, 
										MS2densityThreshold, 
										PP7densityThreshold, 
										options.ms2_new_background, 
										options.pp7_new_background 
										) 
	expressionData, backgroundData, normalizationFactors = normalizedData 
	del normalizedData 
	##########################################################
	print('\n\nNormalization Factors: \n') 
	for channel in normalizationFactors: 
		print('\n\t', channel) 
		for sampleName in normalizationFactors[channel]: 
			firstRepName = datasetList[sampleName][0] 
			firstRepFactor = normalizationFactors[channel][sampleName][0] 
			print ('\n\t\t%s\t%s\t%.6f' %(sampleName, firstRepName.ljust(40, ' '), firstRepFactor)) 
			numberReps = normalizationFactors[channel][sampleName].shape[0] 
			for k in range(1, numberReps): 
				repName = datasetList[sampleName][k] 
				repFactor = normalizationFactors[channel][sampleName][k] 
				print('\t\t\t\t%s\t%.6f' %(repName.ljust(40, ' '), repFactor)) 
	print('\n') 
	##########################################################



#### Model background distribution & Determine threshold for desired FDR: 
numberBins = 400
densityThreshold = 0.05
exprThresh = {} 
controlData = {} 
for channel in backgroundData: controlData[channel] = numpy.vstack(backgroundData[channel][controlDataset]) 
##########################################################
print('\nControl data: \n') 
for channel in controlData: 
	print('\t', channel, '\t', controlData[channel].shape) 
print('\n') 
##########################################################

MS2backMean, MS2backStd = fit_gaussian( 
								controlData['MS2'], 
								options.ms2_background_start,
								options.ms2_background_stop, 
								numberBins, 
								densityThreshold 
								) 
exprThresh['MS2']= set_signal_threshold( 
								MS2backMean, 
								MS2backStd, 
								options.fdr 
								) 
PP7backMean, PP7backStd = fit_gaussian( 
								controlData['PP7'], 
								options.pp7_background_start,
								options.pp7_background_stop, 
								numberBins, 
								densityThreshold 
								) 
exprThresh['PP7']= set_signal_threshold( 
								PP7backMean, 
								PP7backStd, 
								options.fdr 
								) 
print('\nExpr Thresh: \t', exprThresh, '\n') 



#### Plot background distribution histograms for both channels: 
MS2backHistName = 'MS2_Channel_Background_Histogram.pdf' 
MS2backHistTitle = 'MS2 Channel Background' 
plot_background_histogram( 
					controlData['MS2'], 
					MS2backHistName, 
					MS2backHistTitle, 
					MS2backMean, 
					MS2backStd, 
					exprThresh['MS2'], 
					numberBins, 
					'darkcyan' 
					) 
PP7backHistName = 'PP7_Channel_Background_Histogram.pdf' 
PP7backHistTitle = 'PP7 Channel Background' 
plot_background_histogram( 
					controlData['PP7'], 
					PP7backHistName, 
					PP7backHistTitle, 
					PP7backMean, 
					PP7backStd, 
					exprThresh['PP7'], 
					numberBins, 
					'darkmagenta' 
					) 


#### Subtract background from expression data: 
backgroundCorrectedData = subtract_background(expressionData, exprThresh) 


#### Define region of interest relative to the center of gravity of transcription in one channel: 
domainRefChannel = options.center_of_gravity
centersGravity_X = all_centers_of_gravity( 
							nucleusCentroids_X, 
							options.embryo_map, 
							backgroundCorrectedData, 
							domainRefChannel 
							) 
nucleiInDomains = nuclei_in_domains( 
							nucleusCentroids_X, 
							options.embryo_map, 
							centersGravity_X, 
							options.domain_width, 
							options.domain_offset 
							) 
selectedNuclei = filter_nuclei( 
							backgroundCorrectedData, 
							nucleiInDomains 
							) 



#### Compute nucleus activity states for all datasets (Instantaneous & Cumulative): 
nucleusActivityProfiles = nucleus_activity_states_alldata(selectedNuclei, options.nucleus_counts_window) 
nucleusCumulativeActivityProfiles = nucleus_activity_states_cumulative_alldata(selectedNuclei) 


#### Plot active nucleus counts over time for each individual replicate: 
# Go through both channels: 
for channel in ['MS2', 'PP7']: 
	# Go through all genotypes / experimental conditions, and plot separately: 
	for genotypeName in sorted(nucleusActivityProfiles[channel].keys()): 
		# Generate plot with nucleus activity profile for each replicate: 
		plotFileName = '%s_%s_ActiveNucleusCounts.pdf' %(genotypeName, channel) 
		plotTitle = '%s Replicates – %s Channel' %(genotypeName, channel) 
		plot_nucleus_activity_replicates_per_sample(
								datasetList[genotypeName], 
								nucleusActivityProfiles[channel][genotypeName], 
								options.stackTime,
								options.nucleus_counts_window,
								options.offset,
								options.fdr, 
								plotFileName, 
								plotTitle
								) 
	# Then also plot all replicates for all samples together: 
	plotFileName = expressionDataDir.rstrip('/') + '_%s_NucleusCounts_Replicates.pdf' % channel 
	plotTitle = 'All Replicates – %s Channel' % channel 
	plot_nucleus_activity_all_replicates( 
							nucleusActivityProfiles[channel], 
							options.stackTime, 
							options.nucleus_counts_window, 
							options.offset, 
							options.fdr, 
							plotFileName, 
							options.colors, 
							plotTitle 
							) 


#### Plot INSTANTANEOUS active nucleus counts over time, for each genotype: 
for channel in ['MS2', 'PP7']: 
	plotFileName = expressionDataDir.rstrip('/') + '_%s_NucleusCounts.pdf' % channel 
	plotTitle = 'Active Nucleus Counts (%s)' % channel 
	plot_nucleus_activity_profiles( 
							nucleusActivityProfiles[channel], 
							options.stackTime, 
							options.nucleus_counts_window, 
							options.offset, 
							options.fdr, 
							None, 
							plotFileName, 
							options.colors, 
							plotTitle 
							) 


#### Plot CUMULATIVE active nucleus counts over time, for each genotype: 
for channel in ['MS2', 'PP7']: 
	plotFileName = expressionDataDir.rstrip('/') + '_%s_NucleusCountsCumulative.pdf' % channel 
	plotTitle = 'Cumulative Nucleus Counts (%s)' % channel 
	plot_nucleus_activity_profiles( 
							nucleusCumulativeActivityProfiles[channel], 
							options.stackTime, 
							1, 
							options.offset, 
							options.fdr, 
							options.critical_fraction, 
							plotFileName, 
							options.colors, 
							plotTitle 
							) 


#### Plot intensities over time for active nuclei, for each genotype:
for channel in ['MS2', 'PP7']: 
	plotFileName = expressionDataDir.rstrip('/') + '_%s_Intensities.pdf' % channel 
	plotTitle = 'Transcriptional Activity in Active Nuclei (%s)' % channel 
	plot_transcription_intensities_profiles( 
							selectedNuclei[channel], 
							options.intensity_threshold, 
							options.intensities_window, 
							options.normalize_intensities, 
							options.stackTime, 
							options.offset, 
							plotFileName, 
							options.colors, 
							plotTitle 
							) 


#### Compute total output per nucleus: 
nucleusTotalOutput = total_output_per_nucleus( 
							selectedNuclei, 
							options.spatial_profiles_start, 
							options.spatial_profiles_stop 
							) 


#### Compute & plot spatial transcriptional activity profiles along A-P axis, for each genotype: 
spatialProfileData = compute_spatial_activity_profiles( 
							nucleusTotalOutput,
							nucleusCentroids_X,
                            options.embryo_map, 
							nucleiInDomains, 
                            centersGravity_X,
                            options.domain_width, 
							options.domain_offset, 
							options.spatial_bins, 
							options.overlapping_bins 
							) 
spatialActivityProfiles, spatialBinMids = spatialProfileData 
del spatialProfileData 

for channel in ['MS2', 'PP7']: 
	if options.spatial_profiles_start == 0: startName = 'First' 
	else: startName = str(options.spatial_profiles_start).rjust(3, '0') 
	if options.spatial_profiles_stop: stopName = str(options.spatial_profiles_stop).rjust(3, '0') 
	else: stopName = 'Last' 
	plotFileName = expressionDataDir.rstrip('/') + '_%s_SpatialPatterns_%s_to_%s.pdf' %(channel, startName, stopName)  
	plotTitle = 'Spatial Transcription Activity Profiles (%s)' % channel 
	plot_spatial_activity_profiles( 
							spatialActivityProfiles[channel], 
							spatialBinMids, 
							options.pixel_size, 
							options.normalize_spatial_profiles, 
							plotFileName, 
							options.colors, 
							plotTitle
							) 


#### Plot both channels together in a single graph (OPTIONAL): 
if options.plot_channels_together: 
	for dataSetName in datasetList: 
		## Plot instantaneous active nucleus counts: 
		plotFileName = expressionDataDir.rstrip('/') + '_%s_NucleusCounts.pdf' % dataSetName 
		plotTitle = 'Active Nucleus Counts (%s)' % dataSetName 
		dataSetActivityProfiles = {'MS2': {}, 'PP7': {}} 
		dataSetActivityProfiles['MS2'] = nucleusActivityProfiles['MS2'][dataSetName] 
		dataSetActivityProfiles['PP7'] = nucleusActivityProfiles['PP7'][dataSetName] 
		plot_nucleus_activity_profiles( 
							dataSetActivityProfiles, 
                            options.stackTime,
                            options.nucleus_counts_window,
                            options.offset,
                            options.fdr,
                            plotFileName,
                            options.colors,
                            plotTitle
                            ) 
		## Plot spatial activity profiles: 
		plotFileName = expressionDataDir.rstrip('/') + '_%s_SpatialPatterns_%s_to_%s.pdf' %(dataSetName, startName, stopName) 
		plotTitle = 'Spatial Transcription Activity Profiles (%s)' % dataSetName 
		dataSetSpatialProfiles = {'MS2': {}, 'PP7': {}} 
		dataSetSpatialProfiles['MS2'] = spatialActivityProfiles['MS2'][dataSetName] 
		dataSetSpatialProfiles['PP7'] = spatialActivityProfiles['PP7'][dataSetName] 
		plot_spatial_activity_profiles( 
							dataSetSpatialProfiles, 
							spatialBinMids, 
							options.pixel_size, 
							True, 
							plotFileName, 
							options.colors, 
							plotTitle
							) 
		## Plot scatterplot of total output (MS2 vs. PP7): 
		plotFileName = expressionDataDir.rstrip('/') + '_%s_TotalOutput_Scatterplot_%s_to_%s.pdf' %(dataSetName, startName, stopName) 
		plotTitle = 'Total transcriptional output (%s)' % dataSetName 
		dataSetTotalOutput = {'MS2': {}, 'PP7': {}}
		dataSetTotalOutput['MS2'] = nucleusTotalOutput['MS2'][dataSetName] 
		dataSetTotalOutput['PP7'] = nucleusTotalOutput['PP7'][dataSetName] 
		plot_total_output_scatterplot( 
							dataSetTotalOutput, 
							plotFileName, 
							plotTitle 
							) 
		


#### Compute total transcriptional output per embryo: 
embryoTotalOutput = total_output_per_embryo(nucleusTotalOutput) 
write_total_output_per_embryo(embryoTotalOutput) 



#### Compute critical activation time per embryo: 
embryoActivationTimes = compute_critical_activation_time( 
							nucleusCumulativeActivityProfiles, 
							controlDataset, 
							options.critical_fraction 
							) 


#### Plot categorical scatterplot of total transcriptional output per embryo: 
plotFileName = expressionDataDir.rstrip('/') + '_TotalOutput.pdf' 
plotTitle = 'Total output' 
generate_embryo_total_output_scatterplot( 
							embryoTotalOutput, 
							options.total_output_channel, 
							plotFileName, 
							options.colors, 
							options.horizontal_swarmplot, 
							options.reverse_order,
							plotTitle, 
							'Total output (a.u.)' 
							) 


#### Plot categorical scatterplot of gene activation times per embryo: 
plotFileName = expressionDataDir.rstrip('/') + '_ActivationTimes.pdf' 
plotTitle = 'Time to Gene Activation' 
generate_activation_time_scatterplot( 
							embryoActivationTimes, 
							options.total_output_channel,
							plotFileName, 
							options.colors,
							options.horizontal_swarmplot,
							options.reverse_order, 
							options.stackTime, 
							options.offset,
							plotTitle, 
							'Activation Time (min)' 
							) 


#### Compute maximum active fraction of nuclei per embryo: 
maxActiveFraction = compute_maximum_active_fraction(nucleusActivityProfiles) 


#### Plot categorical scatterplot of maximum active fraction per embryo: 
plotFileName = expressionDataDir.rstrip('/') + '_MaxActiveFraction.pdf' 
plotTitle = 'Maximum Fraction of Active Nuclei' 
generate_maximum_active_fraction_scatterplot( 
							maxActiveFraction, 
							plotFileName, 
							options.colors,
							options.horizontal_swarmplot, 
							options.reverse_order, 
							plotTitle,
							'Maximum Active Fraction (%)' 
							) 

print('\n') 


















