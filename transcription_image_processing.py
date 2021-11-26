#######################################
## transcription_image_processing.py ## 
#######################################
##
##
##
##
##
## Philippe J. BATUT 
## Levine lab
## Lewis-Sigler Institute
## Princeton University 
## pbatut@princeton.edu 
## 
## April 2018 
## 
##
##
##
##
##
## DEPENDENCIES: 
##
##		Python 3.6.5 			https://www.python.org
##		Scipy 1.0.1 			https://www.scipy.org
##		Numpy 1.14.2 			http://www.numpy.org
## 		czifile					https://github.com/AllenCellModeling/czifile
##		tifffile				https://github.com/blink1073/tifffile 
##		scikit-image 0.13.1 	http://scikit-image.org 
## 		OpenCV 3.4.1			https://opencv.org 
##		StatsModels				http://www.statsmodels.org 
##		
##
##
##
## USAGE: 
##
##		$ python transcription_image_processing.py [OPTIONS] imageFile.czi 
##
##		
##
##
## OPTIONS: 
## 
##		--first-frame (-F):			Specify the first frame to load from the image file (0-based inclusive; Default: First) 
##		--last-frame (-L):			Specify the last frame to load from the image file (0-based exclusive; Default: Last) 
##		--channels (-C):			Order of the channels in the image file, in RGB order (Default: 1,0,2) 
##		--remove-edges (-E):		Filter out embryo edges from imdividual images prior to segmentation (Default: False) 
##		--min-edge-area (-A):		Minimum surface area, in pixels, for putative embryo edges to be filtered out (Default: 3,000) 
##		--segPlanes (-P):			Number of Z-planes (above and below median plane) to use for segmentation (Default: 12) 
##		--gaussian-kernel (-G):		Width of the Gaussian kernel used for smoothing prior to segmentation (Default: 15) 
##		--morphology-kernel (-M): 	Width of the morphological kernel used for feature opening/closing during segmentation (Default: 5) 
##		--distance-threshold (-D): 	Threshold applied to the distance transform of the image during watershed (Default: 6) 
##		--size-threshold (-S): 		For filtering of excessively small/large objects after segmentation. Fold-change relative to  median size (Default: 3.0) 
##		--border-fraction (-B): 	Width of image border to ignore for analysis, as percentage of image size (Default: 0.01) 
##		--time-kernel (-T):			Width of time kernel for temporal smoothing of segmentation masks (Default: 7 time points) 
##		--spot-radius (-R):			Radius of the volumetric kernel used to scan nuclear volumes and compute signal intensities (Default: 3 voxels) 
##		--spot-depth (-d):			Depth (in Z) of the volumetric kernel (Default: 4 voxels) 
##		--green-background-start:	First time point to consider for computation of background distribution for GFP channel (Default: First) 
##		--green-background-stop:	Last time point to consider for computation of background distribution for GFP channel (Default: Last) 
##		--red-background-start:		First time point to consider for computation of background distribution for RFP channel (Default: 25) 
##		--red-background-stop:		Last time point to consider for computation of background distribution for RFP channel (Default: Last) 
##		--fdr: 						Experimentally defined False Discovery Rate for background correction 
##									(Only used for the generation of a few diagnostic plots. Does NOT affect output data files) 
##		--embryo-map:				Time point for nuclei segmentation masks used to generate embryo heatmaps (Does NOT affect data files) 
## 		--segmentation-movies:		Output MP4 movies for visualization of nuclei segmentation over the time series (Default: False) 
##		--spot-detection-movies:	Output MP4 movies for visualization of MS2/PP7 spots over the time series (Default: False) 
##		--cores:					Number of processors to use for transcription signal extraction step (Default: 8) 
##
##
##
##
## ARGUMENTS: 
##
## 		Input CZI 4D image file (3D image stack, time series, 3 channels expected including nuclear labeling for segmentation of individual nuclei) 
##
##
##
##
## OUTPUT FILES: 
##
##		Nuclei_Segmentation			Directory containing tracked nuclei segmentation masks for all time points
##		*_Background.txt			Background estimates for all nuclei at all time points (2 files: 1 per transcription measurement channel)
##		*_Signal.txt				Transcription spot intensities for all nuclei at all time points (2 files: 1 per transcription measurement channel)
##		*_Nucleus_X.txt/*_Y.txt		X-Y coordinates of tracked segmentation mask centroids throughout the time series
##		*.png / *.mp4				Various graphical outputs (plots and movies) for data visualization and quality control. 
##
##
##
##









import os 
import cv2 
import sys 
import time 
import scipy 
import numpy 
import random 
import subprocess 
import matplotlib 
matplotlib.use('Agg') 
import czifile as czi 
import tifffile as tiff 
import multiprocessing
import scipy.stats as stats 
from skimage import filters 
import scipy.optimize as optim 
import matplotlib.pyplot as plt 
from optparse import OptionParser 
import matplotlib.animation as anim 
import scipy.spatial.distance as dist 
from matplotlib.animation import FFMpegWriter
import statsmodels.stats.multitest as multitest 
startTime = time.time() 








##########################################################################################################################################################################
##########################################################################################################################################################################




## DEFINE FUNCTION to convert 16-bit image to 8-bit: 

def convert_to_8bit(image): 
	scaleFactor = 65535 / 255 
	convertedImage = numpy.uint8((numpy.int16(image) / scaleFactor)) 
	return convertedImage 




## DEFINE FUNCTION to remove artifacts (embryo edges (optional), lipid blobs): 

def remove_artifacts(dataset, histChannel, gaussKernel, morphoKernel, removeEdges, minArea): 
	# Load data & create segmentation mask: 
	Ntime, Nchannels, Nz, Ny, Nx = dataset.shape 
	artifactsSeg = numpy.zeros((Ntime, Nz, Ny, Nx), dtype = 'uint8') 
	# Define kernels for Gaussian blurring: 
	edgesBlurKernel = 2 * gaussKernel - 1 
	blobsBlurKernel = int((gaussKernel - 1) / 2) 
	if blobsBlurKernel % 2 == 0: blobsBlurKernel += 1 
	# Define morphological kernel: 
	mKernel = numpy.ones((morphoKernel, morphoKernel), dtype = numpy.int8)
	mKernel = numpy.uint8(mKernel * cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphoKernel, morphoKernel)))
	# Go through all time points: 
	for time in range(Ntime): 
		# Go through all Z-planes: 
		for Z in range(Nz): 
			# Extract histone channel & blur: 
			histoneImage = dataset[time, histChannel, Z, :, :] 
			blurImage = cv2.GaussianBlur(histoneImage, (edgesBlurKernel, edgesBlurKernel), -1) 
			totalSignal = numpy.sum(blurImage) 
			# Create segmentation masks for artifacts: 
			blobsMask = numpy.zeros(histoneImage.shape, dtype = 'uint8') 
			if removeEdges: edgesMask = blobsMask.copy() 
			# Define segmentation thresholds: 
			if totalSignal == 0: 
				print('\tERROR: No Signal (T=%s, Z=%s)' %(time, Z)) 
				continue 
			segThresh = filters.threshold_otsu(blurImage) 
			edgesSegThresh = 1.25 * segThresh 
			blobsSegThresh = 3 * segThresh 
			# Remove embryo edges if desired: 
			if removeEdges: 
				segImage = numpy.uint8(cv2.threshold(blurImage, edgesSegThresh, 255, cv2.THRESH_BINARY)[1]) 
				segContours = cv2.findContours(segImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1] 
				nbObjects = len(segContours) 
				for k in range(nbObjects):  
					if cv2.contourArea(segContours[k]) > minArea: cv2.drawContours(edgesMask, segContours, k, 255, -1) 
				edgesMask = cv2.morphologyEx(edgesMask, cv2.MORPH_OPEN, mKernel) 
				edgesMask = cv2.morphologyEx(edgesMask, cv2.MORPH_CLOSE, mKernel) 
				edgesMask = cv2.dilate(edgesMask, mKernel, iterations = 2) 
			# Remove lipid blobs: 
			blurImage = cv2.GaussianBlur(histoneImage, (blobsBlurKernel, blobsBlurKernel), -1) 
			blobsMask[blurImage > blobsSegThresh] = 255 
			blobsMask = cv2.morphologyEx(blobsMask, cv2.MORPH_OPEN, mKernel) 
			blobsMask = cv2.morphologyEx(blobsMask, cv2.MORPH_CLOSE, mKernel) 
			blobsMask= cv2.dilate(blobsMask, mKernel, iterations = 1) 
			# Apply masks to the relevant channels: 
			if removeEdges: 
				edgesMaskInv = numpy.where(edgesMask > 0, 0, 1) 
				for channel in range(Nchannels): dataset[time, channel, Z, :, :] = dataset[time, channel, Z, :, :] * edgesMaskInv 
			blobsMaskInv = numpy.where(blobsMask > 0, 0, 1) 
			dataset[time, histChannel, Z, :, :] = dataset[time, histChannel, Z, :, :] * blobsMaskInv 
			# Save combined segmentation mask: 
			artifactsMask = blobsMask.copy() 
			if removeEdges: 
				artifactsMask += edgesMask 
				artifactsMask[artifactsMask > 255] = 255 
			artifactsMask = numpy.uint8(artifactsMask) 
			artifactsSeg[time, Z, :, :] = artifactsMask 
	# Compile Z-projected segmentation mask: 
	artifactsSeg = numpy.sum(artifactsSeg, 1) 
	# Return segmentation mask & filtered data: 
	return artifactsSeg, dataset 




## DEFINE FUNCTION to resolve fusions of neighboring elements after dilatation: 

def resolve_fusions(initialObjectMarkers, objectMarkers, nucleiGauss, morphoKernel): 
	# Go through all final objects: 
	numberFinalObjects = numpy.amax(objectMarkers) 
	segmentationMask = numpy.uint8(numpy.where(objectMarkers > 0, 255, 0)) 
	mKernel = numpy.ones((morphoKernel, morphoKernel), dtype = numpy.int8) 
	mKernel = numpy.uint8(mKernel * cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphoKernel, morphoKernel))) 
	for segObjectIndex in range(1, numberFinalObjects + 1): 
		# Identify object markers for initial round of watershed: 
		initialMarkers = numpy.where(objectMarkers == segObjectIndex, initialObjectMarkers, 0) 
		initialMarkerIndexes = numpy.unique(initialMarkers) 
		initialMarkerIndexes = initialMarkerIndexes[initialMarkerIndexes > 0] 
		numberInitialMarkers = initialMarkerIndexes.shape[0]  
		# Identify initial objects merged during dilatation : 
		if numberInitialMarkers <= 1: continue 
		# Apply watershed algorithm locally to find boundaries: 
		backgroundArea = numpy.where(objectMarkers == segObjectIndex, 255, 0) 
		foregroundArea = numpy.where(initialMarkers > 0, 255, 0) 
		ambiguousArea = cv2.subtract(backgroundArea, foregroundArea) 
		markersForWatershed = 1 + initialMarkers 
		markersForWatershed[ambiguousArea == 255] = 0 
		nuclearSignal = numpy.where(objectMarkers == segObjectIndex, nucleiGauss, 0) 
		nuclearSignalRGB = cv2.cvtColor(convert_to_8bit(nuclearSignal), cv2.COLOR_GRAY2BGR) 
		correctedObjectMarkers = cv2.watershed(nuclearSignalRGB, markersForWatershed) 
		# Apply opening to re-segmented objects: 
		correctedObjectMarkers = numpy.float64(numpy.where(correctedObjectMarkers > 1, 255, 0)) 
		correctedObjectMarkers = numpy.uint8(cv2.morphologyEx(correctedObjectMarkers, cv2.MORPH_OPEN, mKernel)) 
		# Correct object boundaries in segmentation mask: 
		segmentationMask[objectMarkers == segObjectIndex] = correctedObjectMarkers[objectMarkers == segObjectIndex] 
	# Label corrected segmentation mask: 
	segmentationMask = cv2.connectedComponents(segmentationMask, connectivity = 4, ltype = cv2.CV_32S)[1] 
	# Return corrected segmentation mask: 
	return segmentationMask 




## DEFINE FUNCTION to segment nuclei: 

def segment_nuclei(nucleiSignal, includePlanes, gaussKernel, morphoKernel, distTransformThresh): 
	# Z-stack range to segment from: 
	nbPlanes = nucleiSignal.shape[0] 
	midPlane = int(nbPlanes / 2) 
	firstPlane = midPlane - includePlanes 
	if nbPlanes % 2 == 0: lastPlane = midPlane + includePlanes 
	else: lastPlane = midPlane + includePlanes + 1 
	nucleiSignal = nucleiSignal[firstPlane : lastPlane, :, :] 
	# Average intensity projection & Gaussian smoothing: 
	nucleiAvg = numpy.mean(nucleiSignal, 0) 
	nucleiGauss = cv2.GaussianBlur(nucleiAvg, (gaussKernel, gaussKernel), -1) 
	# Apply initial thresholding: 
	otsuThresh = filters.threshold_otsu(nucleiGauss) 
	otsuThresh = 0.95 * otsuThresh 
	otsuThresh, nucleiOtsu = cv2.threshold(nucleiGauss, otsuThresh, 255, cv2.THRESH_BINARY) 
	mKernel = numpy.ones((morphoKernel, morphoKernel), dtype = numpy.int8) 
	mKernel = numpy.uint8(mKernel * cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphoKernel, morphoKernel))) 
	nucleiOpen = cv2.morphologyEx(nucleiOtsu, cv2.MORPH_OPEN, mKernel) 
	nucleiClose = cv2.morphologyEx(nucleiOpen, cv2.MORPH_CLOSE, mKernel).astype('uint8') 
	# Apply watershed algorithm: 
	backgroundArea = cv2.dilate(nucleiClose, mKernel, iterations = 2) 
	distTransform = cv2.distanceTransform(nucleiClose, cv2.DIST_L2, 5) 
	foregroundArea = cv2.threshold(distTransform, distTransformThresh, 255, cv2.THRESH_BINARY)[1] 
	foregroundAreaOpen = cv2.morphologyEx(foregroundArea, cv2.MORPH_OPEN, mKernel).astype('uint8') 
	ambiguousArea = cv2.subtract(backgroundArea, foregroundAreaOpen) 
	initialObjectMarkers = 1 + cv2.connectedComponents(foregroundAreaOpen, connectivity = 4, ltype = cv2.CV_32S)[1]  
	initialObjectMarkers[ambiguousArea == 255] = 0 
	nucleiGaussRGB = cv2.cvtColor(convert_to_8bit(nucleiGauss), cv2.COLOR_GRAY2BGR) 
	initialObjectMarkers = cv2.watershed(nucleiGaussRGB, initialObjectMarkers) 
	# Apply opening to segmented objects: 
	initialObjectMarkers = numpy.float64(numpy.where(initialObjectMarkers > 1, 255, 0)) 
	initialObjectMarkers = numpy.uint8(cv2.morphologyEx(initialObjectMarkers, cv2.MORPH_OPEN, mKernel)) 
	objectSeeds = cv2.connectedComponents(initialObjectMarkers, connectivity = 4, ltype = cv2.CV_32S)[1] 
	# Dilate resulting objects, and resolve potential fusions: 
	miniKernel = numpy.ones((3, 3), dtype = numpy.int8) 
	miniKernel = numpy.uint8(miniKernel * cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) 
	for n in range(1): 
		objectMarkers = cv2.dilate(initialObjectMarkers, miniKernel, iterations = 4) 
		objectMarkers = cv2.connectedComponents(objectMarkers, connectivity = 4, ltype = cv2.CV_32S)[1] 
		initialObjectMarkers = cv2.connectedComponents(initialObjectMarkers, connectivity = 4, ltype = cv2.CV_32S)[1] 
		resolvedObjectMarkers = resolve_fusions(initialObjectMarkers, objectMarkers, nucleiGauss, morphoKernel) 
		initialObjectMarkers = numpy.uint8(numpy.where(resolvedObjectMarkers > 0, 255, 0)) 
	# Return initial seed regions & final segmented objects: 
	return nucleiGauss, objectSeeds, resolvedObjectMarkers 




## DEFINE FUNCTION to rescue dropout nuclei: 

def recover_dropout_nuclei(segMask, referenceMask): 
	numberReferenceObjects = numpy.amax(referenceMask) 
	rescueObjectCounter = numpy.amax(segMask) + 1 
	correctedMask = segMask.copy() 
	# Go through all objects in reference segmentation mask: 
	for segObjectIndex in range(1, numberReferenceObjects + 1): 
		# Identify overlapping objects in current segmentation mask: 
		currentObjects = numpy.where(referenceMask == segObjectIndex, segMask, 0) 
		referenceObjectIndexes = numpy.unique(currentObjects) 
		referenceObjectIndexes = referenceObjectIndexes[referenceObjectIndexes > 0] 
		# If none, add in object from reference mask: 
		numberOverlaps = referenceObjectIndexes.shape[0] 
		if numberOverlaps == 0: 
			correctedMask[referenceMask == segObjectIndex] = rescueObjectCounter 
			rescueObjectCounter += 1 
	# Return corrected segmentation mask: 
	return correctedMask 




## DEFINE FUNCTION to compute object surface areas: 

def object_areas(segObjects): 
	Nobjects = numpy.amax(segObjects) 
	objectAreas = numpy.zeros((Nobjects + 1), dtype = 'int') 
	for k in range(Nobjects + 1): objectAreas[k] = segObjects[segObjects == k].shape[0] 
	return objectAreas 




## DEFINE FUNCTION to exclude large and small objects: 

def remove_size_outliers(segObjects, sizeFactor): 
	objectAreas = object_areas(segObjects) 
	medianArea = numpy.median(objectAreas[1:]) 
	lowerThresh = int(medianArea / sizeFactor) 
	upperThresh = int(medianArea * sizeFactor) 
	for k in range(1, numpy.amax(segObjects) + 1): 
		if objectAreas[k] < lowerThresh: segObjects[segObjects == k] = 0 
		if objectAreas[k] > upperThresh: segObjects[segObjects == k] = 0 
	segObjects = numpy.uint8(numpy.where(segObjects > 0, 255, 0)) 
	segObjects = cv2.connectedComponents(segObjects, connectivity = 4, ltype = cv2.CV_32S)[1]
	return segObjects 




## DEFINE FUNCTION to exclude objects at the borders of the frame: 

def exclude_borders(segObj, borderFraction): 
	# Compute border width: 
	Ntimes, sizeY, sizeX = segObj.shape 
	borderWidthX = int(sizeX * borderFraction) 
	borderWidthY = int(sizeY * borderFraction) 
	bottomLimitY = borderWidthY 
	topLimitY = sizeY - borderWidthY 
	bottomLimitX = borderWidthX 
	topLimitX = sizeX - borderWidthX 
	# Exclude objects at borders: 
	noBordersObj = numpy.zeros(segObj.shape, dtype = 'int32') 
	for t in range(Ntimes): 
		objCounter = 1 
		Nobjects = numpy.amax(segObj[t]) 
		for k in range(1, Nobjects + 1): 
			if k not in segObj[t]: continue 
			objCoord = numpy.where(segObj[t] == k) 
			minY = numpy.amin(objCoord[0]) 
			maxY = numpy.amax(objCoord[0]) 
			minX = numpy.amin(objCoord[1]) 
			maxX = numpy.amax(objCoord[1]) 
			if (minY <= bottomLimitY or minX <= bottomLimitX or maxY >= topLimitY or maxX >= topLimitX): continue 
			noBordersObj[t][objCoord] = objCounter 
			objCounter += 1 
	# Return filtered segmentation masks: 
	return noBordersObj 




## DEFINE FUNCTION to compute the centroid coordinates for a set of objects: 

def define_centroids(segMask): 
	# Define parameters & data structure: 
	Ny, Nx = segMask.shape 
	Nobjects = numpy.amax(segMask) 
	centroids = numpy.zeros((Nobjects, 2), dtype = 'int32') 
	# Define centroid coordinates: 
	for objIndex in range(1, Nobjects + 1): 
		currentObject = numpy.uint8(numpy.where(segMask == objIndex, 255, 0)) 
		objContours = cv2.findContours(currentObject, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1] 
		objMoments = cv2.moments(objContours[0]) 
		objCentroidX = int(objMoments['m10'] / objMoments['m00']) 
		objCentroidY = int(objMoments['m01'] / objMoments['m00']) 
		centroids[objIndex - 1] = objCentroidX, objCentroidY 
	# Return centroid coordinates: 
	return centroids 




## DEFINE FUNCTION to assign tracking ID in segmentation mask: 

def assign_tracking_IDs(segObjects, trackingIDs): 
	# Create data structure for tracked segmentation mask: 
	trackedObjects = numpy.zeros(segObjects.shape, dtype = 'int32') 
	Ntimes, Nobjects = trackingIDs.shape 
	# Go through all time points: 
	for t in range(Ntimes): 
		# Annotate each object with its tracking ID: 
		for k in range(Nobjects): 
			segID = trackingIDs[t, k]
			objCoord = numpy.where(segObjects[t] == segID) 
			trackedObjects[t][objCoord] = k + 1 
	# Return segmentation mask with tracking IDs: 
	return trackedObjects 




## DEFINE FUNCTION to track nuclei using Hungarian (Munkres) algorithm: 

def track_nuclei_munkres(segObjects, borderFraction): 
	# Exclude objects at borders of frame: 
	segObjects = exclude_borders(segObjects, borderFraction) 
	# Define parameters & data structures: 
	Ntimes, Ny, Nx = segObjects.shape 
	numberInitialObjects = numpy.amax(segObjects[0]) 
	trackingIDs = numpy.zeros((Ntimes, numberInitialObjects), dtype = 'int32') 
	for k in range(numberInitialObjects): trackingIDs[0, k] = k + 1 
	# Compute centroid coordinates for all time points: 
	centroidCoord = [] 
	for t in range(Ntimes): 
		coordinates = define_centroids(segObjects[t]) 
		centroidCoord.append(coordinates) 
	# Track nuclei across time points: 
	for t in range(1, Ntimes): 
		# Consider consecutive time points: 
		refCoord = centroidCoord[t - 1] 
		newCoord = centroidCoord[t] 
		# Compute all pairwise Euclidean distances: 
		distMatrix = dist.cdist(
							refCoord, 
							newCoord, 
							metric = 'euclidean' 
							) 
		# Compute solution of assignment problem (Munkres algorithm): 
		shiftMaxRange = 5 
		refIndexes, newIndexes = optim.linear_sum_assignment(distMatrix) 
		medianShift = numpy.median(distMatrix[refIndexes, newIndexes]) 
		maxShift = shiftMaxRange * medianShift 
		# Increase cost of excessive distances to prevent spurious assignments: 
		extraCost = 100 * medianShift 
		distMatrix = numpy.where(distMatrix > maxShift, distMatrix + extraCost, distMatrix) 
		# Re-compute solution of assignment problem (Munkres algorithm): 
		refIndexes, newIndexes = optim.linear_sum_assignment(distMatrix) 
		numberAssignments = refIndexes.shape[0] 
		for n in range(numberAssignments): 
			nucleusShift = distMatrix[refIndexes[n], newIndexes[n]] 
			if nucleusShift > maxShift: continue 
			refObj = refIndexes[n] + 1 
			newObj = newIndexes[n] + 1 
			rowIndex = numpy.where(trackingIDs[t - 1] == refObj)[0]
			if rowIndex.shape[0] != 0: trackingIDs[t, rowIndex[0]] = newObj 
	# Filter out objects for which tracking failed: 
	trackingSuccess = numpy.ones(trackingIDs.shape[1], dtype = 'bool') 
	trackingSuccess[numpy.unique(numpy.where(trackingIDs == 0)[1])] = False 
	trackingIDs = trackingIDs[:, trackingSuccess == True] 
	# Assign tracking IDs in segmentation mask: 
	trackedObjects = assign_tracking_IDs(segObjects, trackingIDs) 
	# Return segmentation mask with tracked objects: 
	return trackedObjects 




## DEFINE FUNCTION to generate temporal kernel for shape filtering: 

def generate_temporal_kernel(kernelLength): 
	tempoKernel = numpy.zeros(kernelLength, dtype=int) 
	halfKernel = int(kernelLength / 2) 
	weight = 1 
	for i in range(halfKernel): 
		tempoKernel[i] = weight 
		tempoKernel[-(i+1)] = weight 
		weight += 1 
	tempoKernel[halfKernel] = weight 
	tempoKernel = numpy.reshape(tempoKernel, (kernelLength, 1, 1)) 
	return tempoKernel 




## DEFINE FUNCTION to apply temporal filtering to tracked object shapes: 

def features_temporal_filter(trackedObjects, kernelLength, morphoKernel): 
	# Define temporal kernel for filtering: 
	tempoKernel = generate_temporal_kernel(kernelLength) 
	kernelLength = tempoKernel.shape[0] 
	halfKernel = int(kernelLength / 2) 
	# Define threshold for filtering: 
	filterThresh = tempoKernel[halfKernel, 0, 0] + tempoKernel[halfKernel - 1, 0, 0] + 1 
	# Define morphological kernels: 
	mKernel = numpy.ones((morphoKernel, morphoKernel), dtype = numpy.int8)
	mKernel = numpy.uint8(mKernel * cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphoKernel, morphoKernel))) 
	miniKernel = numpy.ones((3, 3), dtype = numpy.uint8)
	# Create data structure to store filtered shapes: 
	Ntimes, Ny, Nx = trackedObjects.shape 
	Nobjects = numpy.amax(trackedObjects) 
	filteredShapes = numpy.zeros(trackedObjects.shape, dtype = 'int32') 
	for t in range(halfKernel): 
		filteredShapes[t] = trackedObjects[t] 
		filteredShapes[-(t+1)] = trackedObjects[-(t+1)] 
	# Apply temporal filtering: 
	for k in range(1, Nobjects + 1): 
		for timepoint in range(halfKernel, Ntimes - halfKernel): 
			timeProjection = numpy.zeros((kernelLength, Ny, Nx), dtype = 'uint8') 
			timeSlice = 0 
			for t in range(timepoint - halfKernel, timepoint + halfKernel + 1): 
				objectMask = numpy.where(trackedObjects[t] == k, 1, 0) 
				timeProjection[timeSlice] = objectMask 
				timeSlice += 1 
			timeProjection = timeProjection * tempoKernel 
			timeProjection = numpy.sum(timeProjection, 0) 
			filteredObject = numpy.uint8(numpy.where(timeProjection >= filterThresh, 1, 0)) 
			filteredObject = cv2.morphologyEx(filteredObject, cv2.MORPH_OPEN, mKernel) 
			filteredObject = cv2.dilate(filteredObject, miniKernel, iterations = 1) 
			filteredObjectCoord = numpy.where(filteredObject == 1) 
			filteredShapes[timepoint][filteredObjectCoord] = k 
	# Return filtered object shapes: 
	return filteredShapes 




## DEFINE FUNCTION to remove zero-area objects: 

def remove_zero_area_objects(trackedObjects): 
	Ntimes = trackedObjects.shape[0] 
	Nobjects = numpy.amax(trackedObjects) 
	filteredObjects = numpy.zeros(trackedObjects.shape, dtype = 'int32') 
	newIndex = 1 
	for objectIndex in range(1, Nobjects + 1): 
		keepObject = True 
		for t in range(Ntimes): 
			if numpy.where(trackedObjects[t] == objectIndex)[0].shape[0] == 0: keepObject = False 
		if keepObject: 
			filteredObjects[trackedObjects == objectIndex] = newIndex 
			newIndex += 1 
		else: 
			print('\t(!) Zero-area nucleus: \t', objectIndex) 
	return filteredObjects 




## DEFINE FUNCTION to compute centroids for all nuclei over time: 

def compute_centroids_over_time(trackedObjects): 
	# Create data structures: 
	Ntimes = trackedObjects.shape[0] 
	Nobjects = numpy.amax(trackedObjects)
	x_coord = numpy.zeros((Ntimes, Nobjects), dtype = 'float64') 
	y_coord = numpy.zeros((Ntimes, Nobjects), dtype = 'float64') 
	# Go through all time points & compute centroids: 
	for t in range(Ntimes): 
		centroids = define_centroids(trackedObjects[t]) 
		x_coord[t, : ] = centroids[ : , 0] 
		y_coord[t, : ] = centroids[ : , 1] 
	# Return centroid coordinates: 
	return x_coord, y_coord 




## DEFINE FUNCTION to extract expression signal & background for an individual nucleus: 

def nucleus_quantification(nucleusCoord, nucleusMask, intensities, spotRadius, spotDepth): 
	
	### Define 3D kernel for spot detection: 
	halfSpotDepth = int(spotDepth / 2) 
	spotDiameter = 2 * spotRadius + 1 
	spotKernel = numpy.ones((spotDiameter, spotDiameter), dtype = 'int64') 
	spotKernel = spotKernel * cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spotDiameter, spotDiameter)) 
	spotKernel3D = numpy.zeros((spotDepth, spotDiameter, spotDiameter), dtype = 'int64') 
	for k in range(spotDepth): spotKernel3D[k] = spotKernel 
	spotKernel3D = numpy.where(spotKernel3D == 1) 
	spotSize = spotKernel3D[0].shape[0] 
	
	### Find spot with maximum intensity: 
	Nz, Ny, Nx = intensities.shape 
	spotIntensities = numpy.zeros(((Nz - spotDepth + 1), Ny, Nx), dtype = 'int64') 
	numberPositions = nucleusCoord[0].shape[0] 
	# Go through all X-Y positions: 
	for position in range(numberPositions): 
		Y_coord = nucleusCoord[0][position] 
		X_coord = nucleusCoord[1][position] 
		# Go through all Z positions: 
		for Z_coord in range(Nz - spotDepth + 1): 
			spotCoord_Z = spotKernel3D[0] + Z_coord 
			spotCoord_Y = spotKernel3D[1] + Y_coord 
			spotCoord_X = spotKernel3D[2] + X_coord 
			spotIntensityDistribution = intensities[spotCoord_Z, spotCoord_Y, spotCoord_X] 
			spotIntensitySum = numpy.sum(spotIntensityDistribution) 
			spotIntensities[Z_coord, Y_coord, X_coord] = spotIntensitySum 
	# Find position with maximum signal: 
	maxSignal = numpy.amax(spotIntensities) 
	maxPosition = numpy.where(spotIntensities == maxSignal) 
	maxPosition_Z = maxPosition[0][0] 
	maxPosition_Y = maxPosition[1][0] 
	maxPosition_X = maxPosition[2][0] 
	maxPositionCoord = numpy.array([maxPosition_Z, maxPosition_Y, maxPosition_X]) 
	# Extract intensities distribution at maximum signal position: 
	spotCoord_Z = spotKernel3D[0] + maxPosition_Z 
	spotCoord_Y = spotKernel3D[1] + maxPosition_Y 
	spotCoord_X = spotKernel3D[2] + maxPosition_X 
	maxSpotIntensityDistribution = intensities[spotCoord_Z, spotCoord_Y, spotCoord_X].flatten() 
	# Computer coordinates of spot center: 
	spotCenter_Z = maxPosition_Z + int(spotDepth / 2) 
	spotCenter_Y = maxPosition_Y + spotRadius 
	spotCenter_X = maxPosition_X + spotRadius 
	spotCenterCoord = [spotCenter_Z, spotCenter_Y, spotCenter_X] 

	### Compute background estimate (2nd-brightest spot, after exclusion of the brighest one): 
	darkRadius = round(2.0 * spotDiameter) 
	darkDepth = round(1.5 * spotDepth) 
	# Black out brightest spot: 
	spotIntensities[ 
				max(0, (maxPosition_Z - darkDepth)) : min(Nz, (maxPosition_Z + darkDepth)), 
				max(0, (maxPosition_Y - darkRadius)) : min(Ny, (maxPosition_Y + darkRadius)), 
				max(0, (maxPosition_X - darkRadius)) : min(Nx, (maxPosition_X + darkRadius)) 
				] = 0  
	# Find position with 2nd-highest signal: 
	backSignal = numpy.amax(spotIntensities) 
	backPosition = numpy.where(spotIntensities == backSignal) 
	backPosition_Z = backPosition[0][0] 
	backPosition_Y = backPosition[1][0] 
	backPosition_X = backPosition[2][0] 
	backPositionCoord = numpy.array([backPosition_Z, backPosition_Y, backPosition_X]) 
	# Extract intensities distribution at 2nd-highest signal position: 
	backSpotCoord_Z = spotKernel3D[0] + backPosition_Z 
	backSpotCoord_Y = spotKernel3D[1] + backPosition_Y 
	backSpotCoord_X = spotKernel3D[2] + backPosition_X 
	backIntensityDistribution = intensities[backSpotCoord_Z, backSpotCoord_Y, backSpotCoord_X].flatten() 

	### Return background intensity distribution, spot intensity distribution, spot coordinates: 
	return backIntensityDistribution, maxSpotIntensityDistribution, spotCenterCoord 




## DEFINE FUNCTION to extract expression signal & background over time series: 

def extract_transcription_signal(trackedObjects, exprSignal, spotRadius, spotDepth):  
	# Create data structures to store expression data: 
	Nz, Ny, Nx = exprSignal.shape 
	Nobjects = numpy.amax(trackedObjects) 
	nuclearBackground = numpy.zeros(Nobjects, dtype = 'float64') 
	nuclearSignal = numpy.zeros(Nobjects, dtype = 'float64') 
	nuclearSpotCoord = numpy.zeros((Nobjects, 3), dtype = 'int64') 
	# Go through all nuclei, extract background & signal: 
	for nucleus in range(1, Nobjects + 1): 
		nucleusCoord = numpy.where(trackedObjects == nucleus) 
		nucleusMask = numpy.where(trackedObjects == nucleus, 1, -1) 
		nucleusData = nucleus_quantification(
										nucleusCoord, 
										nucleusMask, 
										exprSignal, 
										spotRadius, 
										spotDepth 
										) 
		backgroundDistribution, signalDistribution, spotCoordinates = nucleusData 
		# Save mean values for background & signal: 
		nuclearBackground[nucleus - 1] = numpy.mean(backgroundDistribution) 
		nuclearSignal[nucleus - 1] = numpy.mean(signalDistribution) 
		# Save spot coordinates: 
		nuclearSpotCoord[nucleus - 1, :] = spotCoordinates 
	# Return background, signal and spot coordinates: 
	return nuclearBackground, nuclearSignal, nuclearSpotCoord 




## DEFINE FUNCTION to fit Gaussian to background distribution: 

def fit_gaussian(background, backStart, backStop, Nbins, histThresh): 
	backgroundDist = background[backStart : backStop, ].flatten() 
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




## DEFINE FUNCTION to define signal threshold based on desired FDR: 

def set_signal_threshold(distMean, distStdDev, fdr): 	
	percentile = 1 - fdr 
	signalThresh = stats.norm.ppf(percentile, distMean, distStdDev) 
	return signalThresh 




## DEFINE FUNCTION to correct signal intensities: 

def correct_signal_intensity(signal, signalThresh): 
	signal = signal - signalThresh 
	signal = numpy.where(signal > 0, signal, 0) 
	return signal 




## DEFINE FUNCTION to plot all profiles in one figure: 

def plot_individual_profiles(profiles, signalThresh, plotName, title):
	# Define color palette: 
	palette = 'hsv' 
	Ncols = 50 
	colors = range(Ncols)
	colorPalette = plt.cm.get_cmap(palette, Ncols)
	# Define axes: 
	Ntimes, Nnuclei = profiles.shape
	timeScale = range(Ntimes)
	minX = 0
	maxX = Ntimes
	minY = 0.9 * numpy.amin(profiles) 
	maxY = 1.1 * numpy.amax(profiles) 
	plt.figure(1, (10.5, 7))
	for k in range(Nnuclei):
		plt.plot(
				timeScale,
				profiles[:, k],
				linewidth = 1,
				color = tuple(colorPalette(random.sample(colors, 1))[0][:3])
				)
	# Add signal threshold line: 
	plt.hlines( 
			signalThresh, 
			minX, 
			maxX, 
			linestyle = 'dashed',
			linewidth = 1,
			color = 'black', 
			zorder = 2 * Nnuclei 
			) 
	# Label axes: 
	plt.axis([minX, maxX, minY, maxY])
	plt.title(title)
	plt.xlabel('Developmental Time (Frames)')
	plt.ylabel('Fluorescence intensity (a.u.)')
	# Save figure to file: 
	plt.savefig(plotName, format = 'png', dpi = 300)
	plt.close()
	return




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
	plt.savefig(plotName, format = 'png', dpi = 300)
	plt.close()
	return




## DEFINE FUNCTION to plot signal histogram: 

def plot_signal_histogram(signal, plotName, title, signalThresh, Nbins, histColor): 
	# Get axis dimensions: 
	minSignal = numpy.amin(signal) 
	maxSignal = numpy.amax(signal) 
    # Create histogram: 
	plt.figure(1, (10.5, 7))
	plt.title(title) 
	signalValues = signal.flatten() 
	hDens, hBins, hPat = plt.hist( 
							signalValues, 
							bins = Nbins, 
							range = (minSignal, maxSignal), 
							density = True, 
							histtype = 'stepfilled', 
							color = histColor 
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
	plt.savefig(plotName, format = 'png', dpi = 300)
	plt.close()
	return




## DEFINE FUNCTION to generate total output heatmap: 

def total_output_heatmap(nucleiMask, expression, plotName, title, colorMap): 
	# Compute total output per nucleus: 
	totalOutput = expression.mean(0) 
	totalOutput = numpy.array((totalOutput / numpy.amax(totalOutput) * 255), dtype = 'int') 
	# Generate heatmap: 
	globalMask = numpy.where(nucleiMask > 0, 255, 0) 
	outputHeatmap = numpy.zeros(nucleiMask.shape) 
	Nnuclei = numpy.amax(nucleiMask) 
	for N in range(Nnuclei): 
		outputHeatmap[nucleiMask == N + 1] = totalOutput[N] 
	# Create plot: 
	plt.figure(1, (10.5, 7)) 
	plt.title(title) 
	plt.imshow(
			globalMask, 
			cmap = 'Greys',
			alpha = 0.5 
			) 
	plt.imshow( 
			outputHeatmap, 
			cmap = colorMap, 
			vmin = 0,
			vmax = 0.75 * 255, 
			alpha = 0.6 
			) 
	# Save figure to file: 
	plt.savefig(plotName, format = 'png', dpi = 300) 
	plt.close()
	return 




## DEFINE FUNCTION to run subprocess: 

def run_subprocess(arguments, outName):
    outFile = open(outName, 'w')
    errName = outName + '.stderr'
    errFile = open(errName, 'w')
    subprocess.Popen( 
					arguments, 
					stdout = outFile, 
					stderr = errFile 
					).wait()
    outFile.close()
    errFile.close()
    os.remove(errName)
    return




## DEFINE FUNCTION to generate pseudo-colored embryo movie: 

def generate_embryo_movie(embryoMasks, expression, movieName, colorMap): 
	# Get dataset dimensions: 
	Nnuclei = numpy.amax(embryoMasks) 
	Ntimes = expression.shape[0] 
	# Normalize expression values: 
	normExpr = numpy.array((expression / numpy.amax(expression) * 255), dtype = 'int') 
	# Generate heatmap for each time point: 
	tempFilesList = [] 
	for t in range(Ntimes): 
		globalMask = numpy.where(embryoMasks[t] > 0, 255, 0) 
		exprHeatmap = numpy.zeros(embryoMasks[0].shape) 
		for N in range(Nnuclei): 
			exprHeatmap[embryoMasks[t] == N + 1] = normExpr[t, N] 
		plt.figure(1, (10.5, 7)) 
		timeLabel = str(t+1).rjust(3, '0') 
		plt.title('Time %s' % timeLabel) 
		plt.imshow( 
				globalMask, 
				cmap = 'Greys', 
				alpha = 0.5 
				) 
		plt.imshow( 
				exprHeatmap, 
				cmap = colorMap, 
				vmin = 0,
				vmax = 0.6 * 255, 
				alpha = 0.6 
				) 
		figFileName = '%s_%s.png' %(movieName, timeLabel) 
		plt.savefig(figFileName, format = 'png', dpi = 240) 
		plt.close() 
		tempFilesList.append(figFileName) 
	# Generate MP4 movie & clean up individual frames: 
	ffmpeg_logfile = movieName + '_ffmpeg.txt' 
	ffmpegArgs = [
				'ffmpeg',
				'-r', '60',
				'-i', movieName + '_%03d.png',
				'-vcodec', 'libx264',
				'-crf', '25',
				'-pix_fmt', 'yuv420p',
				'-vf', 'setpts=10*PTS',
				movieName + '.mp4'
				]
	run_subprocess(ffmpegArgs, ffmpeg_logfile) 
	for fileName in tempFilesList: os.remove(fileName) 
	os.remove(ffmpeg_logfile) 
	return 




## DEFINE FUNCTION to generate segmentation masks movie:  

def generate_segmentation_movie(smooth_histones, embryoMasks, movieName): 

	# Get dataset dimensions: 
	Ntimes = smooth_histones.shape[0] 
	segMask = numpy.where(embryoMasks > 0, 1, 0) 
	# Generate histone / segmentation mask overlay for each time point: 
	tempFilesList = [] 
	for t in range(Ntimes): 
		plt.figure(1, (10.5, 7)) 
		timeLabel = str(t+1).rjust(3, '0') 
		plt.title('Time %s' % timeLabel) 
		plt.imshow(smooth_histones[t]) 
		plt.imshow( 
				segMask[t], 
				cmap = 'Greys', 
				alpha = 0.2 
				) 
		figFileName = '%s_%s.png' %(movieName, timeLabel) 
		plt.savefig(figFileName, format = 'png', dpi = 240) 
		plt.close() 
		tempFilesList.append(figFileName) 
	# Generate MP4 movie & clean up individual frames: 
	ffmpeg_logfile = movieName + '_ffmpeg.txt' 
	ffmpegArgs = [
				'ffmpeg',
				'-r', '60',
				'-i', movieName + '_%03d.png',
				'-vcodec', 'libx264',
				'-crf', '25',
				'-pix_fmt', 'yuv420p',
				'-vf', 'setpts=10*PTS',
				movieName + '.mp4'
				]
	run_subprocess(ffmpegArgs, ffmpeg_logfile) 
	for fileName in tempFilesList: os.remove(fileName) 
	os.remove(ffmpeg_logfile) 
	return 




## DEFINE FUNCTION to generate nucleus tracking movie:  

def generate_nucleus_tracking_movie(embryoMasks, movieName): 
	# Get dataset dimensions: 
	Ntimes = embryoMasks.shape[0] 
	Nnuclei = numpy.amax(embryoMasks) 
	# Randomly re-assign nucleus labels: 
	newIDs = {}; newIDs[1] = 600 
	for k in range(2, 1 + Nnuclei): newIDs[k] = random.randrange(100, 600) 
	# Re-label nuclei: 
	newMasks = numpy.zeros(embryoMasks.shape, dtype = 'int32') 
	for k in range(1, 1 + Nnuclei): newMasks[embryoMasks == k] = newIDs[k] 
	# Generate randomly colored segmentation masks: 
	tempFilesList = [] 
	for t in range(Ntimes): 
		plt.figure(1, (10.5, 7)) 
		timeLabel = str(t+1).rjust(3, '0') 
		plt.title('Time %s' % timeLabel) 
		plt.imshow(newMasks[t]) 
		figFileName = '%s_%s.png' %(movieName, timeLabel) 
		plt.savefig(figFileName, format = 'png', dpi = 240) 
		plt.close() 
		tempFilesList.append(figFileName) 
	# Generate MP4 movie & clean up individual frames: 
	ffmpeg_logfile = movieName + '_ffmpeg.txt' 
	ffmpegArgs = [
				'ffmpeg',
				'-r', '60',
				'-i', movieName + '_%03d.png',
				'-vcodec', 'libx264',
				'-crf', '25',
				'-pix_fmt', 'yuv420p',
				'-vf', 'setpts=10*PTS',
				movieName + '.mp4'
				]
	run_subprocess(ffmpegArgs, ffmpeg_logfile) 
	for fileName in tempFilesList: os.remove(fileName) 
	os.remove(ffmpeg_logfile) 
	return 




## DEFINE FUNCTION to generate spot detection movie:  

def generate_spot_detection_movie(images, embryoMasks, spots, movieName): 

	# Get dataset dimensions: 
	Ntimes = images.shape[0] 
	segMask = numpy.where(embryoMasks > 0, 1, 0) 
	minSignal = numpy.amin(images) 
	maxSignal = numpy.amax(images) 
	lowerSig = minSignal + 0.08 * (maxSignal - minSignal) 
	upperSig = maxSignal - 0.10 * (maxSignal - minSignal) 
	# Generate fluorescence / spot detection overlay for each time point: 
	tempFilesList = [] 
	for t in range(Ntimes): 
		plt.figure(1, (10.5, 7)) 
		timeLabel = str(t+1).rjust(3, '0') 
		plt.title('Time %s' % timeLabel) 
		projection = numpy.amax(images[t], 0) 
		plt.imshow( 
				projection, 
				cmap = 'hot', 
				vmin = lowerSig, 
				vmax = upperSig 
				) 
		plt.imshow( 
				segMask[t], 
				cmap = 'Greys', 
				alpha = 0.10 
				) 
		spots_map = numpy.zeros(segMask[t].shape) 
		for spot_index in range(spots.shape[1] - 1): 
			spotY, spotX = spots[t, spot_index, 1:] 
			spots_map[spotY-1 : spotY+2, spotX-1 : spotX+2] = 1 
		plt.imshow( 
				spots_map, 
				cmap = 'Greys', 
				alpha = 0.25 
				) 
		figFileName = '%s_%s.png' %(movieName, timeLabel) 
		plt.savefig(figFileName, format = 'png', dpi = 240) 
		plt.close() 
		tempFilesList.append(figFileName) 
	# Generate MP4 movie & clean up individual frames: 
	ffmpeg_logfile = movieName + '_ffmpeg.txt' 
	ffmpegArgs = [
				'ffmpeg',
				'-r', '60',
				'-i', movieName + '_%03d.png',
				'-vcodec', 'libx264',
				'-crf', '25',
				'-pix_fmt', 'yuv420p',
				'-vf', 'setpts=10*PTS',
				movieName + '.mp4'
				]
	run_subprocess(ffmpegArgs, ffmpeg_logfile) 
	for fileName in tempFilesList: os.remove(fileName) 
	os.remove(ffmpeg_logfile) 
	return 




## DEFINE FUNCTION to write segmentation masks to file: 

def write_segmentation_files(trackedObjects, nameRoot): 
	Ntimes = trackedObjects.shape[0] 
	for t in range(Ntimes): 
		outFileName = nameRoot + '_%s.txt' % str(t+1).rjust(3, '0') 
		numpy.savetxt( 
				outFileName, 
				trackedObjects[t], 
				fmt = '%i', 
				delimiter = '\t', 
				newline = '\n' 
				) 
	return 




## DEFINE FUNCTION to write nucleus coordinates to files: 

def write_nucleus_coordinates_files(x_coord, y_coord, fileRoot): 
	# Point to data structures: 
	dataSet = { 
			'x_coord': x_coord, 
			'y_coord': y_coord
			} 
	fileSuffix = { 
			'x_coord': '_Nucleus_X.txt', 
			'y_coord': '_Nucleus_Y.txt' 
			} 
	# Extract parameters & Generate file header: 
	Ntimes, Nuclei = x_coord.shape 
	header = ['Nucleus'] 
	for t in range(Ntimes): header.append(str(t+1).rjust(3, '0')) 
	header = '\t'.join(header) + '\n' 
	# Write data to files: 
	for dataType in dataSet: 
		data = dataSet[dataType] 
		fileName = fileRoot + fileSuffix[dataType] 
		file = open(fileName, 'w') 
		file.write(header) 
		for k in range(Nuclei): 
			line = [str(k+1).rjust(6, '0')] + numpy.array(data[:, k], dtype = 'str').tolist() 
			file.write('\t'.join(line) + '\n') 
		file.close() 
	# Return: 
	return 




## DEFINE FUNCTION to write expression data to files: 

def write_expression_files(background, signal, fileRoot): 
	# Point to data structures: 
	dataSet = { 
			'background': background, 
			'signal': signal 
			} 
	fileSuffix = { 
			'background': '_Background.txt', 
			'signal': '_Signal.txt' 
			} 
	# Extract parameters & Generate file header: 
	Ntimes, Nuclei = background.shape 
	header = ['Nucleus'] 
	for t in range(Ntimes): header.append(str(t+1).rjust(3, '0')) 
	header = '\t'.join(header) + '\n' 
	# Write data to files: 
	for dataType in dataSet: 
		data = dataSet[dataType] 
		fileName = fileRoot + fileSuffix[dataType] 
		file = open(fileName, 'w') 
		file.write(header) 
		for k in range(Nuclei): 
			line = [str(k+1).rjust(6, '0')] + numpy.array(data[:, k], dtype = 'str').tolist() 
			file.write('\t'.join(line) + '\n') 
		file.close() 
	# Return: 
	return 







##########################################################################################################################################################################
##########################################################################################################################################################################












#### Arguments & Options: 
parser = OptionParser() 
parser.add_option('--first-frame', '-F', type = 'int', default = None) 
parser.add_option('--last-frame', '-L', type = 'int', default = None) 
parser.add_option('--channels', '-C', type = 'str', default = '1,0,2') 
parser.add_option('--remove-edges', '-E', action = 'store_true', default = False) 
parser.add_option('--min-edge-area', '-A', type = 'int', default = 3000) 
parser.add_option('--segPlanes', '-P', type = 'int', default = 12) 
parser.add_option('--gaussian-kernel', '-G', type = 'int', default = 15) 
parser.add_option('--morphology-kernel', '-M', type = 'int', default = 5) 
parser.add_option('--distance-threshold', '-D', type = 'float', default = 6) 
parser.add_option('--size-threshold', '-S', type = 'float', default = 3) 
parser.add_option('--border-fraction', '-B', type = 'float', default = 0.01) 
parser.add_option('--time-kernel', '-T', type = 'int', default = 7) 
parser.add_option('--spot-radius', '-R', type = 'int', default = 3) 
parser.add_option('--spot-depth', '-d', type = 'int', default = 4) 
parser.add_option('--green-background-start', type = 'int', default = 0) 
parser.add_option('--green-background-stop', type = 'int', default = None) 
parser.add_option('--red-background-start', type = 'int', default = 25) 
parser.add_option('--red-background-stop', type = 'int', default = None) 
parser.add_option('--fdr', type = 'float', default = 0.0000000001) 
parser.add_option('--embryo-map', type = 'int', default = 80)
parser.add_option('--segmentation-movies', action = 'store_true', default = False) 
parser.add_option('--spot-detection-movies', action = 'store_true', default = False) 
parser.add_option('--cores', type = 'int', default = 8) 
(options, args) = parser.parse_args() 
channelsOrder = options.channels.split(',') 
options.channels = options.channels.replace(',', ' ') 
cziFileName = args[0] 



#### Load CZI image file: 
print('\nLoading images...') 
with czi.CziFile(cziFileName) as cziData: movie = cziData.asarray() 
movie = movie[0, :, :, options.first_frame : options.last_frame, :, :, :, :] 
movie = numpy.swapaxes(movie, 1, 2) 
movieDims = movie.shape 
NumberTimepoints, NumberChannels, Zplanes, Ypixels, Xpixels = movieDims[1:6] 
fileNameRoot = cziFileName.rsplit('.', 1)[0] 
print('''
CZI file: \t\t%s
Image size: \t\t%s x %s 
Z-stack depth: \t\t%s 
Time points: \t\t%s
Channels: \t\t%s 
''' %(cziFileName.rstrip('/').split('/')[-1], Xpixels, Ypixels, Zplanes, NumberTimepoints, NumberChannels)) 
if NumberChannels != 3: print('\n\t(!) File format error: Expecting 3 channels \n'); sys.exit() 



#### Report all options: 
if options.first_frame == None: options.first_frame = 'First' 
if options.last_frame == None: options.last_frame = 'Last' 
if options.green_background_stop == None: green_background_stop = 'Last' 
else: green_background_stop = options.green_background_stop 
if options.red_background_stop == None: red_background_stop = 'Last' 
else: red_background_stop = options.red_background_stop 
optionValues = tuple(( 
				options.first_frame, 
				options.last_frame, 
				options.channels, 
				options.remove_edges, 
				options.min_edge_area, 
				options.segPlanes, 
				options.gaussian_kernel,
				options.morphology_kernel, 
				options.distance_threshold, 
				options.size_threshold,
				options.border_fraction * 100, 
				options.time_kernel, 
				options.spot_radius,
				options.spot_depth, 
				options.green_background_start + 1, 
				green_background_stop, 
				options.red_background_start + 1, 
				red_background_stop, 
				options.fdr, 
				options.embryo_map, 
				options.cores 
				))
print('''
First frame:        \t%s 
Last frame:         \t%s 
Channels (RGB):     \t%s
Remove edges:       \t%s 
Min edge area:      \t%s
Segment Z-planes:   \t%s
Gaussian kernel:    \t%s
Morpho kernel:      \t%s
Distance threshold: \t%.1f 
Size threshold:     \t%s 
Exclude border:     \t%.1f%%
Time kernel:        \t%s 
Spot radius:        \t%s 
Spot depth:         \t%s 
GFP Background:     \t%s - %s
RFP Background:     \t%s - %s
False Discovery Rate:\t%.1e 
Embryo map:         \t%s 
Number cores:       \t%s 
''' % optionValues) 



#### Remove embryo edges (optional) & lipid blobs from individual images: 
print('Filtering artifacts...') 
histoneChannelIndex = int(channelsOrder[2]) 
filteringData = remove_artifacts(
							movie[0, :, :, :, :, :, 0], 
							histoneChannelIndex, 
							options.gaussian_kernel, 
							options.morphology_kernel, 
							options.remove_edges, 
							options.min_edge_area 
							) 
artifactsMasks, movie[0, :, :, :, :, :, 0] = filteringData 
del filteringData 



#### Segment nuclei throughout the movie: 
print('Segmenting nuclei...') 

# Initial segmentation on individual time points: 
nucleiChannel = movie[0, :, histoneChannelIndex, :, :, :, 0] 
objectSeedRegions = numpy.zeros((NumberTimepoints, Ypixels, Xpixels), dtype = 'int32') 
segmentedNuclei = numpy.zeros((NumberTimepoints, Ypixels, Xpixels), dtype = 'int32') 
smoothHistoneImages = numpy.zeros((NumberTimepoints, Ypixels, Xpixels), dtype = 'int32') 
for timepoint in range(NumberTimepoints): 
	segmentationData = segment_nuclei(
								nucleiChannel[timepoint, :, :, :], 
								options.segPlanes, 
								options.gaussian_kernel, 
								options.morphology_kernel, 
								options.distance_threshold 
								) 
	smoothHistoneImages[timepoint, :, :], objectSeedRegions[timepoint, :, :], segmentedNuclei[timepoint, :, :] = segmentationData 
del segmentationData 


# Error correction using previous / following time points: 
print('Error correction...') 

# Recover nuclei that failed to be identified in one or two consecutive time points: 
for timepoint in range(1, NumberTimepoints): 
	segmentationMask = segmentedNuclei[timepoint, :, :] 
	previousMask = objectSeedRegions[timepoint - 1, :, :] 
	correctedMask = recover_dropout_nuclei( 
										segmentationMask, 
										previousMask
										) 
	segmentedNuclei[timepoint, :, :] = correctedMask 

for timepoint in range(NumberTimepoints - 1): 
	segmentationMask = segmentedNuclei[timepoint, :, :] 
	followingMask = objectSeedRegions[timepoint + 1, :, :] 
	correctedMask = recover_dropout_nuclei( 
										segmentationMask, 
										followingMask
										) 
	segmentedNuclei[timepoint, :, :] = correctedMask 

# Correct occasional fusions of neighboring nuclei, occuring in one or two consecutive time points: 
for timepoint in range(1, NumberTimepoints): 
	segmentationMask = segmentedNuclei[timepoint, :, :] 
	previousMarkers = objectSeedRegions[timepoint - 1, :, :] 
	correctedMask = resolve_fusions(
							previousMarkers, 
							segmentationMask, 
							smoothHistoneImages[timepoint, :, :], 
							options.morphology_kernel 
							) 
	segmentedNuclei[timepoint, :, :] = correctedMask 

for timepoint in range(NumberTimepoints - 1): 
	segmentationMask = segmentedNuclei[timepoint, :, :] 
	nextMarkers = objectSeedRegions[timepoint + 1, :, :] 
	correctedMask = resolve_fusions(
							nextMarkers, 
							segmentationMask, 
							smoothHistoneImages[timepoint, :, :], 
							options.morphology_kernel 
							) 
	segmentedNuclei[timepoint, :, :] = correctedMask 

# Filter out any outliers in the object size distribution:  
print('Filtering objects...') 
for timepoint in range(NumberTimepoints): 
	segmentedNuclei[timepoint, :, :] = remove_size_outliers( 
													segmentedNuclei[timepoint, :, :], 
													options.size_threshold 
													)



#### Track individual nuclei throughout the movie:  
print('Tracking nuclei...') 
trackedNuclei = track_nuclei_munkres(
								segmentedNuclei, 
								options.border_fraction 
								) 
numberNucleiTracked = numpy.amax(trackedNuclei[0]) 



#### Apply temporal filtering to nucleus shapes: 
print('Time filtering...') 
trackedNuclei = features_temporal_filter( 
									trackedNuclei, 
									options.time_kernel, 
									options.morphology_kernel 
									) 



### Remove zero-area nuclei (due to tracking errors): 
trackedNuclei = remove_zero_area_objects(trackedNuclei) 



#### Compute centroids for all nuclei over time: 
nucleusCentroidsX, nucleusCentroidsY = compute_centroids_over_time(trackedNuclei) 



#### Extract signal from the 2 transcription channels: 
print('Extracting signal...') 
GFPchannel = int(channelsOrder[1]) 
RFPchannel = int(channelsOrder[0]) 
Nnuclei = numpy.amax(trackedNuclei[0]) 

# Extract transcription signal from GFP channel: 
pool = multiprocessing.Pool(options.cores) 
allFunctionArguments = [] 
for timepoint in range(NumberTimepoints): 
	currentArgs = ( 
				trackedNuclei[timepoint], 
				movie[0, timepoint, GFPchannel, :, :, :, 0], 
				options.spot_radius, 
				options.spot_depth 
				) 
	allFunctionArguments.append(currentArgs) 
quantificationData = pool.starmap(extract_transcription_signal, allFunctionArguments) 
pool.close() 
pool.join() 
GFPbackground = numpy.zeros((NumberTimepoints, Nnuclei), dtype = 'float64') 
GFPsignal = numpy.zeros((NumberTimepoints, Nnuclei), dtype = 'float64') 
GFPspots = numpy.zeros((NumberTimepoints, Nnuclei, 3), dtype = 'int64') 
for timepoint in range(NumberTimepoints): 
	exprData = quantificationData[timepoint] 
	GFPbackground[timepoint], GFPsignal[timepoint], GFPspots[timepoint] = exprData 

# Extract transcription signal from RFP channel: 
pool = multiprocessing.Pool(options.cores) 
allFunctionArguments = [] 
for timepoint in range(NumberTimepoints): 
	currentArgs = ( 
				trackedNuclei[timepoint], 
				movie[0, timepoint, RFPchannel, :, :, :, 0], 
				options.spot_radius, 
				options.spot_depth 
				) 
	allFunctionArguments.append(currentArgs) 
quantificationData = pool.starmap(extract_transcription_signal, allFunctionArguments) 
pool.close() 
pool.join() 
RFPbackground = numpy.zeros((NumberTimepoints, Nnuclei), dtype = 'float64') 
RFPsignal = numpy.zeros((NumberTimepoints, Nnuclei), dtype = 'float64') 
RFPspots = numpy.zeros((NumberTimepoints, Nnuclei, 3), dtype = 'int64') 
for timepoint in range(NumberTimepoints): 
	exprData = quantificationData[timepoint] 
	RFPbackground[timepoint], RFPsignal[timepoint], RFPspots[timepoint] = exprData 



#### Model background distribution & Determine threshold for desired FDR: 
numberBins = 250 
densityThreshold = 0.01 
GFPbackMean, GFPbackStd = fit_gaussian( 
								GFPbackground, 
								options.green_background_start, 
								options.green_background_stop, 
								numberBins, 
								densityThreshold
								) 
GFPsignalThresh = set_signal_threshold( 
								GFPbackMean, 
								GFPbackStd, 
								options.fdr 
								) 
RFPbackMean, RFPbackStd = fit_gaussian( 
								RFPbackground, 
								options.red_background_start, 
								options.red_background_stop, 
								numberBins, 
								densityThreshold
								) 
RFPsignalThresh = set_signal_threshold( 
								RFPbackMean, 
								RFPbackStd, 
								options.fdr 
								) 



#### Compute corrected intensities for both channels, for desired FDR: 
GFPexpression = correct_signal_intensity(GFPsignal, GFPsignalThresh) 
RFPexpression = correct_signal_intensity(RFPsignal, RFPsignalThresh) 



#### Write output data to files: 
print('Writing files...') 
outDir = 'Transcription_Measurements/' 
segDir = 'Nuclei_Segmentation/' 
if outDir[:-1] not in os.listdir(os.getcwd()): os.mkdir(outDir) 
if segDir[:-1] not in os.listdir(outDir): os.mkdir(outDir + segDir) 
segFileNameRoot = outDir + segDir + fileNameRoot + '_Nuclei' 
write_segmentation_files(trackedNuclei, segFileNameRoot) 
coordFileNameRoot = outDir + fileNameRoot 
write_nucleus_coordinates_files(nucleusCentroidsX, nucleusCentroidsY, coordFileNameRoot) 
GFPfileRoot = outDir + fileNameRoot + '_GFP' 
write_expression_files(GFPbackground, GFPsignal, GFPfileRoot) 
RFPfileRoot = outDir + fileNameRoot + '_RFP' 
write_expression_files(RFPbackground, RFPsignal, RFPfileRoot) 



#### Output summary plots & movies: 

# Plot all individual signal & background profiles: 
GFPbackProfilesName = GFPfileRoot + '_Background_Traces.png' 
GFPbackProfilesTitle = 'GFP Channel Background' 
plot_individual_profiles( 
					GFPbackground, 
					GFPsignalThresh, 
					GFPbackProfilesName, 
					GFPbackProfilesTitle 
					) 
RFPbackProfilesName = RFPfileRoot + '_Background_Traces.png' 
RFPbackProfilesTitle = 'RFP Channel Background' 
plot_individual_profiles(
					RFPbackground, 
					RFPsignalThresh, 
					RFPbackProfilesName, 
					RFPbackProfilesTitle 
					) 
GFPsignalProfilesName = GFPfileRoot + '_Signal_Traces.png' 
GFPsignalProfilesTitle = 'GFP Channel Signal' 
plot_individual_profiles( 
					GFPsignal, 
					GFPsignalThresh, 
					GFPsignalProfilesName, 
					GFPsignalProfilesTitle 
					) 
RFPsignalProfilesName = RFPfileRoot + '_Signal_Traces.png' 
RFPsignalProfilesTitle = 'RFP Channel Signal' 
plot_individual_profiles( 
					RFPsignal, 
					RFPsignalThresh, 
					RFPsignalProfilesName, 
					RFPsignalProfilesTitle 
					) 

# Plot background & signal histograms for both channels: 
GFPbackHistName = GFPfileRoot + '_Background_Histogram.png' 
GFPbackHistTitle = 'GFP Channel Background' 
plot_background_histogram( 
					GFPbackground, 
					GFPbackHistName, 
					GFPbackHistTitle, 
					GFPbackMean, 
					GFPbackStd, 
					GFPsignalThresh, 
					numberBins, 
					'darkcyan' 
					) 
RFPbackHistName = RFPfileRoot + '_Background_Histogram.png' 
RFPbackHistTitle = 'RFP Channel Background' 
plot_background_histogram( 
					RFPbackground, 
					RFPbackHistName, 
					RFPbackHistTitle, 
					RFPbackMean, 
					RFPbackStd, 
					RFPsignalThresh, 
					numberBins, 
					'darkmagenta' 
					) 
GFPsignalHistName = GFPfileRoot + '_Signal_Histogram.png' 
GFPsignalHistTitle = 'GFP Channel Signal' 
plot_signal_histogram( 
					GFPsignal, 
					GFPsignalHistName, 
					GFPsignalHistTitle, 
					GFPsignalThresh, 
					numberBins, 
					'cyan' 
					) 
RFPsignalHistName = RFPfileRoot + '_Signal_Histogram.png' 
RFPsignalHistTitle = 'RFP Channel Signal' 
plot_signal_histogram( 
					RFPsignal, 
					RFPsignalHistName, 
					RFPsignalHistTitle, 
					RFPsignalThresh, 
					numberBins, 
					'magenta' 
					) 

# Plot total output heatmaps for both channels: 
GFPheatmapName = GFPfileRoot + '_Heatmap.png' 
GFPheatmapTitle = 'GFP Channel – Total Output Heatmap' 
total_output_heatmap( 
				trackedNuclei[options.embryo_map], 
				GFPexpression, 
				GFPheatmapName, 
				GFPheatmapTitle, 
				colorMap = 'Greens' 
				) 
RFPheatmapName = RFPfileRoot + '_Heatmap.png' 
RFPheatmapTitle = 'RFP Channel – Total Output Heatmap' 
total_output_heatmap( 
				trackedNuclei[options.embryo_map], 
				RFPexpression, 
				RFPheatmapName, 
				RFPheatmapTitle, 
				colorMap = 'Reds' 
				) 

# Generate pseudo-colored movies for each channel: 
GFPmovieName = GFPfileRoot + '_Movie' 
generate_embryo_movie( 
				trackedNuclei, 
				GFPexpression, 
				GFPmovieName, 
				colorMap = 'Greens' 
				) 
RFPmovieName = RFPfileRoot + '_Movie' 
generate_embryo_movie( 
				trackedNuclei, 
				RFPexpression, 
				RFPmovieName, 
				colorMap = 'Reds' 
				) 

# Generate segmentation masks & nucleus tracking novies (Optional): 
if options.segmentation_movies: 
	segMovieName = outDir + fileNameRoot + '_Segmentation' 
	generate_segmentation_movie( 
							smoothHistoneImages, 
							trackedNuclei, 
							segMovieName
							) 
	trackMovieName = outDir + fileNameRoot + '_Tracking' 
	generate_nucleus_tracking_movie( 
							trackedNuclei, 
							trackMovieName 
							) 

# Generate spot detection movies (Optional): 
if options.spot_detection_movies: 
	GFPimages = movie[0, :, GFPchannel, :, :, :, 0] 
	GFPmovieName = GFPfileRoot + '_GFP_Spots' 
	generate_spot_detection_movie( 
							GFPimages, 
							trackedNuclei, 
							GFPspots, 
							GFPmovieName 
							) 
	RFPimages = movie[0, :, RFPchannel, :, :, :, 0] 
	RFPmovieName = RFPfileRoot + '_RFP_Spots' 
	generate_spot_detection_movie( 
							RFPimages, 
							trackedNuclei, 
							RFPspots, 
							RFPmovieName 
							) 









#### Final report: 

print('\nNuclei analyzed: \t', numberNucleiTracked) 
print('GFP Threshold:     \t', '%.0f' % GFPsignalThresh) 
print('RFP Threshold:     \t', '%.0f' % RFPsignalThresh) 
endTime = time.time() 
runTime = (endTime - startTime) / 60 
print('\nRun time: \t\t%.0f min \n' % runTime) 


















