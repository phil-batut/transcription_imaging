# transcription_imaging
Processing and analysis of time-lapse single-cell transcription imaging in early Drosophila embryos (MS2/PP7 system)





Philippe J. BATUT 
Levine lab
Lewis-Sigler Institute
Princeton University 
pbatut@princeton.edu 


November 2021 





DOCUMENTATION – transcription_image_processing.py  





This code segments nuclei based on histone labeling and extracts MS2/PP7 transcription measurements 
from a Drosophila Cell Cycle 14 live imaging 4D dataset. 
It takes as input a CZI image file (Zeiss). Expects 3D stacks, time series, 3 channels. 






 DEPENDENCIES: 

		Python 3.6.5 				  https://www.python.org
		Scipy 1.0.1 				  https://www.scipy.org
		Numpy 1.14.2 				  http://www.numpy.org
 		czifile					      https://github.com/AllenCellModeling/czifile
		tifffile					    https://github.com/blink1073/tifffile 
		scikit-image 0.13.1 	http://scikit-image.org 
 		OpenCV 3.4.1				  https://opencv.org 
		StatsModels				    http://www.statsmodels.org 
		




USAGE: 

		$ python transcription_image_processing.py [OPTIONS] imageFile.czi

	



OPTIONS: 
 
		--first-frame (-F):			      Specify the first frame to load from the image file (0-based inclusive; Default: First) 
		--last-frame (-L):			      Specify the last frame to load from the image file (0-based exclusive; Default: Last) 
		--channels (-C):				      Order of the channels in the image file, in RGB order (Default: 1,0,2) 
		--remove-edges (-E):			    Filter out embryo edges from individual images prior to segmentation (Default: False) 
		--min-edge-area (-A):			    Minimum surface area, in pixels, for putative embryo edges to be filtered out (Default: 3,000) 
		--segPlanes (-P):			        Number of Z-planes (above and below median plane) to use for segmentation (Default: 12) 
		--gaussian-kernel (-G):			  Width of the Gaussian kernel used for smoothing prior to nuclei segmentation (Default: 15) 
		--morphology-kernel (-M): 		Width of the morphological kernel used for feature opening/closing during segmentation (Default: 5) 
		--distance-threshold (-D): 		Threshold applied to the distance transform of the image during watershed (Default: 6) 
		--size-threshold (-S): 			  For filtering of excessively small/large objects after segmentation. Fold-change relative to median size (Default: 3.0) 
		--border-fraction (-B): 			Width of image border to ignore for analysis, as percentage of image size (Default: 0.01) 
		--time-kernel (-T):			      Width of time kernel for temporal smoothing of segmentation masks (Default: 7 time points) 
		--spot-radius (-R):			      Radius of the volumetric kernel used to scan nuclear volumes and compute signal intensities (Default: 3 voxels) 
		--spot-depth (-d):			      Depth (in Z) of the volumetric kernel (Default: 4 voxels) 
		--green-background-start:			First time point to consider for computation of background distribution for GFP channel (Default: First) 
		--green-background-stop:			Last time point to consider for computation of background distribution for GFP channel (Default: Last) 
		--red-background-start:			  First time point to consider for computation of background distribution for RFP channel (Default: 25) 
		--red-background-stop:			  Last time point to consider for computation of background distribution for RFP channel (Default: Last) 
		--fdr: 					              Experimentally-defined False Discovery Rate for background correction 
							                    (Only used for the generation of a few diagnostic plots. Does NOT affect output data files) 
		--embryo-map:				          Time point for nuclei segmentation masks used to generate embryo heatmaps (Does NOT affect data files) 
 		--segmentation-movies:			  Output MP4 movies for visualization of nuclei segmentation over the time series (Default: False) 
		--spot-detection-movies:			Output MP4 movies for visualization of MS2/PP7 spots over the time series (Default: False) 
		--cores:					            Number of processors to use for transcription signal extraction step (Default: 8) 






ARGUMENTS: 

 		Input CZI 4D image file (3D image stack, time series, 3 channels expected including nuclear labeling for segmentation of individual nuclei) 





OUTPUT FILES: 

		Nuclei_Segmentation 			Directory containing tracked nuclei segmentation masks for all time points
		*_Background.txt				Background estimates for all nuclei at all time points (2 files: 1 per transcription measurement channel)
		*_Signal.txt				Transcription spot intensities for all nuclei at all time points (2 files: 1 per transcription measurement channel)
		*_Nucleus_X.txt/*_Nucleus_Y.txt		X-Y coordinates of tracked segmentation mask centroids throughout the time series
		*.png / *.mp4				Various graphical outputs (plots and movies) for data visualization and quality control. 




