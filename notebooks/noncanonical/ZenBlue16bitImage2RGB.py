#@ ImagePlus imp
#@ boolean (label = "Close the original", value = True) closeOriginal

# This is a Fiji/ImageJ helper script for pre-processing Zen Blue image files before SharpTrack analysis
# written by Kouichi C. Nakamura
# 
# 1. Analyse at least one image using Fiji and determine the proper DisplayRange for the images using "setMinAndMax.py"
# 2. Depending on the channels order, amend the "Prameters to adjust" section. We need LUTs and DisplayRange for each channel.
# 3. In the current version, the 4th channel (TL) is hidden, but this can be adapted for the situation.
# 4. The script will set new LUTs, set DisplayRanges, turn the image into RGB mode (8 bit).
# 5. Now you'll have to select individual section and use "Duplicate" of Fiji and save the image into a subfolder, typically namred "RGB"
# 6. You may want to make the file names easily sortable, eg. starting with 001_, 002_ etc.
# 7. SharpTrack can handle those RGB images.

######### >>>>>>> Prameters to adjust

disp_range = [[], [], [], []]
disp_range[0] = [0 , 20000] # max 65535
disp_range[1] = [0 , 20000]
disp_range[2] = [0 , 20000]
disp_range[3] = [0 , 65535]

LUTs = ["Cyan","Red","Green","Grays"]

######### <<<<<<<


import os
from ij import IJ, ImagePlus
from ij.gui import HistogramWindow, NonBlockingGenericDialog, DialogListener
import math
# import re
# import time
import inspect #TODO


assert len(disp_range) == len(LUTs)

def extract_repeat(s):
    parts = s.split(" - ")
    
    for part in parts:
        if s.count(part) > 1:
            return part
    return None


def doit():
	file_name = extract_repeat(imp.getTitle())

	# Change LUTs
	
	for i in range(0,len(LUTs)):
	
		# if imp.isHyperStack(): # not working
		imp.setC(i+1);
		IJ.run(imp, LUTs[i], "");
	
	
	# Hide TL channel (optional)
	imp.setDisplayMode(IJ.COMPOSITE);
	imp.setActiveChannels("1110");
	
	
	# set Display Range
	
	for i in range(0,len(LUTs)):
		imp.setDisplayRange(disp_range[i][0], disp_range[i][1])
	
		
	# turn into RGB
	IJ.run(imp, "RGB Color", "");
	
	imp2 = IJ.getImage(); # not working
	imp2.setTitle(file_name);
	
	if closeOriginal: # not working
		imp.close();
		imp.flush();
 
doit()
