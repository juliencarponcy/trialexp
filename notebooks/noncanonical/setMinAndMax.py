# This ImageJ script setMinAndMax.py is to be used to change the Min and Max 
# of each channel of a composite image in two ways.
#  1. You actually change the min and max values of the channel
#  2. While keeping the data unchanged, you change what range of pixel values
#  are shown (ie., Display Ranges. API equivalents are setDisplayRange() and 
#  setDisplayRanges())
#
# This is an alternative to ImageJ's builtin Brightness/Contrast > Set, which
# I found is unreliable.
# 
# Meant be better than buitin Brightness/Contrast, but actually
# it's essentially the same. I'm a bit dissapointed.
#
# KNOWN ISSUES
# When you type the Shadow or Highlight value in the dialog, sometimes
# the Display Range is not properly updated and the image may look ugly.
# If you click either the incremetal or decremental button on the slider,
# this broblem will be fixed. Or if you click OK, Display Range will be updated.
#
# You can check the current Display Ranges by Image > Show Info...
#
# HOW TO USE THIS?
# 1. FIji/ImageJ > File > Open and select this script.
# 2. Click the "Run" button at the bottom of ImageJ Macro Editor
#
# TODO
# * Support other data types including RGB
# 
# NOTE
# Use "IJ.log()" (Log window) or "print()" (window below) for debugging
#
# written by Kouichi C. Nakamura Ph.D.
# MRC Brain Network Dynamics Unit, University of Oxford
# kouichi.c.nakamura@gmail.com
# 18:44 on 6 Mar 2018

from ij import IJ, ImagePlus
from ij.gui import HistogramWindow, NonBlockingGenericDialog, DialogListener
import math
import re
import time
import inspect #TODO
# import numpy .... cannot be used
# GenericDialog does not work

global imp
global hw
global intmin
global intmax
global sld
global nf

intmin = 0
intmax = 0
imp = None
sld = None

def checkab(a,b,intmin,intmax):

	if a < intmin:
		IJ.showMessage("Shadow must be equal to or greater than " + str(intmin))
		imp.resetDisplayRange()
		return

	if b > intmax:
		IJ.showMessage("Highlight must be equal to or less than " + str(intmax))
		imp.resetDisplayRange()
		return

	if a > b:
		IJ.showMessage("Highlight must be equal to or greater than Shadow")
		imp.resetDisplayRange()
		return

# http://imagej.1557.x6.nabble.com/addSlider-question-td5010711.html
class MyDL(DialogListener):
    def dialogItemChanged(self, dialog, event):

		if type(event).__name__ == "ItemEvent":
			
			vec = dialog.getRadioButtonGroups() # vector

			if imp.isHyperStack():
				imp.setC(int(vec[0].getSelectedCheckbox().getLabel()));
				
			else:
				imp.setSlice(int(vec[0].getSelectedCheckbox().getLabel()));
				
			global hw
			hw.close();
			hw = HistogramWindow(imp);
			
			a = imp.getDisplayRangeMin();
			b = imp.getDisplayRangeMax();
			

			sld[0].setValue(int(a)); 
			sld[1].setValue(int(b));	

			nf[0].setText(str(int(a))); # Slider-associated NumericField
			nf[1].setText(str(int(b)));	
				
		
			return 1

		if 0: # type(event).__name__ == "TextEvent": # Disabled: Cannot update quick enough for large images

			
			# https://docs.oracle.com/javase/jp/8/docs/api/java/awt/Scrollbar.html
			if sld[0].getValueIsAdjusting(): # not effective
				return 1
			if sld[1].getValueIsAdjusting():
				return 1

			a = int(sld[0].getValue())
			b = int(sld[1].getValue())

			imp.setDisplayRange(a,b) # setMinAndMax cannot be used for CompositeImage
			imp.updateAndDraw() #TODO does not update properly

			return 1
			
		else:
			return 1





def runscript():

	global imp
	imp = IJ.getImage();

	if re.match(r"Histogram\sof\s",imp.getTitle()):
		IJ.showMessage("Histogram window was chosen")
		return


	bitdepth = imp.getBitDepth()

	if bitdepth == 24: #RGB
		bitdepth = 8

	global intmin
	global intmax
	intmin = int(0)
	intmax = int(math.pow(2,bitdepth) - 1)

	a = imp.getDisplayRangeMin()
	b = imp.getDisplayRangeMax()

	global hw
	hw = HistogramWindow(imp);

	items = ["Modify Min and Max values","Set Display Range (data unchanged)"]
	chanslistN = range(1, 1 + int(imp.getNChannels()));
	chanlist = [str(x) for x in chanslistN];
	
	gd = NonBlockingGenericDialog("Set Min and Max")

	gd.addRadioButtonGroup("Channel",chanlist,1,int(imp.getNChannels()),str(imp.getChannel())) #TODO
	# gd.addMessage("Channel " + str(imp.getChannel()))

	gd.addSlider("Shadow",intmin,intmax,a)
	gd.addSlider("Highlight",intmin,intmax,b)

	global sld
	global nf
	sld = gd.getSliders()
	nf = gd.getNumericFields()

	gd.addRadioButtonGroup("Choose",items,2,1,items[1])

	gd.addMessage("Use the arrow buttons to recover the image from mess")
	gd.addMessage("Ctrl + I for Info")
	gd.addMessage("Ctrl + Shift + Z for Color Tool")

	gd.addDialogListener(MyDL())

	vec = gd.getRadioButtonGroups() # vector


	gd.showDialog()


	if gd.wasCanceled():
		print("Canceled")
		hw.close();
		return # equivalent of IJ.exit in IJ1 macro


	selection = vec[1].getSelectedCheckbox().getLabel()

	# https://docs.oracle.com/javase/jp/8/docs/api/java/awt/CheckboxGroup.html#getSelectedCheckbox--
	# https://docs.oracle.com/javase/jp/8/docs/api/java/awt/Checkbox.html

	hw.close();


	sld = gd.getSliders()
	a = int(sld[0].getValue())
	b = int(sld[1].getValue())

	checkab(a,b,intmin,intmax)

	if selection == items[0]:
		# resample data with Min and Max

		ip = imp.getProcessor()

		val = ip.getPixels()

		ip.convertToFloatProcessor()
		ip.multiply(intmax/(b-a))
		ip.add(-a)
		if bitdepth == 16:
			ip.convertToShortProcessor()
		elif bitdepth == 12:
			#not sure
			ip.convertToShortProcessor()
		elif bitdepth == 8:
			ip.convertToByteProcessor()
		else:
			print("not supported yet")

		imp.setProcessor(ip)

		IJ.log("%s; Channel %d; PixelValues changed: Min %d set to %d, Max %d set to %d" % \
			(imp.getTitle(),imp.getChannel(),a,intmin,b,intmax))

	else:

		imp.setDisplayRange(a,b) # setMinAndMax cannot be used for CompositeImage
		imp.updateAndDraw()

		IJ.log("%s; Channel %d; DisplayRange: Min %d, Max %d" % \
			(imp.getTitle(),imp.getChannel(),a,b))

runscript()
