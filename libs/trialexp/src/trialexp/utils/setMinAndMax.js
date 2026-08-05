# @ ImagePlus (label="Chhoose an image") imp

/*
This ImageJ script setMinAndMax.js is a clone of setMinAndMax.py, and is to be
used to change the Min and Max of each channel of a composite image in two ways.

 1. You actually change the min and max values of the channel
 2. While keeping the data unchanged, you change what range of pixel values

 are shown (ie., Display Ranges. API equivalents are setDisplayRange() and
 setDisplayRanges())

This is an alternative to ImageJ's builtin "Brightness/Contrast > Set", which
I found is unreliable.

Meant to be better than buitin Brightness/Contrast, but actually
it's essentially the same. I'm a bit dissapointed.

KNOWN ISSUES
When you type the Shadow or Highlight value in the dialog, sometimes
the Display Range is not properly updated and the image may look ugly.
If you click either the incremetal or decremental button on the slider,
this broblem will be fixed. Or if you click OK, Display Range will be updated.

You can check the current Display Ranges by Image > Show Info...

HOW TO USE THIS?
1. FIji/ImageJ > File > Open and select this script.
2. Click the "Run" button at the bottom of ImageJ Macro Editor

TODO
* Support other data types including RGB

TODO
* Show the current Display Range values for all the channels

TODO
* Add pull down menu for LUTs


NOTE
Use "IJ.log()" (Log window) or "print()" (window below) for debugging

written by Kouichi C. Nakamura Ph.D.
MRC Brain Network Dynamics Unit, University of Oxford
kouichi.c.nakamura@gmail.com
18:44 on 6 Mar 2018
*/

importClass(Packages.ij.IJ);
importClass(Packages.ij.ImagePlus);
importClass(Packages.ij.gui.HistogramWindow);
importClass(Packages.ij.gui.NonBlockingGenericDialog);
importClass(Packages.ij.gui.DialogListener);

importClass(Packages.java.awt.event.ItemEvent);
importClass(Packages.java.awt.AWTEvent);

/*
import math
import re
import time
import inspect #TODO
# import numpy .... cannot be used
# GenericDialog does not work
*/

var intmin = 0;
var intmax = 0;
var sld = null;

var imp;
var hw;
var intmin;
var intmax;
var sld;
var nf;

function checkab(a,b,intmin,intmax) {

	if (a < intmin) {
		IJ.showMessage("Shadow must be equal to or greater than " + intmin.toString());
		imp.resetDisplayRange();
		return
	}

	if (b > intmax) {
		IJ.showMessage("Highlight must be equal to or less than " + intmax.toString());
		imp.resetDisplayRange();
		return
	}

	if (a > b) {
		IJ.showMessage("Highlight must be equal to or greater than Shadow");
		imp.resetDisplayRange();
		return
	}
}

// http://imagej.1557.x6.nabble.com/addSlider-question-td5010711.html

// var DL = Java.type(DialogListener);
// var myDL = Java.extend(DL,{

var l = 1;


function runscript() {

	imp = IJ.getImage();

	var re = /Histogram\sof\s/;
	if (re.test(imp.getTitle())) { //TODO
		IJ.showMessage("Histogram window was chosen");
		return
	}

	var bitdepth = imp.getBitDepth();

	if (bitdepth == 24) { //RGB
		bitdepth = 8;
	}

	intmin = 0;
	intmax = Math.round(Math.pow(2,bitdepth) - 1);

	var a = imp.getDisplayRangeMin();
	var b = imp.getDisplayRangeMax();

	hw = new HistogramWindow(imp);

	var items = ["Modify Min and Max values","Set Display Range (data unchanged)"];
	//var chanslistN = range(1, 1 + int(imp.getNChannels()));　//TODO
	var chanslistN = new Array();
	for (i = 1; i < 1 + imp.getNChannels(); i++ ){
		chanslistN[i-1] = i;
	}

	//var chanlist = [str(x) for x in chanslistN]; //TODO
	var chanlist = new Array();
	for (i = 0; i < chanslistN.length; i++) {
		chanlist[i] = chanslistN[i].toString();
	}

	var gd = new NonBlockingGenericDialog("Set Min and Max");

	gd.addRadioButtonGroup("Channel (NOT WORKING)",chanlist,1,imp.getNChannels(), imp.getChannel().toString()); //TODO
	// gd.addMessage("Channel " + str(imp.getChannel()));

	var currentChan = imp.getChannel().toString();

	gd.addSlider("Shadow",intmin,intmax,a);
	gd.addSlider("Highlight",intmin,intmax,b);

	sld = gd.getSliders();
	nf = gd.getNumericFields();

	gd.addRadioButtonGroup("Choose",items,2,1,items[1]);

	gd.addMessage("Use the arrow buttons to recover the image from mess");
	gd.addMessage("Ctrl + I for Info");
	gd.addMessage("Ctrl + Shift + Z for Color Tool");

    var LUTs = ["No change", 'Grays','Red', 'Green', 'Blue', 'Magenta', 'Cyan', 'Yellow'];
	gd.addChoice("Change LUT:", LUTs, "Nochange");
	cho = gd.getChoices();


	// https://imagej.net/JavaScript_Scripting#Interfaces_and_anonymous_classes ***
	// https://imagej.net/Scripting_comparisons#In_Javascript ***
	// https://www.codota.com/code/java/methods/ij.gui.GenericDialog/addDialogListener ***
	// https://www.codota.com/code/java/methods/ij.gui.DialogListener/dialogItemChanged ***

	gd.addDialogListener(new DialogListener({
		dialogItemChanged: function(dialog, event) {

			if (dialog !== null) {
				var vec = dialog.getRadioButtonGroups(); // vec[0] is the radio button for channels

				if (vec[0].getSelectedCheckbox().getLabel() == currentChan) {
					// print("No change");
					return 1
				} else {
					// print("Changed");
					currentChan = vec[0].getSelectedCheckbox().getLabel();
					if (imp.isHyperStack()) {
						imp.setC(Math.round(currentChan));
					} else {
						imp.setSlice(Math.round(currentChan));
					}

					hw.close();
					hw = new HistogramWindow(imp);

					var a = imp.getDisplayRangeMin();
					var b = imp.getDisplayRangeMax();


					sld[0].setValue(Math.round(a));
					sld[1].setValue(Math.round(b));

					nf[0].setText(Math.round(a).toString()); // Slider-associated NumericField
					nf[1].setText(Math.round(b).toString());

					return 1

				}
				

			} else {
				return 1
			}
		}
	}));


	var vec = gd.getRadioButtonGroups(); // vector


	gd.showDialog();


	if (gd.wasCanceled()) {
		print("Canceled")
		hw.close();
		return // equivalent of IJ.exit in IJ1 macro
	}

	selection = vec[1].getSelectedCheckbox().getLabel();

	// https://docs.oracle.com/javase/jp/8/docs/api/java/awt/CheckboxGroup.html#getSelectedCheckbox--
	// https://docs.oracle.com/javase/jp/8/docs/api/java/awt/Checkbox.html

	hw.close();


	sld = gd.getSliders();
	a = Math.round(sld[0].getValue());
	b = Math.round(sld[1].getValue());

	checkab(a,b,intmin,intmax);

	if (selection == items[0]) {
		// resample data with Min and Max

		var ip = imp.getProcessor();

		var val = ip.getPixels();

		ip.convertToFloatProcessor();
		ip.multiply(intmax/(b-a));
		ip.add(-a);
		if (bitdepth == 16) {
			ip.convertToShortProcessor();
		}
		else if (bitdepth == 12) {
			// not sure
			ip.convertToShortProcessor();
		}
		else if ( bitdepth == 8) {
			ip.convertToByteProcessor();
		} else {
			print("not supported yet");
		}

		imp.setProcessor(ip);

		IJ.log("%s; Channel %d; PixelValues changed: Min %d set to %d, Max %d set to %d",
		imp.getTitle(),imp.getChannel(),a,intmin,b,intmax);
		}

	else {

		imp.setDisplayRange(a,b); // setMinAndMax cannot be used for CompositeImage
		imp.updateAndDraw();

		print(imp.getTitle() + "; Channel " + imp.getChannel() + "; DisplayRange: Min " +
			a + ", Max " + b);
	}

	
	switch (cho[0].getSelectedIndex()) {
		case 0: // No Change
			break;
		case 1: // Grays
			IJ.run(imp, "Grays", "");
			print('LUT:Grays');
			break;
		case 2:
			IJ.run(imp, "Red", "");
			print('LUT:Red');
			break;
		case 3:
			IJ.run(imp, "Green", "");
			print('LUT:Green');
			break;
		case 4:
			IJ.run(imp, "Blue", "");
			print('LUT:Blue');
			break;
		case 5:
			IJ.run(imp, "Magenta", "");
			print('LUT:Magenta');
			break;
		case 6:
			IJ.run(imp, "Cyan", "");
			print('LUT:Cyan');
			break;
		case 7:	
			IJ.run(imp, "Yellow", "");
			print('LUT:Yellow');
			break;
		default:
			break;
	}
	
}

runscript();
