//input = "C:/Life/Thesis/test/";

input="C:/Life/Thesis/cropped_pic/7000700NIN-1/";
//input="C:/Life/Thesis/cropped_pic/7000700IN-1/";
output = "C:/Life/Thesis/output/";

setBatchMode(true);
list = getFileList(input);
//for (i = 0; i <3 ; i++){
for (i = 0; i < list.length; i++){
	filename = list[i];
	redcyto(input, output, filename,i);
	
}
setBatchMode(false);



function redcyto(input, output, filename, rowNum){
	open(input + filename);
	run("Bandpass Filter...", "filter_large=40 filter_small=3 suppress=None tolerance=5 autoscale saturate");
	run("Colour Deconvolution", "vectors=H&E hide");

	
	hemo=filename+"-(Colour_1)";
	selectWindow(filename+"-(Colour_1)");//this is nucleaus 
	run("Duplicate...", "title=_HE1_Otsu");
	selectWindow(filename+"-(Colour_1)");//this is nucleaus 
	run("Duplicate...", "title=_HE1_Huang");
	close(filename+"-(Colour_3)");//this is green
	close(filename+"-(Colour_1)");//close original

	//Otsu threshold//
	selectWindow("_HE1_Otsu");	run("8-bit");
	run("Auto Threshold", "method=Otsu ignore_white white");
	run("Remove Outliers...", "radius=3 threshold=50 which=Dark");
	run("Watershed");
	//run("Set Measurements...", "area center bounding shape integrated redirect=_HE1_Otsu decimal=4");
	//run("Analyze Particles...", "  show=[Overlay Masks] display exclude include summarize");//this shows up summary
	//IJ.renameResults("Results","x1"); 
	//IJ.renameResults("Summary","y1");close("Summary");
	//selectWindow("Results");
	saveAs("tiff", output+filename+"_HE1_Otsu"); 
	//saveAs("Results", output+filename+"_HE1_Otsu.csv"); 
	//run("Clear Results"); 
	
	//Huang threshold//
	selectWindow("_HE1_Huang");run("8-bit");
	run("Auto Threshold", "method=Huang ignore_white white");	
	run("Watershed");
	run("Remove Outliers...", "radius=3 threshold=50 which=Dark");
	//run("Set Measurements...", "area center bounding shape integrated redirect=_HE1_Huang decimal=4");
	//run("Analyze Particles...", "  show=[Overlay Masks] display exclude include summarize");//this shows up summary
	//IJ.renameResults("Results","x2"); 
	//IJ.renameResults("Summary","y2");close("Summary");
	//selectWindow("Results");
	saveAs("tiff", output+filename+"_HE1_Huang"); 
	//saveAs("Results", output+filename+"_HE1_Huang.csv");
	//run("Clear Results"); 
	


	}
function histogramToResult(prefix,nBins){
	//run("Clear Results");
	getHistogram(values, counts, nBins);
	for (row=0; row<nBins; row++) {
        setResult(prefix+"Value", row, values[row]);
        setResult(prefix+"Count", row, counts[row]);
        }
  	updateResults();
  	}