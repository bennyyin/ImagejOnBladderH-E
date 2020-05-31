input = "C:/Life/Thesis/test/";
input="C:/Life/Thesis/cropped_pic/7000700NIN-1/";
input="C:/Life/Thesis/cropped_pic/7000700IN-1/";
//input = "C:/Life/Thesis/input/";
output = "C:/Life/Thesis/output/";

setBatchMode(true);
run("Set Measurements...", "  redirect=None decimal=5");
list = getFileList(input);
//for (i = 0; i < 1; i++){
for (i = 0; i < list.length; i++){
	filename = list[i];
	//action(input, output, filename, i, "1");
	action(input, output, filename, i);
}
setBatchMode(false);



function action(input, output, filename, rowNum){
	open(input + filename);
	selectWindow(filename);
	setResult("name", rowNum, filename);
	run("Bandpass Filter...", "filter_large=40 filter_small=3 suppress=None tolerance=5 autoscale saturate");
	
	run("Duplicate...", "title=rawCNT");
	selectWindow("rawCNT");
	run("Duplicate...", "title=cyto");
	selectWindow("cyto");//cyto
	runMacro("FeatureExtraction_findCyto.ijm");//"image_cyto"
	run("Despeckle");run("Despeckle");
	
	selectWindow("rawCNT");//rawCNT
	runMacro("FeatureExtraction_findRedCNT.ijm");//"image_cyto"
	imageCalculator("Subtract create", "rawCNT","cyto");
	//run("Remove Outliers...", "radius=4 threshold=15 which=Bright");
	//imageCalculator("Subtract create", filename,"Result of rawCNT");
	imageCalculator("Subtract create", filename,"rawCNT");saveAs("tiff", output + filename+"_rawCNT");//image name changes after saving
	getHistogram(myvalues, mycount, 2);//splitting the bins into 2
	setResult("redCNT_area", rowNum, mycount[0]);
	
	
	imageCalculator("Subtract create", filename+"_rawCNT.tif","cyto");saveAs("tiff", output + filename+"_redCNT-cyto");
	close("rawCNT");close("cyto");
	getHistogram(myvalues, mycount, 2);//splitting the bins into 2
	setResult("redCNT-cyto_area", rowNum, mycount[0]);
}