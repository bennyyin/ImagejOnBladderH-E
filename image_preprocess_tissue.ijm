//input = "C:/Life/Thesis/test/";
//input = "C:/Life/Thesis/test1/";
//input="C:/Life/Thesis/cropped_pic/7000700NIN-1/";
//input="C:/Life/Thesis/cropped_pic/7000700IN-1/";
input="C:/Life/Thesis/cropped_pic/7000700IN-2/";
//input = "C:/Life/Thesis/input/";
//output = "C:/Life/Thesis/test_processed/";
//output = "C:/Life/Thesis/cropped_pic/7000700IN-1_processed/";
output = "C:/Life/Thesis/cropped_pic/7000700IN-2_processed/";
//output = "C:/Life/Thesis/cropped_pic/7000700NIN-1_processed/";
//output = "C:/Life/Thesis/test1/";

setBatchMode(true);//so the windows do show up
//run("Set Measurements...", "  redirect=None decimal=3");
list = getFileList(input);
//for (i = 0; i < 1; i++){
for (i = 0; i < list.length; i++){
	filename = list[i];//list of filename (string)
	//print(i+","+filename);
	open(input + filename);
	
	run("Duplicate...", "title=tissue_layer");selectWindow("tissue_layer");
	runMacro("FeatureExtraction_findTissue.ijm",i);run("Despeckle");run("Remove Outliers...", "radius=2 threshold=50 which=Dark");run("Erode");run("Invert");
	selectWindow(filename);
	//run("Enhance Contrast...", "saturated=0.3");
	//run("Bandpass Filter...", "filter_large=40 filter_small=3 suppress=None tolerance=5 autoscale saturate");
	//imageCalculator("AND create", filename,"tissue_layer");
	imageCalculator("Add create", filename,"tissue_layer");
	//saveAs("tiff", output + filename+"test1");	while (nImages>0) {selectImage(nImages);close();}
	saveAs("tiff", output + filename);	while (nImages>0) {selectImage(nImages);close();}
	
	
}
setBatchMode(false);
