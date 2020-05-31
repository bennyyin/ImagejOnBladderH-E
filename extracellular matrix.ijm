//extract cnt by masking crack cytosol and nuclous nucleous 


//8 bit to mask RBG

input = "C:/Life/Thesis/test/";
//input="C:/Life/Thesis/cropped_pic/7000700NIN-1/";
//input="C:/Life/Thesis/cropped_pic/7000700IN-1/";
//input = "C:/Life/Thesis/input/";
output = "C:/Life/Thesis/output/";


//setBatchMode(true);
list = getFileList(input);
//for (i = 0; i <1; i++){
//for (i = 0; i < list.length; i++){
for (i = 0; i < 1; i++){
	filename = list[i];
	open(input+filename);//run("Invert");
	selectWindow(filename);run("Duplicate...", "title=background");
	selectWindow(filename);run("Duplicate...", "title=crack");
	selectWindow(filename);run("Duplicate...", "title=cyto");
	
	selectWindow("cyto");//cyto
	runMacro("FeatureExtraction_findCyto.ijm");//"image_cyto"

	//mask
	run("Invert LUT");run("Invert");run("Duplicate...", "title=mask");
	
	selectWindow("background");//background
	runMacro("FeatureExtraction_findBackground.ijm");//run("Invert");
	run("Invert LUT");run("Invert");
	run("Duplicate...", "title=background_invert");run("Invert");

	
	selectWindow("crack");//crack
	runMacro("FeatureExtraction_findCracks.ijm", "1");run("Invert LUT");run("Invert");
	run("Duplicate...", "title=crack_invert");run("Invert");

	imageCalculator("Subtract", "crack", "background");//real crack
	
	//making mask
	//imageCalculator("Add", "mask", "background");
	imageCalculator("Subtract", "mask", "crack");
	selectWindow("mask");run("Invert");
	
	selectWindow(filename);
	runMacro("FeatureExtraction_findEosin.ijm",filename);//run("Invert");
	imageCalculator("Subtract create", "eosin", "mask");		
	//imageCalculator("Subtract create", filename, "cyto");
	
	

	}


	
setBatchMode(false);