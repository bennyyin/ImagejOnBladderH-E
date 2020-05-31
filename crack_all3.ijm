input = "C:/Life/Thesis/test/";
input="C:/Life/Thesis/cropped_pic/7000700NIN-1/";
//input="C:/Life/Thesis/cropped_pic/7000700IN-1/";
input="C:/Life/Thesis/cropped_pic/7000700IN-2/";
//input = "C:/Life/Thesis/input/";
output = "C:/Life/Thesis/output/";

setBatchMode(true);
run("Set Measurements...", "  redirect=None decimal=5");
list = getFileList(input);
//for (i = 0; i < 2; i++){
for (i = 0; i < list.length; i++){
	filename = list[i];
	
	crack(input, output, filename, i, "1");//first iteration for name
	crack(input, output, filename, i, "15");//the next iteration would need 
	
}
setBatchMode(false);
function crack(input, output, filename, rowNum, n){
	open(input + filename);
	selectWindow(filename);
	run("Bandpass Filter...", "filter_large=40 filter_small=3 suppress=None tolerance=5 autoscale saturate");
	runMacro("FeatureExtraction_findCracksC.ijm",n);
	//runMacro("FeatureExtraction_findCracks.ijm",n);
	selectWindow(filename);
	//run("Analyze Particles...", "size=15-Infinity pixel circularity=0.00-0.90 show=Masks display clear");
	//selectWindow(filename);
	//close();
	//selectWindow("Mask of "+filename);
	
	
			
	
	getHistogram(myvalues, mycount1, 2);//splitting the bin into 2  for 8-bit 255/'2' for 16 bit 48059/'2'
	setResult("crack_area_"+n, rowNum, mycount1[0]);
	setResult("non-crack_area_"+n, rowNum, mycount1[1]);
	totalarea= mycount1[0]+mycount1[1];
	setResult("total_area_"+n, rowNum, totalarea);
	crackratio = mycount1[0]/totalarea;
	setResult("crackratio_"+n, rowNum, crackratio);
	saveAs("tiff", output + filename+n+"_all_crack");			
	run("Find Edges");saveAs("tiff", output + filename+"_all_edges");	
	getHistogram(myvalues, mycount2, 2);//splitting the bin into 2  for 8-bit 255/'2' for 16 bit 48059/'2'
	setResult("crack_edge_"+n, rowNum, mycount2[1]);
	edgeratio = mycount2[1]/totalarea;
	setResult("edge_ratio_"+n, rowNum, edgeratio);
	updateResults();//show results table
//	saveAs("tiff", output + filename+"_all_crack");			
	//run("Measure");
	//close();
	//saveAs("Results", "C:/Users/benny/Desktop/Macro/Result_"+filename+".csv");
}