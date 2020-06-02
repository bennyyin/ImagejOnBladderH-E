//Cell cyto
input = "C:/Life/Thesis/test/";
//input = "C:/Life/Thesis/input/";
//input="C:/Life/Thesis/cropped_pic/7000700NIN-1/";
//input="C:/Life/Thesis/cropped_pic/7000700IN-1/";
output = "C:/Life/Thesis/cyto_test/";

setBatchMode(true);
list = getFileList(input);
//for (i = 0; i < 1; i++){
for (i = 0; i < list.length; i++){
	filename = list[i];
	cellcyto(input, output, filename,i);
}
setBatchMode(false);



function cellcyto(input, output, filename,rowNum){
	open(input + filename);
	run("Bandpass Filter...", "filter_large=40 filter_small=3 suppress=None tolerance=5 autoscale saturate");
	//run("Color Threshold...");
	// Color Thresholder 2.0.0-rc-65/1.51w
	// Autogenerated macro, single images only!
	min=newArray(3);
	max=newArray(3);
	filter=newArray(3);
	a=getTitle();
	run("HSB Stack");
	run("Convert Stack to Images");
	selectWindow("Hue");
	rename("0");
	selectWindow("Saturation");
	rename("1");
	selectWindow("Brightness");
	rename("2");
	min[0]=171; 
	max[0]=247;
	filter[0]="pass";
	min[1]=0;
	max[1]=255;
	filter[1]="pass";
	min[2]=161;
	max[2]=255;
	filter[2]="pass";
	for (i=0;i<3;i++){
	  selectWindow(""+i);
	  setThreshold(min[i], max[i]);
	  run("Convert to Mask");
	  if (filter[i]=="stop")  run("Invert");
	}
	imageCalculator("AND create", "0","1");
	imageCalculator("AND create", "Result of 0","2");
	for (i=0;i<3;i++){
	  selectWindow(""+i);
	  close();
	}
	selectWindow("Result of 0");
	close();
	selectWindow("Result of Result of 0");
	rename(a);
	// Colour Thresholding-------------

	setResult("name", rowNum, filename);
	getHistogram(myvalues, mycount1, 2);
	setResult("cyto_area", rowNum, mycount1[0]);//black pixel number
	updateResults();//show results table
	saveAs("tiff", output + filename+n+"_cyto");			
	
}