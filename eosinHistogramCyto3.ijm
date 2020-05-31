//excluding none-tissue area and extract the

//input = "C:/Life/Thesis/test/";
input="C:/Life/Thesis/cropped_pic/7000700NIN-1/";
//input="C:/Life/Thesis/cropped_pic/7000700IN-1/";
//input = "C:/Life/Thesis/input/";
output = "C:/Life/Thesis/output/";


setBatchMode(true);
list = getFileList(input);
//for (i = 0; i <3; i++){
for (i = 0; i < list.length; i++){
	filename = list[i];
	open(input+filename);//run("Invert");
	a=getTitle();
	run("Colour Deconvolution", "vectors=[H&E] hide");
	//run("Colour Deconvolution", "vectors=[H&E 2] hide");
	selectWindow(a+"-(Colour_2)");run("Invert");close("\\Others");
	
	cellyto = cellcyto(input, output, filename, i);run("Invert");
	imageCalculator("Subtract create",  a+"-(Colour_2)",cellyto);
	//imageCalculator("Subtract create", "Result of 1.tif-(Colour_2)","1.tif");	
	//noRBC = findRBC(input, output, filename);
	//imageCalculator("Subtract create", a+"-(Colour_2)",noRBC);close("\\Others");
	
	getHistogram(values, red_counts, 128);
	//Array.show("red_count",red_counts);
	setResult("name", i, filename);
	runMacro("FeatureExtraction_4bins.ijm", i);
	getDimensions(width, height,channels, slices, frames);
	non_cyto_region=0;
	for(j = 0; j <1 ; j++){non_cyto_region+=red_counts[j];}
	sum2=width*height-non_cyto_region;//total - subtract the black pixels (non tissue)
	setResult("non_Cyto_pixels", i, sum2);
	//print(sum2/(width*height)*100);
	setResult("intensity_mean", i, findMean(red_counts));
	setResult("intensity_median", i, findMedian(red_counts));
	setResult("intensity_mode", i, findMode(red_counts));
	if(findMode(red_counts)<50){open(input+filename);saveAs("Jpeg", output+filename);}
	for(j = 0; j <128 ; j++){setResult(j, i, red_counts[j]);}
	updateResults();//show results table
	}
setBatchMode(false);


function findRBC(input, output, filename){
	open(input + filename);
	min=newArray(3);
	max=newArray(3);
	filter=newArray(3);
	a=getTitle();
	run("RGB Stack");
	run("Convert Stack to Images");
	selectWindow("Red");
	rename("0");
	selectWindow("Green");
	rename("1");
	selectWindow("Blue");
	rename("2");
	min[0]=133;
	max[0]=255;
	filter[0]="pass";
	min[1]=0;
	max[1]=96;
	filter[1]="pass";
	min[2]=0;
	max[2]=117;
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
	noRBC = filename+"_noRBC";
	rename(noRBC);
	return noRBC;
	// Colour Thresholding-------------
	}

function cellcyto(input, output, filename, rowNum){
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
	cyto = filename+"_cyto";
	rename(cyto);
	setResult("name", rowNum, filename);
	getHistogram(myvalues, mycount1, 2);
	setResult("cyto_area", rowNum, mycount1[0]);//black pixel number
	updateResults();//show results table
	return cyto;
	// Colour Thresholding-------------
}
function cutBins(array, binNumber){}

function findMode(array){
	max=-1;
	max_index=-1;
	for (i = 2 ; i<128; i++){//excluding first 2 intensity
		intensity=i;
		numberofInten=array[i];
		if(numberofInten>max){
			max_index=intensity;
			max=numberofInten;
			}
		}
	return max_index;
}

function findMean(array){
	totalpixn=0;
  	totalinten=0;
	for (i = 2 ; i<128; i++){
		intensity=i;
    	numberofInten=array[i];
    	totalpixn += numberofInten;
    	totalinten += (intensity*numberofInten);
	}
	return (totalinten/totalpixn);
}
	
function findMedian(array){
	intenSum=0;
	for (i = 2 ; i<128; i++){
		intenSum += array[i];
	}
	intenMiddle =intenSum/2;
	for (j =2 ; j<128 ; j++){
		intenMiddle -= array[j];
    	if(intenMiddle<0){
      		return (j);
    	}
	}
}