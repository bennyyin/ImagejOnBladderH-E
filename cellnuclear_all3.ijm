input = "C:/Life/Thesis/test/";

//input="C:/Life/Thesis/cropped_pic/7000700NIN-1/";
//input="C:/Life/Thesis/cropped_pic/7000700IN-1/";
input="C:/Life/Thesis/cropped_pic/7000700IN-2/";
output = "C:/Life/Thesis/output/";

setBatchMode(true);
list = getFileList(input);
//run("Set Scale...", "distance=0 known=0 pixel=1 unit=pixel");
//for (i = 0; i <2 ; i++){
for (i = 0; i < list.length; i++){
	filename = list[i];
	//setResult("name", i, filename);
	redcyto(input, output, filename,i);
	
}
setBatchMode(false);

function toResults2(winName, filename, rowNum){//summary with multiple rows into one row in results window
	setResult("name", rowNum, filename);
	selectWindow(winName); 
	//getInfo() returns that whole spreadsheet as one giant string with \n
	lines = split(getInfo(), "\n"); //each line as one string that separated by /t
	features = split(lines[0], "\t");
	for (i=1; i< lines.length;i++){
		values = split(lines[i], "\t"); 
		postfix = values[0];
		for(j=1; j<values.length;j++){
			setResult(features[j]+postfix, rowNum, values[j]);
			}
		}
	//updateResults();//show results table
	}
	


function redcyto(input, output, filename, rowNum){
	
	open(input + filename);
	run("Enhance Contrast...", "saturated=0.3");
	run("Colour Deconvolution", "vectors=H&E hide");

	//hemo=filename+"-(Colour_1)";
	selectWindow(filename+"-(Colour_1)");//this is nucleaus 
	run("Duplicate...", "title=_HE1_Otsu");
	selectWindow(filename+"-(Colour_1)");//this is nucleaus 
	run("Duplicate...", "title=_HE1_Huang");
	
	close(filename+"-(Colour_3)");//this is green
	close(filename+"-(Colour_1)");//close original

	//Otsu threshold//
	selectWindow("_HE1_Otsu");
	run("Auto Threshold", "method=Otsu ignore_white white");run("Duplicate...", "title=_HE1_Otsu_raw");selectWindow("_HE1_Otsu");
	run("Remove Outliers...", "radius=3 threshold=50 which=Dark");
	run("Watershed");
	run("Set Measurements...", "area center bounding shape integrated redirect=_HE1_Otsu decimal=4");	
	run("Analyze Particles...", "  show=[Overlay Masks] display exclude include summarize");//this shows up summary
	//IJ.renameResults("Results","x1"); 
	IJ.renameResults("Summary","OtsuSummary");close("Summary");
	selectWindow("Results");
	//saveAs("Results", output+filename+"_HE1_Otsu.csv"); 
	run("Clear Results"); //The results records every nucleous as row and property as col(a spreadsheet per images)
	
	//Huang threshold//
	selectWindow("_HE1_Huang");
	run("Auto Threshold", "method=Huang ignore_white white");run("Duplicate...", "title=_HE1_Huang_raw");selectWindow("_HE1_Huang");
	
	run("Remove Outliers...", "radius=3 threshold=50 which=Dark");
	run("Watershed");
	run("Set Measurements...", "area center bounding shape integrated redirect=_HE1_Huang decimal=4");
	run("Analyze Particles...", "  show=[Overlay Masks] display exclude include summarize");//this shows up summary
	//IJ.renameResults("Results","x2"); 
	IJ.renameResults("Summary","HuangSummary");close("Summary");
	selectWindow("Results");
	//saveAs("Results", output+filename+"_HE1_Huang.csv");
	run("Clear Results");//The results records every nucleous as row and property as col(a spreadsheet per images)

	//Otsu and Huang threshold//
	imageCalculator("AND create", "_HE1_Otsu_raw","_HE1_Huang_raw");selectWindow("Result of _HE1_Otsu_raw");rename("Otsu_and_Huang");
	run("Remove Outliers...", "radius=3 threshold=50 which=Dark");
	run("Watershed");
	run("Set Measurements...", "area center bounding shape integrated redirect=_HE1_Huang decimal=4");
	run("Analyze Particles...", "  show=[Overlay Masks] display exclude include summarize");//this shows up summary
	//IJ.renameResults("Results","x2"); 
	IJ.renameResults("Summary","Otsu_HuangSummary");close("Summary");
	selectWindow("Results");
	//saveAs("Results", output+filename+"_HE1_Otsu_Huang.csv");
	run("Clear Results");//The results records every nucleous as row and property as col(a spreadsheet per images)	
	
	//toResults2(winName,filename,rowNum);updateResults();
	toResults2("OtsuSummary",filename,0);//close("Summary");updateResults();//Otsu result
	close("OtsuSummary");
	toResults2("HuangSummary",filename,0);//close("Summary");updateResults();//Huang result
	close("HuangSummary");
	toResults2("Otsu_HuangSummary",filename,0);//close("Summary");updateResults();//Huang result
	close("Otsu_HuangSummary");
	
	IJ.renameResults("Results","Nuclear_Results"); 
	
	selectWindow("_HE1_Huang");run("8-bit");
	saveAs("tiff", output + filename+"_HE1_Huang");
	selectWindow("_HE1_Otsu");run("8-bit");
	saveAs("tiff", output + filename+"_HE1_Otsu");
	selectWindow("Otsu_and_Huang");run("8-bit");
	saveAs("tiff", output + filename+"_Otsu_and_Huang");close();
	//close("_HE1_Otsu");close("_HE1_Huang");//image after counting 


	//red and blue signal//////
	selectWindow(filename+"-(Colour_2)");//this is overall red 
	run("Clear Results");
	histogramToResult("HE1_E_",256);//IJ.renameResults("x"); 
	
	selectWindow(filename);
	run("Colour Deconvolution", "vectors=[H&E 2] hide");
	selectWindow(filename+"-(Colour_2)");//this is overall red but excluding nucleas
	histogramToResult("HE2_E_",256);//IJ.renameResults("y"); 
	close(filename+"-(Colour_2)");
	
	selectWindow(filename+"-(Colour_1)");//this is nucleaus 
	histogramToResult("HE2_H_",256);//IJ.renameResults("z"); 
	close(filename+"-(Colour_1)");//this is nucleaus 
	close(filename+"-(Colour_3)");//this is green
	close(filename);
	//IJ.renameResults("Results","Eosin_histogram"); 
	selectWindow("Results");
	//saveAs("Results", output+filename+"Eosin_histogram.csv");
	run("Clear Results");
	if(rowNum==0){
		toResults2("Nuclear_Results",filename,rowNum);
		IJ.renameResults("Results","Final_Results");
		close("Nuclear_Results");
		}
	else{
		IJ.renameResults("Final_Results","Results"); 
		toResults2("Nuclear_Results",filename,rowNum);
		IJ.renameResults("Results","Final_Results");
		close("Nuclear_Results");
		}
	while (nImages>0) { 
          selectImage(nImages); 
          close(); 
      } 
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