//if an image state "Inverting LTU": white=0 black=255
input1="C:/Life/Thesis/eosin_test/";
input2="C:/Life/Thesis/background_test/";
list1 = getFileList(input1);//eosin from cellprofiler########
list2 = getFileList(input2);//background, tissue should be in dark#########
//file number should be the same

for (i = 0; i < list1.length; i++){
	open(input1 + list1[i]);
	open(input2 + list2[i]);
	selectWindow(list2[i]);//background
	imageCalculator("AND create", list2[i],list1[i]);
	run("Invert");
	selectWindow("Result of "+list2[i]);
	run("Invert LUT");
	}