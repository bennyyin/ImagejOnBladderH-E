
function action(input, output, filename) {
	open(input + filename);
	getDimensions(width, height, channels, slices, frames);
	num=1;
	for (x =560; x+700 <height; x+=500){
		for (y = 400; y+700 < width; y+=500){
			open(input + filename);
			makeRectangle(x, y, 700, 700);run("Crop");
			saveAs("tiff", output + filename+"_"+num);
			close();
			num++;
		}
	}close();
}

input = "C:/Life/Thesis/macros/pre_image/";
output = "C:/Life/Thesis/macros/crop_images/";
setBatchMode(true);
list = getFileList(input);
//for (i = 0; i < list.length; i++){
for (i = 0; i < 3; i++){
        action(input, output, list[i]);
        }
setBatchMode(false);
