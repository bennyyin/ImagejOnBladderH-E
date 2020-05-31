

getSelectionBounds(xbase, ybase, width, height); 
labels = newArray(width+1);
labels[0] = "Y";
for (i=0; i<width; i++)
   labels[i+1] = toString(xbase+i);
rgb = bitDepth == 24;
run("Clear Results");
for (row=0; row<height; row++) {
   setResult(labels[0], row, ybase+row);
   for (i=0; i<width; i++) {
         v = getPixel(xbase+i, ybase+row);
         if (rgb){
	       	red = (v>>16)&0xff;  // extract red byte (bits 23-17)
    	   	green = (v>>8)&0xff; // extract green byte (bits 15-8)
       		blue = v&0xff;       // extract blue byte (bits 7-0)
          	output = toString(red)+","+toString(green)+","+toString(blue);         	
          	setResult(labels[i+1], row, output);    	
          	}                      
         }
}