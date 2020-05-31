//4bins
bins = 128;
getHistogram(values, counts, bins);
n = 0;
sum = 0;
min = 9999999;
max = -9999999;
Region2=0;
Region3=0;
Region4=0;
Region1=0;
Region0=0;
TotalPixel=0;
PercentRegion1=0;
PercentRegion2=0;
PercentRegion4=0;
PercentRegion3=0;
PercentRegion0=0;
Score=0;


for (i=0; i<85; i++){
	if (i<2)
		Region0=Region0+counts[i];//pixel that do not need to consider
	else if (i>=2 && i<19)
		Region1=Region1+counts[i];
	else if (i>18 && i<37)	
		Region2=Region2+counts[i];
	else if (i>36 && i<55)	
		Region3=Region3+counts[i];
	else if (i>54 && i<85)	
		Region4=Region4+counts[i];
	else
		print(i);
}
TotalPixel=TotalPixel+Region1+Region2+Region3+Region4+Region0;
  
PixelUnderConsideration=TotalPixel-Region0;

PercentRegion4=(Region4/PixelUnderConsideration)*100;//highest
PercentRegion3=(Region3/PixelUnderConsideration)*100; 
PercentRegion2=(Region2/PixelUnderConsideration)*100;
PercentRegion1=(Region1/PixelUnderConsideration)*100;//lowest

print("Percentage contibution of High Positive:  "+PercentRegion4);print("Percentage contibution of Positive:  "+PercentRegion3);print("Percentage contibution of Low Positive:  "+PercentRegion2);print("Percentage contibution of Negative:  "+PercentRegion1);

Score=Score+(PercentRegion4/100)*4+(PercentRegion3/100)*3+(PercentRegion2/100)*2+(PercentRegion1/100)*1;  
      
if(Score>=2.95){
	print("The score is High Positive  ");
}
if((Score>=1.95) && (Score<=2.94)){
	print("The score is Positive  ");
}
if((Score>=0.95) && (Score<=1.94)){
	print("The score is Low Positive  ");
}
if((Score>=0.0) && (Score<=0.94)){
	print("The score is Negative  ");
}

