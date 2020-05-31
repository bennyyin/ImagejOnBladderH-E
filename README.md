# ImagejOnBladderH-E
376 invasive images vs 460 non-invasive imagess:
# crack 376vs460 old images: 'crack_imagej_features2.csv' 
#376vs469 old images: 'crack_eosin_nucleo_cellPro_nuclei_normalized_836.csv'

After adding 341 invasive images (717+460=1177)
717 invasive image and 460 non-invasive images:
#(341+376)vs460 new IN images (with ex_time with x125): 'crack_eosin_nucleo_cellPro_nuclei_normalized_1177.csv'
~first incorporation of features from CellProfiler
#717(341+376)vs460 new IN images (- ex_time with x125): 'crack_eosin_nucleo_cellPro_nuclei_normalized2_1177.csv'
#top 60 feautres
#717(341+376)vs460 new IN images (60 features: - ex_time -x125 ): 'crack_eosin_nucleo_cellPro_nuclei_normalized3_1177.csv'

~Crack pattern only:
#717(341+376)vs460 new IN images (cracks 15 features: - ex_time -x125 ): 'crack_eosin_nucleo_cellPro_nuclei_normalized4_1177.csv'
~pink cytoplasm pattern only:
#717(341+376)vs460 new IN images (pink cyto 13 features: - ex_time -x125 ): 'crack_eosin_nucleo_cellPro_nuclei_normalized5_1177.csv'
~cyotplasmic reaction pattern only:
#717(341+376)vs460 new IN images (cytoplasmic reaction 680 features: - ex_time -x125 ): 
‘crack_eosin_nucleo_cellPro_nuclei_normalized675_1177.csv’
~nuclei pattern only:
#717(341+376)vs460 new IN images nuclei features 1206 features (with nuclei features from cellPro): 'Nuclei_processed1210_1177.csv'

~All 3 patterns(crack+pink+cytoplasmic)
#(717+376)vs460 new IN images 696 features (with CNT features from cellPro): 'crack_eosin_nucleo_cellPro_nuclei_normalized700_1177.csv'

Note:
ex_time-x125: features that are about time to execute a calculation(None meaningful features)
-ex_time-x125: WITHOUT features that are about time to execute a calculation
We started the analysis with 376 invasive images vs 460 non-invasive images.
Later we added 341 invasive images into our pool:
Making 717 invasive images vs 460 non-invasive images
But when doing the training and testing, 460/717 of invasive images were randomly sampled. This made a balanced dataset: 460vs460
