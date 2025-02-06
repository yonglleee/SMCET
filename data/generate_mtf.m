%% MTF filters the image I_MS using a Gaussin filter matched with the Modulation Transfer Function (MTF) of the MultiSpectral (MS) sensor. 
clear; close all;
addpath matlab_tool;

scale = 4;

for i = 1 :190
	path_ms = strcat('../dataset/train/ms/', num2str(i), '.tif');
	ms = imread(path_ms);
    image_ms = double(ms);
    image_ms = MTF(image_ms,'WV2',scale);
    if exist('../dataset/train/mtf/') == 0
    	mkdir('../dataset/train/mtf/');
    end
    str1 = strcat('../dataset/train/mtf/', num2str(i), '.tif');
    imwrite(uint8(image_ms), str1, 'tif');
end

for i = 191 : 210
	path_ms = strcat('../dataset/test/ms/', num2str(i), '.tif');
	ms = imread(path_ms);
    image_ms = double(ms);
    image_ms = MTF(image_ms,'WV2',scale);
    if exist('../dataset/test/mtf/') == 0
    	mkdir('../dataset/test/mtf/');
    end
    str1 = strcat('../dataset/test/mtf/', num2str(i), '.tif');
    imwrite(uint8(image_ms), str1, 'tif');
end