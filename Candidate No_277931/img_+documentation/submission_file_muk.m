% loading all png and solution .mat files
% Running findColours on each
% Checkin whether the results matches with the actual answer
% Calculating the overall score
% SCore



% finding all file png files
% 'images' path.
Dir=dir('images/*.png');

score = [];


%loading as well as now processin each file
for ind=1:length(Dir)

    % accessing file name of png file
    filename = fullfile(Dir(ind).folder,Dir(ind).name);

    % pathname of solution .mat file 

    [folder, baseFileName, ~] = fileparts(filename);
    mat_filenames = fullfile(folder, sprintf('%s.mat',baseFileName));


  
    % result testing 
  

    % calling the given findColours function
    try
        res = findColours(filename);
    catch ME
    % displaying error message
    disp(['Error occurred: ', ME.message]);
    end
    % checking answers
    mm = check_answer(res,mat_filenames);

    score=[score,mm];

end
    % score printing 
strn=repmat('%.2f ', 1, length(score));
fprintf('Score is: ');
fprintf(strn,score);
fprintf('\nMean score %f\n',mean(score));

function colordetect=findColours(filename)
    % loading images and returning as double type images
imagepa=loadImage(filename);
    % locatin and returning four black circles of coordinates
circleCoordinatesim=findCircles(imagepa);
    % un-distortion images


images = correctImage(circleCoordinatesim,imagepa);
    
    % getcolours taking double image array and using undistorted images

if contains(filename, 'noise_')|| contains(filename, 'org_')
    % getColours function returns array of the given image colours 
    result = getColours(imagepa);
    
elseif contains(filename, 'proj_')
   
    % Applying transformation to correct distortion
    x=correctImage(circleCoordinatesim,imagepa);
    imshow(x);
    figure;
    result=getColours(x);
elseif contains(filename,'rot_')
       
        y=correctImage(circleCoordinatesim,imagepa);
        imshow(y);
        figure;
        result=getColours(y);
else
    % filename format is not recognized
    disp('Wrong file format or distorted image');
    
end
disp(result)
colordetect = result;
end



function image=loadImage(filename)
image= imread(filename); %read img file
image=im2double(image);
   % changing image type to double
end

function circleCoordinates=findCircles(image)
    % converting color img to gray
grayimgs=rgb2gray(image); 
    % calculating threshold on gray imge
threshold=graythresh(grayimgs); 
    % creating threshold based binary img
bin_imgs= imbinarize(grayimgs,threshold);
    % inverting binary img and finding dark circles on light bkgrnd
inv_bin_img=imcomplement(bin_imgs); 
    % connected components identification in inverted binary img
    % area of each connected components calculation
connected_componentsim=bwconncomp(inv_bin_img);
areas=cellfun(@numel,connected_componentsim.PixelIdxList);
    % descending sort
[area_sort,indices_sort]=sort(areas,'descend');
    % coordinates of first 4 largest black blobs
num_blobsimg = 5;
blob_coordinats = zeros(num_blobsimg, 2);
for i = 2:num_blobsimg
    blob_indicesimg = connected_componentsim.PixelIdxList{indices_sort(i)};
    [rows, cols] = ind2sub(size(inv_bin_img), blob_indicesimg);
    blob_coordinats(i, :) = [ mean(cols),mean(rows)];
end
    % Removing first coordinate from blob_coordnts matrix
blob_coordinats(1, :) = [];


    % Sortin coordinates clockwise
    % starting from bottom-left
sortedCoordinates = sortrows(blob_coordinats);

if sortedCoordinates(2,2) < sortedCoordinates(1,2)
    % swap coords(If the 2ND coordinate is below the 1ST)
    sortedCoordinates([1 2],:) = sortedCoordinates([2 1],:);
end

if sortedCoordinates(4,2) > sortedCoordinates(3,2)
    % swap(if the 4TH coordinate is above 3rd)
sortedCoordinates([3 4],:) = sortedCoordinates([4 3],:);
end

circleCoordinates=sortedCoordinates;


end

function outputImage = correctImage(Coordinates, image)

    % fixed box with coordinates
boxf = [[0 ,0]; [0 ,480];[480 ,480]; [480 ,0]];

    % Calculating the transformation matrix based on the given Coordinates
    % transforming the matrix to fixed box(utilizing projective-transformation)
TFimg = fitgeotrans(Coordinates,boxf,'projective');

    % Creating an image referece object with the input image size
outviewimg = imref2d(size(image));

    % Applyin calculated transformation-matrix on input imag
    % creatin a new img with fill value 255(white) outside the boundaries of input imae
Bimg = imwarp(image,TFimg,'fillvalues',255,outputview=outviewimg);

    % Croping the image by [480x480]
Bimg = imcrop(Bimg,[0 0 480 480]);

    % suppress the glare in the imae[flat-field-correction]
Bimg = imflatfield(Bimg,40);

    % Adjusting image levels to improve contrast
Bimg = imadjust(Bimg,[0.4 0.65]);

    % Assigning corrected image to the outputImage variable
outputImage = Bimg;
end

    % array of colours from the image
function colours=getColours(image)

    % Converting the image to uint8 format
imgW=im2uint8(image);

    % suppressing noise[Median filter]
imgW = medfilt3(imgW,[7 7 1]);

    % Increasing contrast
imgW = imadjust(imgW,stretchlim(imgW,0.025));

    % Converting RGB to grayscale and threshold
Conimg = rgb2gray(imgW)>20;

    % Removing pos specks from binary img
Conimg = bwareaopen(Conimg,100);

    % Removing neg specks from binary img
Conimg = ~bwareaopen(~Conimg,100);

    % Removing outer white
Conimg = imclearborder(Conimg);

    % Erode image[shriking white bright region]
Conimg = imerode(Conimg,ones(10));

    % Segmenting image
[K O] = bwlabel(Conimg);

    % Storing average color of each region
Concolorsimg = zeros(O,3);

    % Getting average color in each labelled regin
for p = 1:O 
    % step[ through patches]
    each_pch = K==p;
    all_pch_areas = imgW(each_pch(:,:,[1 1 1]));
    Concolorsimg(p,:) = mean(reshape(all_pch_areas,[],3),1);
end

    % Normalizing color values to range [0, 1]
Concolorsimg = Concolorsimg./255;

    % Snapping centers to grid
Y = regionprops(Conimg,'centroid');
X = vertcat(Y.Centroid);
lim_X = [min(X,[],1); max(X,[],1)];
X = round((X-lim_X(1,:))./range(lim_X,1)*3 + 1);

    % Reordering color samples
idx = sub2ind([4 4],X(:,2),X(:,1));
Concolorsimg(idx,:) = Concolorsimg;

    % Specifing color names
clrnamespec = {'white','red','green','blue','yellow'};

    % declaring reference colors list in RGB
clrrefs = [1 1 1; 1 0 0; 0 1 0; 0 0 1; 1 1 0];

    % measuring colours distance in RGB
Icld = Concolorsimg - permute(clrrefs,[3 2 1]);
Icld = squeeze(sum(Icld.^2,2));

    % finding nearest match
[~,idx] = min(Icld,[],2);

    % Looking for colour names in patch
Colornames = reshape(clrnamespec(idx),4,4);

    % Returns array of color names
colours= Colornames;

end