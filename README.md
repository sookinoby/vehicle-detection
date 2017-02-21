#Vehicle Detection Project

The goals/steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


##1. Generated additional data
We felt the lack of black car images may affect the performance of the classifier. So we used additional black-car images data. We scaled, translated, flipped the images to avoid overfitting the data. 200 such images were generated and added to the vehicles folders
The code can be found in gen_data.py

##2. Load the images
All the images from the vehicles and non-vehicles folder were loaded. It was resized into 64x64 and converted to vehicles.
```python
#read images from file names and resizes to 64,64
def data_read_images_from_files(file_names):
    images = [];
    for file in file_names:
        image = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB);
        image = cv2.resize(image,dsize=(64,64))
        images.append(image);
    return np.array(images)
```
Then I dumped this image as numpy array

```python
car.dump("car_images.dat")
non_car.dump("non_car_images.dat")
```
Example of Car - Non-Car
![Alt text](output_images/car-non-car.png?raw=true "car and non car images")

##3.Feature extraction
This was an iterative process during the developmental lifecycle of the project. The function we used for feature extraction was

```python
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in imgs:
        file_features = []
        # Read in each one by one
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return np.array(features)
```

We tried different color spaces, different orientation bins, calculating spatial features. We tried different features and we finally settled down on following parameters, since the testing accuracy was higher than other. Testing accuracy was 99.02
    
    ```python
    params = dict()
    params['color_space'] = 'YUV'
    params['hist_bins'] = 32
    params['orient'] = 9
    params['pix_per_cell'] = 8
    params['cell_per_block'] = 2
    params['hog_channel'] = 'ALL'
    params['spatial_feat'] = False
    params['hist_feat'] = True
    params['hog_feat'] = True
    params['spatial_size'] = (16, 16)
    ```

Once the feature is extracted, I dump the feature to file system
```python
car_features.dump("car_features_2.data")
    non_car_features.dump("non_car_features_2.data")
```

visualisation of hog features - Car Image
![Alt text](output_images/hog_vis.png?raw=true "Hog visulisation non-car")

visualisation of hog features - Non-car Image
![Alt text](output_images/hog_vis_non_car.png?raw=true "Hog visualisation car images")

The code can be found in extract_features.py

##4. Data pre-processing for SVM
Once the hog features were extracted, the hog features of car and non-car were stacked. A prediction  vector for car( =1 ) and non-car (=0) was created.

```python
all_data_x = np.vstack([car,non_car]).astype(np.float64)
#creates the y vector(output for the car and non features). Combines them into a single column vector.
#all_data_y holds the desired output
all_data_y = np.hstack((np.ones(len(car)),
              np.zeros(len(non_car))))

```

Then we scaled the feature vector using a standard scaler. This will help optimizer to converge faster.

```python
X_scaler = StandardScaler().fit(all_data_x)
# Apply the scaler to X
scaled_X = X_scaler.transform(all_data_x)
```

We then divided the data set into train and test set

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X_rem, y_rem, test_size=0.2, random_state=rand_state)
```
The code can be found in train.py

##5. Training SVM

Once the data was preprocessed, we used the default SVM to train the classifier as flows.

```python
svc = LinearSVC(random_state=rand_state,verbose=False,max_iter=2000)
svc.fit(X_rem, y_rem)
```
The accuracy was 99%.

The SVM model and the scaling matrix was saved.



##6. Method for detection

The code can be found in pipeline.py

1. Perform the sliding window technique( line no - 114-150)for each frame in the image and save the window. We use 2 different sized sliding windows. 

The parameter of sliding window are
```python

        y_start_stop_list = [[400, 700],[400,700]]
        x_start_stop_list = [[200, None],[200, None]]
        windows_list = [(64, 64),(128, 128)]
        xy_overlap_list = [(0.8, 0.8),(0.8, 0.8)]
```
We have a wrapper windows that performs multiple sliding windows and returns the appended list of windows.
```python
def slide_multiple(image,x_start_stop_list,y_start_stop,windows_list,xy_overlap_list):
    params = get_features_parameters()
    box_list = []
    for i in range(len(x_start_stop_list)):
        windows = slide_window(image, x_start_stop=x_start_stop_list[i], y_start_stop=y_start_stop[i],
                             xy_window= windows_list[i], xy_overlap=xy_overlap_list[i])

        hot_windows = search_windows(image,windows, svc, X_scaler, color_space=params['color_space'],
                                   spatial_size=params['spatial_size'], hist_bins=params['hist_bins'],
                                   orient=params['orient'], pix_per_cell=params['pix_per_cell'],
                                   cell_per_block=params['cell_per_block'],
                                   hog_channel=params['hog_channel'], spatial_feat=params['spatial_feat'],
                                   hist_feat=params['hist_feat'], hog_feat=params['hog_feat'])
        box_list = box_list + hot_windows
    return box_list
```
Below is the visualisation of sliding window
![Alt text](output_images/1slid_windows.jpg?raw=true "Sliding window visualisation")

2. Apply the scaling matrix and perform SVM prediction on windows. If the prediction is true, then the particular window is appended to hot_windows lists.

3. Since a real car will possibly have multiple hot windows at the same location, we combine those windows and construct a heat-map (line 19-28)

4. The heat-map constructed, is thresholded to remove false positive. We threshold based on the intensity of heat-map(lines 20-28) and the size of the bounding box of the heat-map.(lines 13-17)  and append it to heat-map-list (heap-map-list contains the heat map for last 5 frames)

5. We perform steps 1 to 3 for 5 consecutive frames

6. Once the 5 consecutive frame heat-map is obtained, we combine the heat-map(addition) (lines 31-35) and threshold it again to remove false positive.

7. Draw the bounding box on file detection.

## Below is the pipline for detection-
![Alt text](output_images/algo_vis.png?raw=true "Visualisation of the pipeline")


##7.Output
The output video youtube link
[![Output](output_images/youtube_output.png?raw=true)](https://youtu.be/snArg2ai6vA)

The output1.avi file contains the vehicle detection video. Here's a [link to my video result](output/output1.avi)

---

##8.Discussion

There is a number of issues with this method.
1. I have hand tuned the parameters (features, hog, color-histogram, spatial-histogram) to make this work. A deep neural network based features extraction is the way to go.
2. The sliding window technique is slow. 
3. Computer vision detection + tracking of the vehicle is needed.
4. We believe that the laser sensor based detection of the vehicle is better. Computer vision based technique can be used to track vehicle but may not accurately detect vehicles. Sensor function techniques can also be used to fuse images with laser sensors.
