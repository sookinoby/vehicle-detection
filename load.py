import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2


# from skimage.feature import hog
# from skimage import color, exposure
# images are divided up into vehicles and non-vehicles

import os

path = 'vehicles'

#read all the car images file path
car_files = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(path)
    for f in files if f.endswith('.png')]

path2 = 'non-vehicles'


#read all the non-car images file path
non_car_files = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(path2)
    for f in files if f.endswith('.png')]

#print number of files availabel
print(len(car_files))
print(len(non_car_files))


#read images from file names and resizes to 64,64
def data_read_images_from_files(file_names):
    images = [];
    for file in file_names:
        image = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB);
        image = cv2.resize(image,dsize=(64,64))
        images.append(image);
    return np.array(images)





# Define a function to return few characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = car_list[0]
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict

#all car images
car = data_read_images_from_files(car_files);
#all non car images
non_car = data_read_images_from_files(non_car_files)
#get infor about the data
data_info = data_look(car, non_car)

#print the data info
print('Your function returned a count of',
      data_info["n_cars"], ' cars and',
      data_info["n_notcars"], ' non-cars')
print('of size: ', data_info["image_shape"], ' and data type:',
      data_info["data_type"])

#get the first image for visualization from car and non car
car_image_1 = car[0]
not_car_image_1 = non_car[0]

#dunp data back to the file system as numpy array
car.dump("car_images.dat")
non_car.dump("non_car_images.dat")

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image_1)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(not_car_image_1)
plt.title('Example Not-car Image')
plt.show()
