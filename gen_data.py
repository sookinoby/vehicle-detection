import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
import glob

#We noticed that the car svm classifier doesnt detect the black car in images. So added few black car images the training set
#This files is used to generated additional black car data. I created some black car images, and flipped, translated those car images for additional data

#read black car  file name
def readFile():
    names = glob.glob("black-car/*.png")
    return names

#read images
def readImages(filenames):
    images  = []
    for name in filenames:
        image = cv2.imread(name)
        images.append(image)
    return images

#randomly translates images
def random_trans(image, trans_range):
    rows,cols,_ = image.shape;
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = 20 * np.random.uniform() - 20 / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))
    return image_tr

#randomly flip images
def random_flip(image):
    n = np.random.randint(0, 2)
    if n == 0:
        image = cv2.flip(image, 1)
    return image

#generated images
def gen_random_data(images,count):
    counter = 0
    current = 0
    new_images = []
    while counter < count:
        cur = images[current]
        cur_trans = random_trans(cur,20)
        cur_flip = random_flip(cur_trans)
        new_images.append(cur)
        new_images.append(cur_trans)
        new_images.append(cur_flip)
        current = current + 1
        if current > len(images) -1:
            current = 0
        counter = counter + 3
    return new_images

#write generated images to the training set images folder
def write_images(gen_images):
    count = 0
    for image in gen_images:
        cv2.imwrite("extra/bc"+ str(count)+".png",image)
        count = count + 1


filenames = readFile()
images = readImages(filenames)
gen_images = gen_random_data(images,1000)
write_images(gen_images)

#check the image generated
plt.imshow(gen_images[2])
plt.show()