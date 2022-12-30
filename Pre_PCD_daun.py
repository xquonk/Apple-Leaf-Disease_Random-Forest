import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from datetime import datetime
start = datetime.now()

pd.set_option('display.max_columns', None)
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0], lvl=256, sym=True,norm=True):
    glcm = graycomatrix(img,
                        distances=dists,
                        angles=agls,
                        levels=lvl,
                        symmetric=sym,
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)
    feature.append(label)

    return feature

df = pd.DataFrame(columns=['area','perim','red_mean','green_mean','blue_mean','f1','f2','red_std','green_std','blue_std'])
train_folder = 'Train_leaf'
train_data = []
labels = [] #Label feature for dataframe
real_label = {} #real labels from the actual dataset
i = 0 #Enumerate each class of data
kernel = np.ones((5, 5), np.uint8)
for class_ in os.listdir(train_folder):
    [this_class,this_label] = class_.split('___')
    class_folder = os.path.join(train_folder,class_)
    this_class = i
    real_label[int(this_class)]=this_label

    #taking 500 images from dataset
    for img_name in os.listdir(class_folder)[:500]:
        data = []
        img_path = os.path.join(class_folder,img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (250,250), interpolation = cv2.INTER_AREA)

        #Convert to gray
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #Gaussian Blur
        blur =cv2.GaussianBlur(gray,(9,9),0)

        #Otsu Threshold
        otsu_threshold, image_result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #Filling Holes (Opening)
        opening = cv2.morphologyEx(image_result, cv2.MORPH_OPEN, kernel)
        opening = cv2.bitwise_not(opening)

        #Masking to rgb image
        r, g, b = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        r = cv2.bitwise_and(r,r,mask=opening)
        g = cv2.bitwise_and(g,g, mask=opening)
        b = cv2.bitwise_and(b,b, mask=opening)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        red_mean = np.mean(r)
        green_mean = np.mean(g)
        blue_mean = np.mean(b)
        red_std = np.std(r)
        green_std = np.std(g)
        blue_std = np.std(b)
        img = (np.dstack((r, g, b))).astype(np.uint8)
        cv2.imshow('',img)

        #Get the largest value of perimeter and area of contours
        contours, hierarchy = cv2.findContours(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        perim = 0
        for cnt in contours:
            perim1 = cv2.arcLength(cnt,True)
            if perim<perim1:
                area=cv2.contourArea(cnt)
                perim = cv2.arcLength(cnt,True)

        #HSV for getting green part ratio from leaf image
        HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(HSV)
        height,width = h.shape
        green_part = 0
        for row in range(height):
            for col in range(width):
                if (h[row,col]<70)and(h[row,col]>30):
                    green_part = green_part+1
        green_part = green_part/(height*width)
        non_green_part = 1 - green_part
        train_data.append(img)
        labels.append(real_label[int(this_class)])
        data.append(area)
        data.append(perim)
        data.append(red_mean)
        data.append(green_mean)
        data.append(blue_mean)
        data.append(green_part)
        data.append(non_green_part)
        data.append(red_std)
        data.append(green_std)
        data.append(blue_std)
        data = pd.DataFrame([data],columns=['area','perim','red_mean','green_mean','blue_mean','f1','f2','red_std','green_std','blue_std'])
        df = pd.concat([df,data],ignore_index=True)
print('Loading data done!')

glcm_all_agls = []
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
for img, label in zip(train_data,labels):
    glcm_all_agls.append(
            calc_glcm_all_agls(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),
                                label,
                                props=properties)
                            )

properties.append('labels') #Adding 'labels' feature from preprocessing image
glcm_df = pd.DataFrame(glcm_all_agls, columns = properties)
df = pd.merge(df, glcm_df, left_index=True, right_index=True)
df_path = 'D:\data\preprocess_daun.csv'
df.to_csv(df_path)
# print(glcm_df.head(15)) optional: if you wanna see the datahead of glcm features

#preprocess time
print('Computation speed :',datetime.now()-start)