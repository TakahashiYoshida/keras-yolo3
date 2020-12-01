import os
import argparse
import shutil
import time
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img
#from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import random

try:
    shutil.rmtree("edit_image")
except:
    pass

os.mkdir("edit_image")
image_list=glob.glob("raw_image/*/*.jpg")
#image_list=glob.glob("edit_image/*.png")
print(image_list)

for index,image in enumerate(image_list):

    img=cv2.imread(image)
    
    tmp = img[:, :]
    height, width = img.shape[:2]
    if(height > width):
        size = height
        limit = width
    else:
        size = width
        limit = height
    start = int((size - limit) / 2)
    fin = int((size + limit) / 2)
    img = cv2.resize(np.ones((300, 300, 3), np.uint8)*255, (size, size))
    if(size == height):
        img[:, start:fin] = tmp
    else:
        img[start:fin, :] = tmp
    
    
    filename=os.path.splitext(os.path.basename(image))[0]
    foldername=os.path.basename(os.path.dirname(image))
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray)
    
    thresh, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(binary)
    
    
    cv2.drawContours(mask, contours, -1, color=255, thickness=-1)
    print(np.unique(mask))
    #plt.imshow(mask)
    
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    
    rgba[..., 3] = mask
    #plt.imshow(rgba)

    #plt.show()
    try:
        os.mkdir("edit_image/"+foldername)
    except:
        pass
    cv2.imwrite("edit_image/"+foldername+"/"+filename+".png", rgba)

image_list=glob.glob("raw_image/*/*.png")
for index,image in enumerate(image_list):
    filename=os.path.splitext(os.path.basename(image))[0]
    foldername=os.path.basename(os.path.dirname(image))
    try:
        os.mkdir("edit_image/"+foldername)
    except:
        pass
    img=cv2.imread(image,cv2.IMREAD_UNCHANGED)
    cv2.imwrite("edit_image/"+foldername+"/"+filename+".png", img)
    #shutil.copy(image,"edit_image/"+foldername+"/")

def draw_images(generator, x, dir_name, index):
    save_name = 'extened-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=output_dir,
                       save_prefix=save_name, save_format='png')

    for i in range(1000):
        bach = g.next()

  
dir_list=glob.glob("edit_image/*")
seed=1

for output_dir in dir_list:
    print(output_dir)
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)


    images = glob.glob(os.path.join(output_dir, "*.png"))


    datagen = ImageDataGenerator(rotation_range=180,
                                width_shift_range=0,
                                shear_range=0.3,
                                height_shift_range=0,
                                zoom_range=[1.3,1.5],
                                horizontal_flip=False,
                                fill_mode="nearest",
                                channel_shift_range=90,
                                brightness_range=[0.7,1],
                                #preprocessing_function=blur
                                )


    for i in range(len(images)):
        img = load_img(images[i],color_mode="rgba")
        img = img.resize((416,416 ))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        draw_images(datagen, x, output_dir, i)

class_dict={"background":0,
           "rasp0w":10,
           "rasp3a+":20,
           "rasp3b":30,
           "rasp3b+":40,
           "rasp4b":50}


try:
    shutil.rmtree("../VOCdevkit")
except:
    pass

os.mkdir("../VOCdevkit")
os.mkdir("../VOCdevkit/VOC2007")
os.mkdir("../VOCdevkit/VOC2007/JPEGImages")
os.mkdir("../VOCdevkit/VOC2007/mask")


back_list=glob.glob("Images/*/*.jpg")

dir_list=glob.glob("edit_image/**/")

keys=[]
values=[]
for d in dir_list:
    keys.append(d)
    values.append(glob.glob(d+"*.png"))
dic=dict(zip(keys,values))

def get_back(back_list):
    try:
        back=random.choice(back_list)
        back=cv2.imread(back)
        back=cv2.resize(back,(416,416))
        back=back.astype(np.float64)
    except:
        back=get_back(back_list)
    return back
len_dataset=10000
w_dir="train/"
for i in tqdm(range(len_dataset)):
    if i>(len_dataset*0.85): w_dir="test/"
    back=get_back(back_list)
    back_mask=np.full((back.shape[0],back.shape[1]),0)
    #back=cv2.cvtColor(back, cv2.COLOR_BGR2RGB)
    max_grid=min(back.shape[0],back.shape[1])//2
    grid_size=random.randint(max_grid//6,max_grid)
    h_grid=back.shape[0]//(grid_size*2)
    w_grid=back.shape[1]//(grid_size*2)
    p_top=back.shape[0]//2-grid_size*h_grid
    p_left=back.shape[1]//2-grid_size*w_grid
    #print(p_top,p_left)
    grid=np.ones((h_grid*2,w_grid*2))
    cnt=1#cnt=random.randint(1,5)#
    #image_list=glob.glob("edit_image/*/*.png")
    f=True
    for idx,img in enumerate(random.choices(dir_list,k=cnt)):
        img=random.choice(dic[img])
        dirname=os.path.basename(os.path.dirname(img))
        img=cv2.imread(img,cv2.IMREAD_UNCHANGED)

        img_mask_bin=img[...,3]
        img_mask_bin=img_mask_bin/255.0
        #img_mask_bin=img_mask_bin.astype(np.int64)
        img_mask_bin=np.round(img_mask_bin)
        img_mask_bin=img_mask_bin.astype(np.float32)
        #img_mask_bin=np.where(img_mask_bin!=0,1,0)
        print(np.unique(img_mask_bin,return_counts=True))
        img_mask=cv2.cvtColor(img_mask_bin,cv2.COLOR_GRAY2BGR)
        #img_mask=img_mask/255.0
        img=img[:,:,:-1]
        img=img.astype(np.float64)
        
        
        flag=True
        c=0
        while flag:
            c+=1
            left=random.randint(0,w_grid*2)#
            top=random.randint(0,h_grid*2)#

            img_grid=random.randint(2,max(min(w_grid*2+1,h_grid*2+1),3))#

            if left+img_grid<=w_grid*2 and top+img_grid<=h_grid*2:
                if np.all(grid[top:top+img_grid,left:left+img_grid]):
                    #try:
                    grid[top:top+img_grid,left:left+img_grid]=np.zeros((img_grid,img_grid))
                    #grid_shape=min(grid_size*img_grid,back.shape[0]-grid_size*top,back.shape[1]-grid_size*left)
                    back[p_top+grid_size*top:p_top+grid_size*top+grid_size*img_grid,p_left+grid_size*left:p_left+grid_size*left+grid_size*img_grid]*=1-cv2.resize(img_mask,(grid_size*img_grid,grid_size*img_grid))
                    back[p_top+grid_size*top:p_top+grid_size*top+grid_size*img_grid,p_left+grid_size*left:p_left+grid_size*left+grid_size*img_grid]+=cv2.resize(img_mask,(grid_size*img_grid,grid_size*img_grid))*cv2.resize(img,(grid_size*img_grid,grid_size*img_grid))
                    img_mask_bin=cv2.resize(img_mask_bin,(grid_size*img_grid,grid_size*img_grid))
                    img_mask_bin=np.where(img_mask_bin!=0,1,0)
                    back_mask[p_top+grid_size*top:p_top+grid_size*top+grid_size*img_grid,p_left+grid_size*left:p_left+grid_size*left+grid_size*img_grid]=img_mask_bin*(class_dict[dirname]+idx)
                    flag=False
                    f=False
                    #except:
                        #pass
            if c==1000:
                flag=False
                #print(back.shape,grid_size*top,grid_size*top+grid_shape,grid_size*left,grid_size*left+grid_shape)

    if f:
        print(i)
        #print(np.unique(back_mask,return_counts=True))
    else:
        cv2.imwrite("../VOCdevkit/VOC2007/JPEGImages/"+dirname+str(i).zfill(4)+".png", back)
        cv2.imwrite("../VOCdevkit/VOC2007/mask/"+dirname+str(i).zfill(4)+".png", back_mask)
plt.imshow(back)
plt.show()
plt.imshow(back_mask)
plt.show()