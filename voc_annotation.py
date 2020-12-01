import xml.etree.ElementTree as ET
import os
from os import getcwd
import glob
from PIL import Image
import numpy as np

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

#classes = ["rasp", "bicycle", "bird", "boat", "bottle"]

"""
def convert_annotation(year, image_id, list_file):
    #in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))

    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)),
             int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)),
             int(float(xmlbox.find('ymax').text)))


        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
"""
wd = getcwd()

image_ids=glob.glob("VOCdevkit/VOC2007/JPEGImages/*.png")
#print(image_ids)
len_img=len(image_ids)
year,image_set=sets[0]
list_file = open('%s_%s.txt'%(year, image_set), 'w')
for idx,image_id in enumerate(image_ids):
    image_id=os.path.splitext(os.path.basename(image_id))[0]
    mask=Image.open("VOCdevkit/VOC2007/mask/"+image_id+".png")
    mask=np.array(mask)
    obj_ids=np.unique(mask)
    obj_ids=obj_ids[1:]
    masks=mask==obj_ids[:,None,None]
    if idx==int(len_img*0.8):
        list_file.close()
        year,image_set=sets[1]
        list_file = open('%s_%s.txt'%(year, image_set), 'w')
    if idx==int(len_img*0.9):
        list_file.close()
        year,image_set=sets[2]
        list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for index,obj_id in enumerate(obj_ids):
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.png'%(wd, year, image_id))
        pos=np.where(masks[index])
        xmin = int(np.min(pos[1]))
        xmax = int(np.max(pos[1]))
        ymin = int(np.min(pos[0]))
        ymax = int(np.max(pos[0]))
        b=(xmin, ymin, xmax, ymax)
        cls_id=obj_id//10-1
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        list_file.write('\n')


list_file.close()
"""
for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.png'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()

"""