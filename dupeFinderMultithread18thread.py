#  Copyright (c) 2020.
#  Project FindDuplicates
#  Author dedek@gloffer.com
#

import cv2
import numpy as np
import hashlib as hl
import matplotlib.pyplot as plt
import threading
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import time
import os
from PIL import Image
import sys
import argparse
from pathlib import Path
import shutil
import ntpath
import string
import re

###
### THIS SCRIPT ALLOWS FINDING DUPLICATE IMAGES IN A DEFINED SPACE ON A DRIVE. 
###

os.system("taskset -p 0xff %d" % os.getpid())
# CLASS DEFINITION
threadPool = ThreadPool(8)


# FUNCTION DEFINITION
# vytvori hash value pro zadany obrazek. Obrazek prevede do velikosti 8x7 px.
# Obrazek pred vyvorenim hashe zmeni na cernobily
# RUN ON MULTI THREAD
# @return string with hash value
def dhash(image, hash_size=8):
    # Grayscale and shrink the image in one step.
    image = image.convert('L').resize(
        (hash_size + 1, hash_size),
        Image.ANTIALIAS,
    )
    pixels = list(image.getdata())
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = image.getpixel((col, row))
            pixel_right = image.getpixel((col + 1, row))
            difference.append(pixel_left > pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2 ** (index % 8)
        if (index % 8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
            decimal_value = 0
    return ''.join(hex_string)

# prevod listu obrazku do cernobile barvy a ulozeni low res. "masky" pro detekci duplikatu
# RUN ON MULTITHREAD
def makeMasks(file_list, destination):
    fileSize = len(file_list)
    fileCounter = 0
    for filename in file_list:
            fileCounter = fileCounter+1
            print('Masking file ',fileCounter,'/',fileSize,' ',filename)
            img = Image.open(filename)
            size = 257, 256
            # img = img.resize((size), Image.ANTIALIAS)
            thresh = 210
            fn = lambda x: 255 if x > thresh else 0
            r = img.convert('L').point(fn, mode='1')


            tempname = destination + re.sub('[^0-9a-zA-Z._]+', '', str(filename))

            # tempname = tempname[:250]
            print (tempname)
            r.save(tempname)
            maskedfiles.append(tempname)

# bit difference between two values.
def hammingDistance(x, y):
    hammingdistancecounter=0
    for i in range( len(x)):
        if (x[i]!=y[i]):
            hammingdistancecounter = hammingdistancecounter+1
    return hammingdistancecounter

# VARIABLE DEFINITION
parser = argparse.ArgumentParser(description='Duplicate image finder with the use of perceptual hashing.')
parser.add_argument("--images", type=str, help="PATH ke slozce se soubory. Pridat na konec lomitko '/'. ")
parser.add_argument("--destination", type=str, help="PATH ke slozce pro vystup. Pridat na konec lomitko '/'. ")
args = parser.parse_args()

imageFolder = args.images
destinationFolder = args.destination
print(imageFolder, destinationFolder)
maskedfiles = []
maskedfiles2 = []
imagelist = dict()
hash_keys = dict()
duplicates = []
computedDuplicates = []
progressSize = 0
progressCounter = 0
if (imageFolder and destinationFolder):
    resultJPG = list(Path(imageFolder).rglob("*.[jJ][pP][gG]"))
    resultPNG = list(Path(imageFolder).rglob("*.[pP][nN][gG]"))
    file_list = resultPNG+resultJPG

    maskProcess1= multiprocessing.Process(target=makeMasks, args=(file_list, destinationFolder,))
    maskProcess2 = multiprocessing.Process(target=makeMasks, args=(file_list, destinationFolder,))
    maskProcess3 = multiprocessing.Process(target=makeMasks, args=(file_list, destinationFolder,))
    maskProcess4 = multiprocessing.Process(target=makeMasks, args=(file_list, destinationFolder,))
    procIter = 0
    maskProcesses = []
    while(procIter<18):
        maskProcesses[procIter]=(multiprocessing.Process(target=makeMasks, args=(file_list, destinationFolder,)))
        maskProcesses[procIter].start()
        maskProcesses[procIter].join()
        procIter = procIter+1


    progressSize = len(maskedfiles)
    for index, filename in enumerate(maskedfiles):
        with Image.open(filename) as f:
            progressCounter = progressCounter+1
            print('Hashing file ',progressCounter,'/',progressSize)
            filehash = dhash(f)
            imagelist[filename]=index
            # if filehash is not in hash_keys:
            hash_keys[index] = filehash

    for x in hash_keys:
        if (x not in computedDuplicates):
            for y in hash_keys:
                if (x != y):
                    hammDist = hammingDistance(hash_keys[x], hash_keys[y])
                    if (hammDist < 4):
                        duplicates.append((x, y))
                        #pridat appendovane prvky do listu, takze uz pro ne nemusim duplikaty hledat
                        computedDuplicates.append(y)

    parentDupe = set(map(lambda x: x[0], duplicates))
    childDupe = [[y[1] for y in duplicates if y[0] == x] for x in parentDupe]

    for index, parentVal in enumerate(parentDupe):
        print(parentVal,'->',childDupe[index])
    progressCounter=0
    progressSize=len(parentDupe)

    for index,parentVal in enumerate(parentDupe):
        # pro kazdeho parenta nakopiruju jeho file do slozky a pak vsechny jeho child duplikaty
        progressCounter = progressCounter+1
        print('Creating data structures. File',progressCounter,'/',progressSize)
        # tady prijdou vsechny jeho child files
        dirpath = destinationFolder + hash_keys.get(parentVal) + '/'
        os.makedirs(dirpath, exist_ok=True)
        # copy the parent image in the mask folder
        parentMaskDestination = dirpath + re.sub('[^0-9a-zA-Z.]+', '', maskedfiles[parentVal])
        parentMaskDestination = parentMaskDestination[:250]
        shutil.copy(maskedfiles[parentVal], parentMaskDestination)
        os.remove(maskedfiles[parentVal])
        for childVal in childDupe[index]:
            maskDestination = dirpath + re.sub('[^0-9a-zA-Z.]+', '', maskedfiles[childVal])
            maskDestination = maskDestination[:250]
            shutil.copy(maskedfiles[childVal], maskDestination)
            os.remove(maskedfiles[childVal])
    # testing block

    cv2.waitKey()
else:
    print('no files in image dir')


