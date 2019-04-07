
# coding: utf-8

# In[ ]:

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import sys
import ids
from PIL import Image
import time
import tifffile
import serial
from tqdm import tqdm

MEA = '''
.....................
. ID     .....        .
.      ..     ..      .
.      .       .      .
.      ..     ..      .
.        .....        .                     .
. O                 O .
  .....................
'''

print 'Align MEA properly to x and y axis of joystick-controller and place MEA on the specimen holder like that:'
print MEA
print 'Open ueyedemo and navigate to the upper left corner of the MEA'
MEA_ID = raw_input('Enter MEA-ID:')

begin=time.time()
today=datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day
Folder_name='-'.join(map(str,today))
filepath = '/home/user/Desktop/Data/'+str(Folder_name)+'/'+str(MEA_ID)+'/'
print 'Current data is recorded in the folder ' + filepath

#if already created, don't create.
try:
    os.makedirs(filepath)
except:
    pass

print 'Mounting USB port...'

try:
    with serial.Serial('/dev/ttyACM0', timeout=1) as ser:
        x = ser.read()
        s = ser.read(10)
        line = ser.readline()
    ser.open()
except:
    os.system('sudo chmod 666 /dev/ttyACM1')
    with serial.Serial('/dev/ttyACM1', timeout=1) as ser:
        x = ser.read()
        s = ser.read(10)
        line = ser.readline()
    ser.open()

print 'USB port is mounted'
#Test usb
ser.write(b'GR,20000,20000\r\n'); time.sleep(0.75); ser.write(b'GR,-20000,-20000\r\n')

time.sleep(1)

print 'Connecting to camera...'
cam = ids.Camera()
cam.color_mode = ids.ids_core.COLOR_RGB8
cam.exposure = 20
cam.auto_exposure=True
cam.gain=100
print 'Camera connected. Taking test images:'

#Test images
pbar = tqdm(total=5,unit='image')
for i in range(5):
	cam.continuous_capture = True
	img,meta=cam.next()
	cam.continuous_capture = False
	time.sleep(1)
	pbar.update(1)
pbar.close()


print 'Starting image acquisition...'
pbar = tqdm(total=140,unit='image')
stitch=[]
for i in range(10):
	for u in range(14):
		cam.continuous_capture = True
		img,meta=cam.next()
		cam.continuous_capture = False
        	stitch.append(img)
		ser.write(b'GR,29678,0\r\n'); time.sleep(0.25)
		pbar.update(1)
	if i != 9:
		#Move to next row
		ser.write(b'GR,-415492,20200\r\n'); time.sleep(1)
	else:
		#Move back to start
		time.sleep(0.5); ser.write(b'GR,-415492,-181800\r\n'); cam.close(); pbar.close()

'''
When using the high-frame camera, use this loop instead:
stitch=[]
for i in range(7):
	for u in range(7):
		cam.continuous_capture = True
		img,meta=cam.next()
		cam.continuous_capture = False
        	stitch.append(img)
		time.sleep(1); ser.write(b'GR,56595,0\r\n'); time.sleep(1)
	if i != 6:
		#Move to next row
		time.sleep(1); ser.write(b'GR,-396165,35000\r\n'); time.sleep(1)
	else:
		#Move back to start
		time.sleep(1); ser.write(b'GR,-396165,-210000\r\n'); time.sleep(1)
'''


print 'Saving images to .tiff-file...'	
stitch=np.asarray(stitch)
tifffile.imsave(str(filepath)+'/'+str(MEA_ID)+'.tif',stitch[:,:,:,1],compress=6)
ending=time.time()
elapsed_time = ending-begin
print 'Stitching finished. Time elapsed: {} s.'.format(elapsed_time)
