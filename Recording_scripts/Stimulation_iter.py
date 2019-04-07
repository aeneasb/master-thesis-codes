## STIMULATION SCRIPT ##

# C++ and Python libraries

import sys
import os
import time
import numpy as np

import mea1k
import libarray
import mea1kusr.init
import mea1kusr.save as stst

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import datetime
import easygui as g
from h5py import File
from shutil import copyfile

from pandas import HDFStore,DataFrame,Series

def manual_annot(el_array):

    #Calling ginput to interactively select electrodes
    fig=plt.figure(figsize=(22,12))
    imgplot = plt.imshow(el_array[:,:],'hot')
    plt.title('Choose Stimulation electrodes: left click: add point | right click: remove last point | middle click: stop input')
    plt.show(block=False)
    points = np.around(np.asarray(plt.ginput(n=200,show_clicks=True,mouse_add=1,mouse_pop=3,mouse_stop=2)))
    plt.close()

    #Convert the coordinates into ints
    xses=map(int,[t[0] for t in points])
    yses=map(int,[t[1] for t in points])
    el_array[yses,xses]=0.5

    el_grid=np.reshape(np.arange(26400),(120,220))
    chosen_el = el_grid[yses,xses]
    return chosen_el

def switchOffAllChannels():
    c.add( mea1k.cmdStimBuffer(              # shutdown all stimchannels
        	dac=0,        # 0 == VRef, 1 == DAC0
        	broadcast=1,
	        channel=0,
	        autozero=0,   # 0 == off
        	azoff=1,      # 1 == disable AZmode
        	Imode=0,      # 0 == voltage
        	Irange=0,     # 0 == small
        	power=1))
    return c


def switchOnChannels(channels, dac):
    c = mea1k.Config()
    for channel in channels:
        c.add( mea1k.cmdStimBuffer(
        	dac=dac,        # 0 == VRef, 1 == DAC0
        	broadcast=0,
	        channel=channel,
	        autozero=0,   # 0 == off
        	azoff=1,      # 1 == disable AZmode
        	Imode=0,      # 0 == voltage
        	Irange=0,     # 0 == small
        	power=0))     # 0 == on
    return c


def voltageBiPhasicPulse(offset, ampBits_1, ampBits_2, samples_1=4, samples_2=4):
    	c = mea1k.Config()
    	c.add( mea1k.cmdDelaySamples(400) ) # waits 20ms
        c.add( mea1k.cmdStatusOut( 1 ) ) #..#
    	# Turn on stimulation buffer
    	c.add( mea1k.cmdDAC( 0, offset-ampBits_1, 512 ) )#pulse one towards up
    	c.add( mea1k.cmdDelaySamples( samples_1 ) ) #=4 ~ 200us ?? doesn't add up
    	c.add( mea1k.cmdDAC( 0, offset+ampBits_2, 512 ) )#pulse two towards down
    	c.add( mea1k.cmdDelaySamples( samples_2 ) )#=4 ~ 200us ?? doesn't add up
    	c.add( mea1k.cmdDAC( 0, offset, 512 ) )#pulse three towards ~0
    	return c



title = 'User specifications'
msg = 'Enter the MEA-ID and mount the MEA on the recording unit:'
MEA_ID = g.enterbox(msg,title)

today=datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day
Folder_name='-'.join(map(str,today))
outputDir = '/home/user/Desktop/Data/'+str(Folder_name)+'/'+str(MEA_ID)+'/Stimulation'

msg='The h5 files are recorded in the folder '+ str(outputDir)
g.msgbox(msg)
msg = 'Specify the directory of the configs.'
default = '/home/user/Desktop/Data'
path = g.diropenbox(msg,title,default)

# Insert here the number of repetitions
number_of_repetitions = 1

# Insert here the time [ms] between the repetitions in ?        fre   q = 1/interTrainDelay; 200 = 5Hz, 100=10Hz, 5 =200Hz
interTrainDelay = 10000 # 10000 = 0.5s

# Insert here the amplitude of the cathodic part of the signal -- 1bit = 2.92 mV
amp1 = 70

# Insert here the amplitude of the anodic part of the signal -- 1bit = 2.92 mV
amp2 = 70

# Insert here the phase of the cathodic part of the signal -- 1S = 50 usec
phase1 = 4

# Insert here the phase of the anodic part of the signal -- 1S = 50 us
phase2 = 4

#Set gain back to normal operation conditions
mea1kusr.init.board();
mea1kusr.init.chip();

save = stst.save('localhost')
save.mkDir(outputDir)
save.reset()

#Create a list with all the stimulation electrodes
stim_list = []

title = 'User specifications'
msg = 'Stimlist: Import an existing list of stimulation electrodes from previous sessions | Select El.: Select stimulation electrodes'
choices = ['Stimlist','Select El.']
reply = g.buttonbox(msg,title,choices=choices)

if reply == 'Stimlist':
	msg = 'Specify the file of the Stimulation list.'
	default = '/home/user/Desktop/Data/'
	stim_path = g.fileopenbox(msg,title,default)
	stim_dataset = File(stim_path,'r')
	for i in range(len(stim_dataset)):
		stim_el = stim_dataset['Stim_el_structure_'+str(i+1)]['values'][:]
		stim_list.append(stim_el)
		copyfile(stim_path,outputDir+'/stimlist_logfile.h5')

if reply == 'Select El.':
	hdf = HDFStore(outputDir+'/stimlist_logfile.h5')
        f = File(path+'/segmentation_logfile.h5', "r")
	for i in range(len(f)):
		#Import the metadata of the configurations.
		routed_electrodes = np.asarray(f['structure_'+str(i+1)]['gotten_el'][:])
		x = routed_electrodes%220
		y = routed_electrodes/220
		el_array=np.zeros((120,220))
		el_array[y,x]=1
		stim_list.append(manual_annot(el_array))
		hdf.put('Stim_el_struct'+str(i+1),Series(stim_list[i]))
 	hdf.close()


print 'make chip'
chip = libarray.Chip();
print 'done'

# Turn on power for the stimulation buffers
mea1k.go( mea1k.cmdCore(
        onChipBias=1,
        stimPowerDown=0,
        outputEn=1,
        spi=0,                      # 0 == DataxDO off
        tx=0,                       # 0 == DAC0
        rstMode=0,                  # 0 == auto
        offsetCyc=7,
        resetCyc=7,
        wlCyc=7))

#default gain of 512 - if needed, change below
# 7x1x1(7)
# 7x8x2(112)
# 16x16x2(512)
# 16x16x4(1024)
# 16x32x4(2048)

mea1k.go( mea1k.cmdReadout(
        s1Bypass=0,
        s1Gain=1,                   # 1 == x16   0 == x7
        s1RstMode=0,                # 0 == disconnect
        s2Bypass=0,
        s2Gain=5,                   # 5 == x16
        s3Bypass=0,
        s3Gain=0))                  # 0 ==x2




for t,e in enumerate(stim_list):
    configFile = path+'/config_structure_'
    configFile += str(t+1)+'/structure_'+str(t+1)+'.hex.nrk'

    chip.loadHEX( configFile )

    print 'Offset MEA around 512 bits...'
    import mea1kusr.api
    Api = mea1kusr.api.Api()
    Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
    print 'Offset done'

    time.sleep(1)
    stimChannels = []

    for items in e:
        chip.electrodeToStim(items)
        stimChannels.append(chip.queryStimAtElectrode(items))

    while -1 in stimChannels:
        next_el = e[stimChannels.index(-1)]+1
        stimChannels.remove(-1)
        chip.electrodeToStim(next_el)
        stimChannels.append(chip.queryStimAtElectrode(next_el))

	chip.download()
    time.sleep(2)
    print stimChannels

    print 'Offset MEA around 512 bits...'
    import mea1kusr.api
    Api = mea1kusr.api.Api()
    Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
    print 'Offset done'

    save.openDir(outputDir)
    save.mapping(chip.get_mapping( ))

    c = mea1k.Config()
    c.add(switchOffAllChannels())
    c.add(switchOnChannels(stimChannels,1))
    c.add(mea1k.cmdDelaySamples(100))
    for u in range(number_of_repetitions):
        c.add(voltageBiPhasicPulse(512,amp1,amp2,phase1,phase2))
        c.add(mea1k.cmdDelaySamples(interTrainDelay))
    train_time = (number_of_repetitions*interTrainDelay)/10000.+1

    save.start('configstructure_{0:03}'.format(t+1))
    c.send()

    time.sleep(train_time)
    save.stop()
    print t
    time.sleep(2)
