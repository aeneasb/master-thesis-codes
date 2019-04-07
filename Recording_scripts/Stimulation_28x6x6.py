## STIMULATION SCRIPT ##

# C++ and Python libraries

import sys
import os
import time
import random
import datetime
import mea1k
import libarray
import numpy as np
import mea1kusr.init
import mea1kusr.save as stst
import easygui as g

##This script intends to stimulate 25x28 electrodes to determine neural activity
# on random MEA cultures.

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


# channels is a list of stimulation buffer numbers
# dac is the dac source. Can be 1,2 or 3.
# dac corresponds to:
# dac = 0 := VRef (midsupply)
# dac = 1 := DAC0
# dac = 2 := DAC1
# dac = 3 := DAC2
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


# cmdDAC( dacNumber , value1, value2 )
# dacNumber = 0 := sets DAC0 to value1
# dacNumber = 1 := sets DAC1 to value1
# dacNumber = 2 := sets DAC2 to value1
# dacNumber = 3 := sets DAC0 to value1 and DAC1 to value2
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


#Linear indices of stimulation electrodes+boxes using the 28x6x6 HEX-files
box_size=6
jump_size=30
sx,sy=7,4
cx,cy=5,5
side=np.arange(box_size)
xx,yy=np.meshgrid(side,side)
xx_list=[]
yy_list=[]
#Electrode coordinates of stimulation electrodes in a box
stim_x,stim_y=3,3

#x/y-coordinates of all stimulation electrodes?boxes.
x_y_elec = [[[stim_x+u*box_size+jump_size*i,stim_y+p*box_size+jump_size*j] \
for i in range(sx) for j in range(sy)] for u in range(cx) for p in range(cy)]
box_coord = [[[xx+u*box_size+jump_size*i,yy+p*box_size+jump_size*j] \
for i in range(sx) for j in range(sy)] for u in range(cx) for p in range(cy)]


#Linear indices of the stimulation electrodes.
lin_ind_elec = [[x_y_elec[j][i][0]+x_y_elec[j][i][1]*220 \
for i in range(sx*sy)] for j in range(cx*cy)]
linear_index_of_boxes = [[box_coord[j][i][0]+box_coord[j][i][1]*220 \
for i in range(sx*sy)] for j in range(cx*cy)]

title = 'Checkpoint'
msg = 'Enter the MEA-ID and mount the MEA on the recording unit:'
MEA_ID = g.enterbox(msg,title)
today=datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day
Folder_name='-'.join(map(str,today))
outputDir = '/home/user/Desktop/Data/'+str(Folder_name)+'/'+str(MEA_ID)+'/28_6_6_Stim'
msg='The h5 files are recorded in the folder '+ str(outputDir)
g.msgbox(msg)


##User specifications
# Insert here the number of pulse repetitions
number_of_repetitions = 5

# Insert here the time [ms] between the repetitions in ms        freq = 1/interTrainDelay; 200 = 5Hz, 100=10Hz, 5 =200Hz
interTrainDelay = 100 # == 200Hz

# Insert here the amplitude of the cathodic part of the signal -- 1bit = 2.92 mV
amp1 = 5

# Insert here the amplitude of the anodic part of the signal -- 1bit = 2.92 mV
amp2 = 5

# Insert here the phase of the cathodic part of the signal -- 1S = 50 usec
phase1 = 4

# Insert here the phase of the anodic part of the signal -- 1S = 50 us
phase2 = 4

#Insert here the config directory
configDir = "/opt/MaxLab/configs/28x6x6/"


#Default initialization
mea1kusr.init.board();
mea1kusr.init.chip();

## Send the signal
print 'make chip'
chip = libarray.Chip();
print 'done'
save = stst.save()

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

print 'Offset MEA around 512 bits...'
import mea1kusr.api
Api = mea1kusr.api.Api()
Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
print 'Offset done'

## Save && run scan
save = mea1kusr.save.save('localhost')
save.mkDir( outputDir )
save.reset( )

for i,j in enumerate(lin_ind_elec):
    cfgFile = configDir +'{0:03}.hex.nrk'.format(i)

    stimChannels = []

    chip.loadHEX( cfgFile )
    time.sleep(2)
    for items in j:
        chip.electrodeToStim(items)
        stimChannels.append(chip.queryStimAtElectrode(items))

    #Sometimes an electrode is not routed for some reason: Try another electrode whithin the box
    while -1 in stimChannels:
        flatened_box = np.reshape(linear_index_of_boxes[i][stimChannels.index(-1)],(1,36))[0]
        #Select another electrode from the box
        next_el = random.choice(flatened_box)
        stimChannels.remove(-1)
        chip.electrodeToStim(next_el)
        stimChannels.append(chip.queryStimAtElectrode(next_el))

    chip.download()
    time.sleep(1)
    print stimChannels

    save.openDir( outputDir )
    save.mapping(chip.get_mapping( ))

    c = mea1k.Config()
    c.add(switchOffAllChannels())
    c.add(switchOnChannels(stimChannels,1))
    c.add( mea1k.cmdDelaySamples(100))
    for p in range(number_of_repetitions):
        c.add(voltageBiPhasicPulse(512,amp1,amp2,phase1,phase2))
        c.add(mea1k.cmdDelaySamples(interTrainDelay))
    train_time = (number_of_repetitions*interTrainDelay+1000)/1000.
    save.start('')
    c.send()
    time.sleep(train_time)   #add here seconds to record after stimulating in the parenthesis
    save.stop()
    time.sleep(2)
