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
import random


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


def switchOnChannel(buffer, dac):
    print electrode
    c = mea1k.Config()
    c.add( mea1k.cmdStimBuffer(
    	dac=dac,        # 0 == VRef, 1 == DAC0
    	broadcast=0,
        channel=buffer,
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


msg = 'Enter the index of the stimulation electrode. Make sure the electrode is routed.'
electrode = int(g.enterbox(msg,title))

msg = 'Open the config file.'
default = '/home/user/Desktop/Data/'
configFile = g.fileopenbox(msg,title,default)[:-2]

today=datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day
Folder_name='-'.join(map(str,today))
outputDir = '/home/user/Desktop/Data/'+str(Folder_name)+'/'+str(MEA_ID)+'/Stimulation_'+str(electrode)

msg='The h5 files are recorded in the folder '+ str(outputDir)
g.msgbox(msg)



# Insert here the number of repetitions
number_of_repetitions = 1

# Insert here the time [ms] between the repetitions in ?        fre   q = 1/interTrainDelay; 200 = 5Hz, 100=10Hz, 5 =200Hz
interTrainDelay = 1000

# Insert here the amplitude of the cathodic part of the signal -- 1bit = 2.92 mV
#amp1 = [10,40,70,100,130,150]
amp1 = [3]

# Insert here the amplitude of the anodic part of the signal -- 1bit = 2.92 mV
#amp2 = [10,40,70,100,130,150]
amp2 = [3]

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


mea1k.go( mea1k.cmdReadout(
        s1Bypass=0,
        s1Gain=1,                   # 1 == x16   0 == x7
        s1RstMode=0,                # 0 == disconnect
        s2Bypass=0,
        s2Gain=5,                   # 5 == x16
        s3Bypass=0,
        s3Gain=0))                  # 0 ==x2

#default gain of 512 - if needed, change below
# 7x1x1(7)
# 7x8x2(112)
# 16x16x2(512)
# 16x16x4(1024)
# 16x32x4(2048)

#mea1k.go(mea1k.cmdVRefMosR(4095))

chip.loadHEX( configFile )

print 'Offset 1...'
import mea1kusr.api
Api = mea1kusr.api.Api()
Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
print 'Offset 1 done'

time.sleep(1)

chip.electrodeToStim(electrode)
stim_channel = chip.queryStimAtElectrode(electrode)
print 'stim_channel: '+str(stim_channel)

chip.download()
time.sleep(2)




save.openDir(outputDir)
save.mapping(chip.get_mapping( ))

for i in amp1:

    print 'Offset 2...'
    import mea1kusr.api
    Api = mea1kusr.api.Api()
    Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
    print 'Offset 2 done'

    c = mea1k.Config()
    c.add(switchOffAllChannels())
    c.add(switchOnChannel(stim_channel,1))
    c.add(mea1k.cmdDelaySamples(100))
    for u in range(number_of_repetitions):
        c.add(voltageBiPhasicPulse(512,i,i,phase1,phase2))
        c.add(mea1k.cmdDelaySamples(interTrainDelay))
    train_time = (number_of_repetitions*interTrainDelay)/10000.+0.3

    save.start('stimulation_el_{1:03}_amp_{0:03}_phase_{2:03}'.format(i,electrode,phase1))
    c.send()
    print 'start recording'
    time.sleep(train_time)
    save.stop()
    print 'stop recording'
    time.sleep(2)
