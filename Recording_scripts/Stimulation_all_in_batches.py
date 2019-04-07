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
from pandas import HDFStore,DataFrame,Series


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
outputDir = '/home/user/Desktop/Data/'+str(Folder_name)+'/'+str(MEA_ID)+'/Stimulation_all'

msg='The h5 files are recorded in the folder '+ str(outputDir)
g.msgbox(msg)
msg = 'Specify the directory of the configs.'
default = '/home/user/Desktop/Data'
path = g.diropenbox(msg,title,default)

# Insert here the number of repetitions
number_of_repetitions = 1

# Insert here the time [ms] between the repetitions in ?        fre   q = 1/interTrainDelay; 200 = 5Hz, 100=10Hz, 5 =200Hz
interTrainDelay = 30

# Insert here the amplitude of the cathodic part of the signal -- 1bit = 2.92 mV
amp1 = 5

# Insert here the amplitude of the anodic part of the signal -- 1bit = 2.92 mV
amp2 = 5

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

f = File(path+'/segmentation_logfile.h5', "r")
for i in range(len(f)):
	#Import the metadata of the configurations.
    routed_electrodes = np.asarray(f['structure_'+str(i+1)]['gotten_el']['values'][:])
    stim_list.append(routed_electrodes)
f.close()

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

mea1k.go(mea1k.cmdVRefMosR(1100))

hdf = HDFStore(outputDir+'/stimlist_logfile.h5')

for t,e in enumerate(stim_list):
    configFile = path+'/config_structure_'
    configFile += str(t+1)+'/structure_'+str(t+1)+'.hex.nrk'
    # Connect one electrode to a stimulation buffer
    chip.loadHEX( configFile )

    print 'Offset 1...'
    import mea1kusr.api
    Api = mea1kusr.api.Api()
    Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
    print 'Offset 1 done'

    time.sleep(1)

    iterator = range(len(e))
    random.shuffle(iterator)
    for batch in range(len(iterator)/30):
        stimChannels = []
        inds = iterator[batch*30:(batch+1)*30]
        for items in inds:
            chip.electrodeToStim(e[items])
            stimChannels.append(chip.queryStimAtElectrode(e[items]))
        hdf.put('Stim_el_struct_'+str(t+1)+'/batch_'+str(batch),Series(e[inds]))
        while -1 in stimChannels:
            stimChannels.remove(-1)

        print stimChannels

    	chip.download()
        time.sleep(2)

        print 'Offset 2...'
        import mea1kusr.api
        Api = mea1kusr.api.Api()
        Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
        print 'Offset 2 done'

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

        save.start('configstructure_{0:03}_batch_{1:03}'.format(t+1,batch))
        c.send()
        print 'start recording'
        time.sleep(train_time)
        save.stop()
        print 'stop recording'
        time.sleep(2)
hdf.close()
