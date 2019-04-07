#!/usr/bin/python
## Purpose of the script
#  Test how well each electrode works
#  
#  Procedure:
#  Apply a sinewave to the int.ref.el. and measure the amplitude on
#  all electrodes by scanning through different random configs

import math
import time
import datetime
import sys
import datetime

import mea1k
import libarray

import mea1kusr.init
import mea1kusr.save

chip_name   = str(raw_input('Enter the MEA ID (eg. 1724).:  '))



if len(sys.argv) > 1:
    chip_name = sys.argv[1]

print chip_name 

exit

today=datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day
Folder_name='-'.join(map(str,today))

configDir = '/opt/MaxLab/configs/28x6x6/'
intermediate_data = '/home/user/Desktop/Data/'+Folder_name+'/'

#date_string = datetime.datetime.now().strftime('%y%m%d')
outputDir = intermediate_data + chip_name + '/PDMS_loc'

print outputDir

## Basic setup of chip & board

# default initialization
mea1kusr.init.board()
mea1kusr.init.chip() # Corresponds to initialize on measerver

chip = libarray.Chip() # Create Chip variable
lc   = mea1k.Loop() # Create loop variable (execute fast commands)

# reduce amplifier gain (ADC), amplifier have three stages: s1, s2, s3
#This amplifier is tuned to enhance neuronal signals (linear in uV-range). Amplifier is not
#optimal to enhance 1 kHz-stimulus in mV range (but still usable).
mea1k.go( mea1k.cmdReadout(
    s1Bypass=0,			# Bypass amplifier if needed
    s1Gain=0,                   # 0 == x7 (=7-fache Verstaerkung)
    s1RstMode=0,                # 0 == disconnect amplifier
    s2Bypass=0,
    s2Gain=2,                   # 2 == x2 (=14-fache Verstaerkung)
    s3Bypass=0,
    s3Gain=0                    # 0 ==x2
))

'''PCB has 8 analog switches in total, 4xSW1 and 4xSW2. SW2 can be completely open (=0). For SW1 we want to connect pin NO1 to pin NO4 -> connect int. ref el to VRefVari (which are the DAC's). SW1_1/2/3/4 is represented by four binary digits. Eg. 0001 (=1) closes first switch, eg. connects NO1 and COM1. 1000 (=9) closes the fourth switch, eg. connects NO4 and COM4. In hexadecimal this is 9 (binary = 1001).'''  
# 1. remove VrefIn from refEl
mea1k.go( mea1k.cmdSw2( 0 ) )
# 2. connect all refels to vrefvari
mea1k.go( mea1k.cmdSw1(  0x9 ) )


## Program an FPGA loop on VRefVari
def sineVari(t_period, n_samples, amp=10, periods=1):
    global lc
    lc.stopLoop()
    lc.clear()
    lc.reset()
    lc.setStart() #Start the period
    for i in range(0,n_samples):
        v = int(-amp*math.sin( periods*2*math.pi /n_samples * i)) #Amplitude of the sin-wave in bits
        s = int(20e3 * t_period/n_samples) #Time between the samples in bits. 1 bit = 50us = sampling freq.
        print i,v,s
        lc.toLoop( mea1k.cmdVRefVari( 2048+v ) ) #Program v around 2048 bits. DAC has 12 bits
						 #(=max. 4098 bits)
        lc.toLoop( mea1k.cmdDelaySamples( s ))
    lc.setStop()
    lc.download() #Download the loop to the FPGA

# 1 kHz w/ 20 samples (Gives exactly the sampling frequency of 20000/s) and 10 bit amplitude
sineVari( 0.001 , 20 , 8)
    
# 25 == 40mV  (i.e. +-25) peak-to-peak
# 50 == 80mV
# 12 == ??mV   peak to peak

lc.startLoop() #Starts to apply the function to MEA
import mea1kusr.api
Api = mea1kusr.api.Api()
Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
lc.stopLoop()


## Save && run scan
save = mea1kusr.save.save('localhost')
save.mkDir( outputDir ) 
save.reset( ) 

##
for cfg in range( 26 ):
    print cfg
    cfgFile = configDir +'{0:03}.hex.nrk'.format(cfg)
    chip.loadHEX( cfgFile ) #Load the config file
    chip.download() #Download the config to the chip
    save.mapping(chip.get_mapping( ))
    time.sleep(1.2)
    lc.startLoop()
    time.sleep(1.8)
    save.start('{0:03}'.format(cfg))
    time.sleep(0.2)
    save.stop()
    time.sleep(1)
    lc.stopLoop()

