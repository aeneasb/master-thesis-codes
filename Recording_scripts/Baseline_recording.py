## Baseline Recording Script ##

import sys

import mea1k
import mea1kusr.init
import mea1kusr.save as stst
import libarray

import easygui as g
import datetime
import time


title = 'User specifications'
msg = 'Enter the MEA-ID and mount the MEA on the recording unit:'
MEA_ID = g.enterbox(msg,title)

today=datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day
Folder_name='-'.join(map(str,today))
outputDir = '/home/user/Desktop/Data/'+str(Folder_name)+'/'+str(MEA_ID)+'/Baseline_recording'
msg='The h5 files are recorded in the folder '+ str(outputDir)
g.msgbox(msg)

msg = 'Enter the duration of the recording in seconds'
duration = int(g.enterbox(msg,title))

msg = 'Open the config file.'
default = '/home/user/Desktop/Data/'
configFile = g.fileopenbox(msg,title,default)[:-2]

#Set gain back to normal operation conditions
mea1kusr.init.board();
mea1kusr.init.chip();

print 'make chip'
chip = libarray.Chip();
print 'done'

#Default gain of 512
mea1k.go( mea1k.cmdReadout(
        s1Bypass=0,
        s1Gain=1,                   # 1 == x16   0 == x7
        s1RstMode=0,                # 0 == disconnect
        s2Bypass=0,
        s2Gain=5,                   # 5 == x16
        s3Bypass=0,
        s3Gain=0))                  # 0 ==x2

save = stst.save('localhost')
try:
    os.makedirs(outputDir)
except:
    pass
save.reset()

chip.loadHEX(configFile)
chip.download()
time.sleep(2)

print 'Offset MEA around 512 bits...'
import mea1kusr.api
Api = mea1kusr.api.Api()
Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
print 'Offset done'

save.openDir(outputDir)
save.mapping(chip.get_mapping())

save.start('Baseline_recording_{0:03}_sec_'.format(duration)+configFile.split('/')[-2])
time.sleep(duration)
save.stop()
time.sleep(1)
print 'Recording finished'
