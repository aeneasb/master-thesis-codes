import h5py
import os
import sys
import npgen

import numpy as np
import easygui as g
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import datetime
import cv2


class voltage_map:
    def __init__(self,path):
        self.path = path
        self.el_array = []
        self.thresh_array = []

    def testElectrodes(self):
    	import math
    	import mea1k
    	import libarray
    	import mea1kusr.init
    	import mea1kusr.save
        import mea1kusr.api

    	## Program an FPGA loop on VRefVari
    	def sineVari(t_period, n_samples, amp=5, periods=1):
            lc.stopLoop()
            lc.clear()
            lc.reset()
            lc.setStart() #Start the period'
            for i in range(0,n_samples):
                v = int(-amp*math.sin( periods*2*math.pi /n_samples * i)) #Amplitude of the sin-wave in bits
                s = int(20e3 * t_period/n_samples) #Time between the samples in bits. 1 bit = 50us = sampling freq.
            	#print i,v,s
            	lc.toLoop( mea1k.cmdVRefVari( 2048+v ) ) #Program v around 2048 bits. DAC has 12 bits
            			 #(=max. 4098 bits)
            	lc.toLoop( mea1k.cmdDelaySamples( s ))
            lc.setStop()
            lc.download() #Download the loop to the FPGA


    	configDir = "/opt/MaxLab/configs/28x6x6/"
        outputDir = self.path

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

    	'''PCB has 8 analog switches in total, 4xSW1 and 4xSW2. SW2 can be completely open (=0).
        For SW1 we want to connect pin NO1 to pin NO4 -> connect int. ref el to VRefVari
        (which are the DAC's). SW1_1/2/3/4 is 	 represented by four binary digits. Eg. 0001 (=1)
        closes first switch, eg. connects NO1 and COM1. 1000 	(=8) closes the fourth switch,
        eg. connects NO4 and COM4. In hexadecimal this is 9 (binary = 1001).'''
    	# 1. remove VrefIn from refEl
    	mea1k.go( mea1k.cmdSw2( 0 ) )
    	# 2. connect all refels to vrefvari
    	mea1k.go( mea1k.cmdSw1(  0x9 ) )

    	# 1 kHz w/ 20 samples (Gives exactly the sampling frequency of 20000/s) and 10 bit amplitude
    	sineVari( 0.001 , 20 , 10)

    	print 'Offset MEA around 512 bits...'
    	lc.startLoop() #Starts to apply the function to MEA
    	Api = mea1kusr.api.Api()
    	Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
    	lc.stopLoop()
    	print 'Offset done'

    	## Save && run scan
    	save = mea1kusr.save.save('localhost')
    	save.mkDir( outputDir )
    	save.reset( )

    	for cfg in range( 25 ):
    		print cfg
    		cfgFile = configDir +'{0:03}.hex.nrk'.format(cfg)
    		chip.loadHEX( cfgFile ) #Load the config file
    		chip.download() #Download the config to the chip
    		save.mapping(chip.get_mapping( ))
    		lc.startLoop()
    		time.sleep(0.3)
    		save.start('{0:03}'.format(cfg))
    		time.sleep(0.2)
    		save.stop()
    		time.sleep(0.3)
    		lc.stopLoop()

    def import_data(self):

        #Open all H5 datasets
        try:
            h5paths = sorted([os.path.join(self.path,f) for f in os.listdir(self.path) if f.endswith('.raw.h5')])
            f = [h5py.File(i, "r") for i in h5paths]
        except:
            sys.exit('H5 dataset could not be imported.')

        #Figure out all electrode numbers
        electrode_info = [np.asarray(i['mapping']['channel','electrode']) for i in f]
        mask = [i['electrode']!=-1 for i in electrode_info]
        clean_abs_inds = [i[0]['electrode'][i[1]] for i in zip(electrode_info,mask)]
        clean_rel_inds = [i[0]['channel'][i[1]] for i in zip(electrode_info,mask)]

        #For each recording figure out the x and y coordinates per electrode
        x_clean=[v%220 for v in clean_abs_inds]
        y_clean=[v/220 for v in clean_abs_inds]

        cut_traces = []
        for i,v in enumerate(clean_rel_inds):
            cut_traces.append(np.asarray(f[i]['sig'])[v,900:1000])

        cut_traces_max=np.asarray([np.amax(i,axis=1) for i in cut_traces])
        cut_traces_min=np.asarray([np.amin(i,axis=1) for i in cut_traces])
        cut_traces_amp=cut_traces_max-cut_traces_min

        #For each recordig build the elctrode array for visualization
        el_array = np.zeros((120,220))
        for i,j in enumerate(cut_traces_amp):
            el_array[y_clean[i],x_clean[i]]=j

        return el_array


    def Threshimage(self,thresh):
    	el_array_8bit=(self.el_array/(self.el_array.max()/255.0)).astype('uint8')
    	thr, dst=cv2.threshold(el_array_8bit,thresh,255,cv2.THRESH_BINARY)
        return dst

    def saveFig(self,ch):
    	pix=len(np.where(self.thresh_array==200)[0])
    	fig=plt.figure(figsize=(18,9))
    	ax=fig.add_subplot(111)
    	ax.imshow(self.thresh_array[:,:-10],aspect='auto',cmap='hot',interpolation='none')
        ax.set_axis_off()
    	plt.title(str(pix)+' electrodes annotated')
    	plt.savefig(self.newpath+'/structure_'+str(ch)+'.png',bbox_inches='tight',pad_inches=0)
    	plt.close()

    def FloodThresh(self,ch,flags):
        #Change matplotlib backend on the fly
        matplotlib.use('TkAgg',warn=False,force=True)
        reload(plt)
        #Select the region to be floodfilled
    	fig=plt.figure(figsize=(19,10))
        ax =fig.add_subplot(111)
    	ax.imshow(self.thresh_array[:,:-10],'hot',interpolation='none')
        ax.set_axis_off()
    	plt.show(block=False)
    	points = np.around(np.asarray(plt.ginput(n=1,show_clicks=True)))
    	plt.close()
        matplotlib.use('Agg',warn=False,force=True)
        reload(plt)

    	h,w=self.thresh_array.shape[:2]
    	mask=np.zeros((h+2,w+2),np.uint8)
    	cv2.floodFill(self.thresh_array,mask,seedPoint=(int(points[0][0]),int(points[0][1])),newVal=200,flags=flags)
        self.saveFig(ch)

    def manual_annot(self,ch,param):
        matplotlib.use('TkAgg',warn=False,force=True)
        reload(plt)
        fig=plt.figure(figsize=(25,13))
        ax =fig.add_subplot(111)
        ax.imshow(self.thresh_array[:,:-10],'hot',interpolation='none')
        ax.set_axis_off()
    	plt.title('left click: add point | right click: remove last point | middle click: stop input')
    	plt.show(block=False)
    	points = np.around(np.asarray(plt.ginput(n=200,show_clicks=True,mouse_add=1,mouse_pop=3,mouse_stop=2)))
    	plt.close()
        matplotlib.use('Agg',warn=False,force=True)
        reload(plt)
        xses=map(int,[t[0] for t in points])
    	yses=map(int,[t[1] for t in points])
        if param == 'add':
            self.thresh_array[yses,xses]=200
        elif param == 'deselect':
            self.thresh_array[yses,xses]=0

'''----------------------------------------------------------------------------------------------'''

##Decision tree

title = 'Checkpoint 1'
msg = 'Scanning: Locate PDMS | Importing: Import existing scan results'
choices = ['Scanning','Importing']
reply = g.buttonbox(msg,title,choices=choices)


if reply =='Importing':
	title = 'Checkpoint 2'
	msg = 'Specify the directory of the scan-result:'
        default = '/home/user/Desktop/Data'
        path = g.diropenbox(msg,title,default)
        vm = voltage_map(path)

else:
	title = 'Checkpoint 1.1'
	msg = 'Enter the MEA-ID and mount the MEA on the recording unit:'
	MEA_ID = g.enterbox(msg,title)
	today=datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day
	date='-'.join(map(str,today))
	outputDir= '/home/user/Desktop/Data/'+str(date)+'/'+str(MEA_ID)+'/voltage_map'
	msg='The result of the scan is saved in the folder '+ outputDir
	g.msgbox(msg)

        vm = voltage_map(outputDir)
        vm.testElectrodes()

vm.el_array = vm.import_data()

fig=plt.figure(figsize=(18,9))
ax=fig.add_axes([0,0,0.95,1])
#ax=fig.add_subplot(111)
i = ax.imshow(vm.el_array[:,:-10],aspect='auto',cmap='hot',interpolation='none')
colorbar_ax = fig.add_axes([0.95,0,0.05,1])#Place colorbar next to el_array
ax.set_axis_off() #Disable the axis labels
fig.colorbar(i,cax=colorbar_ax)
plt.savefig(vm.path+'/voltage_map.png',bbox_inches='tight',pad_inches=0)
plt.close()

image = vm.path+'/voltage_map.png'
title = 'Checkpoint 2'
msg = 'Proceed with image segmentation?'
if g.ynbox(msg,title,image=image):
	try:
            newpath = vm.path+'/segmentation'
            vm.newpath = newpath
            os.makedirs(newpath)

	except:
    		pass
else:
    sys.exit(0)

th=True
thresh=170
while th:
	vm.thresh_array = vm.Threshimage(thresh)
        fig=plt.figure(figsize=(19,10))
        ax=fig.add_subplot(111)
        ax.imshow(vm.thresh_array[:,:-10],'hot',interpolation='none')
        ax.set_axis_off()
        plt.title('Threshold: '+str(int(thresh)))
        plt.savefig(vm.newpath+'/thresh_'+str(int(thresh))+'.png',bbox_inches='tight',pad_inches=0)
        plt.close()

	image = vm.newpath+'/thresh_'+str(int(thresh))+'.png'
	title = 'Checkpoint 3'
	msg = 'New thresh.: Change threshold | Annotation: Proceed'
        choices = ['New thresh.','Annotation']
        reply = g.buttonbox(msg,title,choices=choices,image=image)

	if reply == 'Annotation':
		th=False
	else:
		title = 'Change Threshold'
		msg = 'Enter a new threshold value from [0,255].'
		thresh = float(g.enterbox(msg,title))
		pass

#hdf = HDFStore(vm.newpath+'/segmentation_logfile.h5')
hdf = h5py.File(vm.newpath+'/segmentation_logfile.h5','w')
ch = 1 #Number of routed/cropped structure
vm.saveFig(ch)
flood=True
while flood==True:
	image = vm.newpath+'/structure_'+str(ch)+'.png'
	title = 'Checkpoint 4'
	msg = '"auto annotation" to choose more pixels. "next strucure" to segment a new structure. "flags=8" considers on 4 neighbouring pixels during automatic annotation. "manual annotation" to select individual pixels. "deselect" to deselect electrodes. "export" to export the current cropping into a config file'
	choices=['auto annot.','next struct.','flags=8','deselect','manual annot.','export','exit']
	reply = g.buttonbox(msg,title,image=image,choices=choices)

	if reply=='auto annot.':
		vm.FloodThresh(ch,4)
		pass
        if reply=='flags=8':
                vm.FloodThresh(ch,8)
                pass
	elif reply=='next struct.':
            	elnum = (np.where(vm.thresh_array==200)[0]*220)+np.where(vm.thresh_array==200)[1]
                if elnum.size:
                    ch+=1
                    vm.thresh_array[np.where(vm.thresh_array==200)]=100 #Change color of already cropped electrodes
                    vm.saveFig(ch)
		pass
	elif reply=='manual annot.':
            vm.manual_annot(ch,'add')
            vm.saveFig(ch)
            pass
	elif reply=='deselect':
            vm.manual_annot(ch,'deselect')
            vm.saveFig(ch)
            pass
	elif reply=='export':
            elnum = np.where(vm.thresh_array==200)[0]*220+np.where(vm.thresh_array==200)[1]
            hdf['structure_'+str(ch)+'/wanted_el']=np.array(elnum)
            config_directory = vm.newpath+'/config_structure_'+str(ch)
            config_name = 'structure_'+str(ch)
            npgen.mkdir_p(config_directory)
            #Create a config file
            npgen.makeConfig( os.path.join(config_directory,config_name), [] , elnum, True )
            #Open the generated routing to check which electrodes have be actually routed
            routed_mapping = open(os.path.join(config_directory,config_name+'.mapping.nrk'),'r')
            lines = routed_mapping.readlines()
            coords = [[int(j) for j in i.split()[1:3] if int(j)!=-1] for i in lines]
            clean_coords = [i for i in coords if i]
            el_ID = [i[0]-10+(i[1]-9)*220 for i in clean_coords] #coordinate system of hex files is shifted with x=10,y=9
            #Store the elnumbers in the logfile
            hdf['structure_'+str(ch)+'/gotten_el']=np.array(el_ID)
            missed_el = list(set(elnum.tolist())-set(el_ID))

            hdf['structure_'+str(ch)+'/missed_el']=np.array(missed_el)
            routed_mapping.close()
            pass
	elif reply=='exit':
		flood=False
		hdf.close()
		pass
