'''
This script is intended to generate one config file cotaining routings in each
chamber + algorithm to find the optimal stimulation electrode.

'''


import os
import sys

import npgen
import mea1k
import libarray
import mea1kusr.init
import mea1kusr.save as stst
import mea1kusr.api

import numpy as np
import easygui as g
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import datetime
import cv2
from PIL import Image, ImageDraw
import h5py


class voltage_map:
    def __init__(self,path):
        self.path = path
        self.el_array = []
        self.thresh_array = []

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
    	fig=plt.figure(figsize=(19,10))
    	ax=fig.add_subplot(111)
    	ax.imshow(self.thresh_array[:,:-10],'hot',interpolation='none')
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

    def draw_ellipse(self,ch):
        matplotlib.use('TkAgg',warn=False,force=True)
        reload(plt)
        fig=plt.figure(figsize=(25,13))
        ax =fig.add_subplot(111)
        ax.imshow(self.thresh_array[:,:-10],'hot',interpolation='none')
        ax.set_axis_off()
    	plt.title('Choose two boundary points to confine the ellipse. left click: add point | right click: remove last point')
    	plt.show(block=False)
    	points = np.around(np.asarray(plt.ginput(n=2,show_clicks=True,mouse_add=1,mouse_pop=3)))
    	plt.close()
        matplotlib.use('Agg',warn=False,force=True)
        reload(plt)
        xses=map(int,[t[0] for t in points])
    	yses=map(int,[t[1] for t in points])

        coords_ellipse = [i for i in zip(xses,yses)]
        img = Image.fromarray(self.thresh_array[:,:])
        draw = ImageDraw.Draw(img)
        draw.ellipse(coords_ellipse, fill=200)
        self.thresh_array = np.array(img)
        img.close()

    def export(self,hdf):
        elnum = np.where(self.thresh_array==100)[0]*220+np.where(self.thresh_array==100)[1]
        config_directory = self.newpath+'/config_structure'
        config_name = 'circles'
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
        for key,values in hdf.iteritems():
            hdf[key+'/gotten_el']=np.array([i for i in values['wanted_el'] if i in el_ID])
            hdf[key+'/missed_el']=np.array([i for i in values['wanted_el'] if i not in el_ID])
        routed_mapping.close()
        hdf.close()

class find_stim_el:
    def __init__(self,path):
        self.circle_path = path

    def get_neighbor(self,center_el):
        box_size = 3
        y = center_el/220
        x = center_el%220
        side_x = np.arange(x-1,x+2)
        side_y = np.arange(y-1,y+2)
        xx,yy = np.meshgrid(side_x,side_y)
        el_grid = xx+yy*220
        return el_grid

    def switchOffAllChannels(self):
        c = mea1k.Config()
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

    def switchOnChannels(self,channels, dac):
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

    def voltageBiPhasicPulse(self,offset, ampBits_1, ampBits_2, samples_1=4, samples_2=4):
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


    def stimulation(self,circle_path,outputDir):


        #Paramters of Stimulation
        number_of_repetitions = 100
        interTrainDelay = 20000 # 10000 = 0.5s
        bin_length = 1000 #Number of datapoints to crop after stimulus
        amp1 = 30
        amp2 = 30
        phase1 = 4
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

        #mea1k.go(mea1k.cmdVRefMosR(1100)) #gives stable traces but many out of bounds after stim

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


        #Import the information about the stimulation electrodes
        el_list = h5py.File(circle_path+'/segmentation_logfile.h5','r')
        routed_el = [values['gotten_el'][:] for key,values in el_list.iteritems()]
        el_list.close()

        logfile = h5py.File(outputDir+'/stimlist_stim_config_.hdf5','w')
        lengths = [i.size for i in routed_el]

        #Start iterating over each electrode
        print '{0:02} iterations to be done.'.format(max(lengths))
        for i in range(max(lengths)):
            configFile = circle_path+'/config_structure/circles.hex.nrk'
            chip.loadHEX( configFile )
            print 'configFile loaded...'

            print 'Offset MEA around 512 bits...'
            Api = mea1kusr.api.Api()
            Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
            print 'Offset done'

            stimlist = []
            neighbors = []
            for j in routed_el:
                if i<j.size:
                    stimlist.append(j[i])
                    putative_neighbors = self.get_neighbor(j[i]).flatten().tolist()
                    neighbors.append(putative_neighbors)
                    fake_neighbors = [m for m in putative_neighbors if m not in j]
                    if fake_neighbors:
                        for n in fake_neighbors:
                            neighbors[-1].remove(n) #Remove fake neighbors from the last list

            # Save the stimulation electrodes as attributes
            stim_group = logfile.create_group('stim_config_'+str(i)+'/stimlist')
            for k,p in enumerate(stimlist):
                stim_group.attrs.create('channel_'+str(k),p)

            neigh_group = logfile['stim_config_'+str(i)].create_group('neighbors')
            for k,p in enumerate(neighbors):
                neigh_group.attrs.create('channel_'+str(k),p)

            stimChannels = []
            for k in stimlist:
                chip.electrodeToStim(k)
                stimChannels.append(chip.queryStimAtElectrode(k))

        	chip.download()
            time.sleep(2)
            print 'El. \tbuffer'
            for p,k in enumerate(stimChannels):
                print stimlist[p], '\t',k

            while -1 in stimChannels:
                stimChannels.remove(-1)

            print 'Offset MEA around 512 bits...'
            Api = mea1kusr.api.Api()
            Api.binary_offset( 512 ); #Offset the electrodes around 512 bits.
            print 'Offset done'

            save.openDir(outputDir)
            save.mapping(chip.get_mapping( ))

            c = mea1k.Config()
            c.add(self.switchOffAllChannels())
            c.add(self.switchOnChannels(stimChannels,1))
            c.add(mea1k.cmdDelaySamples(100))


            for l in range(number_of_repetitions):
                c.add(mea1k.cmdDelaySamples(interTrainDelay-(phase1+phase2)))#-8 to account for the stimulus length
                c.add(self.voltageBiPhasicPulse(512,amp1,amp2,phase1,phase2))
            train_time = (number_of_repetitions*interTrainDelay)/20000.

            save.start('raw_stim_config_'+str(i))
            c.send()
            time.sleep(train_time+2)
            save.stop()
            time.sleep(3)

            raw_file = h5py.File(outputDir+'/raw_stim_config_'+str(i)+'.raw.h5','r')
            DAC = raw_file['sig'][1024,:]
            over_threshold_islets = np.where(DAC>525)[0]
            stim_edges=over_threshold_islets[np.hstack((0,np.asarray([1+el for el in np.where(np.diff(over_threshold_islets)!=1)[0]])))]

            index = np.asarray([np.arange(0,bin_length)+c for c in stim_edges])#+8 to make sure to cut out the stimulus
            seconds = index/20000.

            el_indices = []
            missing = {}
            for keys,values in logfile['stim_config_'+str(i)+'/neighbors'].attrs.iteritems():
                missing[keys]=[]
                for it in values[:]:
                    try:#Some neighbors may not be routed, although they should be
                        el_indices.append(np.where(raw_file['mapping']['electrode']==it)[0][0])
                    except:
                        missing[keys].append(it)

            print 'miss.el. keys \t values'
            for keys,values in missing.iteritems():
                if values:
                    print '\t', keys, values
                    neighbors = list(logfile['stim_config_'+str(i)+'/neighbors'].attrs[keys])
                    del logfile['stim_config_'+str(i)+'/neighbors'].attrs[keys]
                    for val in values:
                        neighbors.remove(val)
                    logfile['stim_config_'+str(i)+'/neighbors'].attrs.modify(keys,neighbors)

            trace = np.empty((1028,bin_length*number_of_repetitions))
            raw_file['sig'].read_direct(trace,source_sel=np.s_[:,index.flatten()])
            f = h5py.File(outputDir+'/stim_config_'+str(i)+'.h5','w')
            cropped_trace = trace[sorted(el_indices)]
            f.create_dataset('sig',data=cropped_trace)
            f['mapping']=raw_file['mapping'][sorted(el_indices)]
            f['proc0/spikeTimes']=raw_file['proc0/spikeTimes'][:]
            f['time']=seconds
            f['settings']=raw_file['settings/gain'][:]
            f['version']=raw_file['version'][:]

            raw_file.close()
            os.remove(outputDir+'/raw_stim_config_'+str(i)+'.raw.h5')
            f.close()
            print 'Iteration and postprocessing no. {0:03}/{1:01} done.'.format(i,max(lengths))

        logfile.close()

'''----------------------------------------------------------------------------------------------'''

##Decision tree

title = 'Checkpoint 1'
msg = 'Stimulation: Import existing configs and stimulate | Annotation: Generate configs'
choices = ['Stimulation','Annotation']
reply = g.buttonbox(msg,title,choices=choices)

if reply =='Stimulation':
    title = 'Checkpoint 2'
    msg = 'Specify the directory of the circle_segmentation:'
    default = '/home/user/Desktop/Data'
    circle_path = g.diropenbox(msg,title,default)

    title = 'User specifications'
    msg = 'Enter the MEA-ID and mount the MEA on the recording unit:'
    MEA_ID = g.enterbox(msg,title)

    today=datetime.datetime.now().year,datetime.datetime.now().month,datetime.datetime.now().day
    Folder_name='-'.join(map(str,today))
    outputDir = '/home/user/Desktop/Data/'+str(Folder_name)+'/'+str(MEA_ID)+'/stimulation'
    msg='The h5 files are recorded in the folder '+ str(outputDir)
    g.msgbox(msg)

    vm = find_stim_el(circle_path)
    vm.stimulation(circle_path,outputDir)
    sys.exit(0)
else:
	pass

title = 'Checkpoint 2'
msg = 'Specify the directory of the voltage map:'
default = '/home/user/Desktop/Data'
path = g.diropenbox(msg,title,default)
vm = voltage_map(path)

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
title = 'Checkpoint 3'
msg = 'Proceed with image segmentation?'
if g.ynbox(msg,title,image=image):
	try:
            newpath = vm.path+'/circle_segmentation'
            vm.newpath = newpath
            os.makedirs(newpath)

	except:
    		pass
else:
    sys.exit(0)

'''-------------------------------------------------------------------------'''

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

'''-------------------------------------------------------------------------'''
hdf = h5py.File(vm.newpath+'/segmentation_logfile.h5','w')
ch = 1 #Number of routed/cropped structure
vm.saveFig(ch)
flood=True
while flood==True:
    image = vm.newpath+'/structure_'+str(ch)+'.png'
    title = 'Checkpoint 4'
    msg = '"auto annotation" to choose more pixels. "next strucure" to segment a new structure. "flags=8" considers on 4 neighbouring pixels during automatic annotation. "manual annotation" to select individual pixels. "deselect" to deselect electrodes. "export" to export the current cropping into a config file'
    choices=['auto annot.','next struct.','deselect','manual annot.','draw ellipse','export and exit']
    reply = g.buttonbox(msg,title,image=image,choices=choices)
    if reply=='auto annot.':
        vm.FloodThresh(ch,4)
        pass
    elif reply=='next struct.':
        elnum = (np.where(vm.thresh_array==200)[0]*220)+np.where(vm.thresh_array==200)[1]
        if elnum.size:
            hdf['structure_'+str(ch)+'/wanted_el']=np.array(elnum)
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
    elif reply=='draw ellipse':
        vm.draw_ellipse(ch)
        vm.saveFig(ch)
        pass
    elif reply=='export and exit':
        vm.export(hdf)
        flood=False
        pass
