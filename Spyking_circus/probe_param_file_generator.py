##This script opens an existing .h5 recording and outputs an .prb file

import os

from h5py import File
import numpy as np

import easygui as g

from shutil import copyfile


title = 'Probe-file generator'
msg = 'Open .h5 file'
default = '/home/user/Desktop/Data/'
path = g.fileopenbox(msg,title,default)

# In[3]:


u = File(path,'r')

 
# In[142]:


electrode_info = np.asarray(u['mapping']['channel','electrode'])
mask = electrode_info['electrode']!=-1
clean_rel_inds = electrode_info['channel'][mask]


# In[143]:


#The cordinates in um have an offset
x_cors = u['mapping']['x']-175
y_cors = u['mapping']['y']-157


# In[11]:


#Alternatively calculation of the coordinates
#x = (electrode_info['electrode']%220)*17.5 #To get same result as above: np.floor()
#y = (electrode_info['electrode']/220)*17.5 #To get same result as above: np.ceil()


# In[144]:


channel_groups    = {1 : {}}
channel_groups[1]["geometry"] = {i:[x_cors[i],y_cors[i]] for i in list(range(1024))}


# In[145]:

MEA_ID = [s for s in path.split('/') if s.isdigit()][0]

sp_c_path = path.split('/')[:-1]
sp_c_path.append('Spyking_circus_results_ID_'+MEA_ID)
sp_c_path.append(path.split('/')[-1].strip('.raw.h5'))

outputdir = '/'.join(sp_c_path)

try:
        os.makedirs(outputdir)
except:
        pass

# Create a probefile anf open in write mode

probefile = open(outputdir+'/probe.prb','w')

probefile.write('total_nb_channels = 1024\nradius    = 250\n')
probefile.write('channel_groups    = {1 : {}}\n\n')
probefile.write('channels = '+str(clean_rel_inds.tolist())+'\n\n')
probefile.write('channel_groups[1]["channels"] = channels\nchannel_groups[1]["graph"]    = []\n\n')
probefile.write('channel_groups[1]["geometry"] = '+str(channel_groups[1]["geometry"]))
probefile.close()


#Open a default parameter file
param = open('/home/user/Desktop/Data/Spyking-Circus/default_parameter_file.params','r')
parameter_string = ''.join(param.readlines())
param.close()

parameter_file = open(outputdir+'/'+path.split('/')[-1][:-3]+'.params','w')
parameter_file.write(parameter_string)
parameter_file.close()

#Make a copy of the original .h5 file
copyfile(path,outputdir+'/'+path.split('/')[-1])

print 'Probefile, paramenter_file and a copy of the data saved under '+outputdir
