# coding: utf-8

# In[4]:


import os
import sys
from h5py import File
from collections import OrderedDict
import numpy as np
from ipywidgets import FloatProgress
from IPython.display import display


class import_metadata:
    '''
    An instance of this class takes a given directory containg multiple .h5 files
    or also a single .h5 file to extract the metadata. The following attributes are
    generated:
    ['clean_abs_indeces']: IDs of all recording electrodes
    ['electrode_map']: 120*220 electrode grid showing all recording electrodes
    ['x_clean']/['y_clean]: Spatial location of every electrode in units of electrodes
    ['SpikeTimes]: Spiketimes detected by the MEA-software
    Upon running import_rawdata:
    ['time']: A list containing time bins in unit of micro-seconds
    ['DAC']: Trace of the DAC channel
    '''
    def __init__(self, path):
        self.dict = self.fetch_dict(path)
        self.metadata = self.extract_meta()

    def fetch_dict(self,path):
        if path == '':
                sys.exit("The path is empty.")
        elif os.path.isfile(path):
            try:
                f=File(path,'r')
            except:
                sys.exit("The path doesn't specify an .h5-file.")
        elif os.path.isdir(path):
            try:
                h5paths = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.h5')]
                #Best solution is to create a list which contains all h5 dicts
                f = [File(i, "r") for i in h5paths]
            except:
                sys.exit('Some .h5-files may be open in another notebook.')
        return f

    def extract_meta(self):

        def merge_five_dicts(x,y,q,t,l):
            z=x.copy()
            z.update(y)
            z.update(q)
            z.update(t)
            z.update(l)
            return z

        def merge_two_dicts(x,y):
            z=x.copy()
            z.update(y)
            return z

        if isinstance(self.dict,File):
            if list(self.dict['mapping']): #Check if mapping is not empty
                f={}
                el_indices = np.asarray(self.dict['mapping']['electrode'])

                mask = el_indices!=-1
                f['clean_abs_inds']=el_indices[mask]

                f['x_clean']=f['clean_abs_inds']%220
                f['y_clean']=f['clean_abs_inds']/220

                el_array=np.zeros((120,220))
                el_array[f['y_clean'],f['x_clean']]=1
                f['electrode_map']=el_array

                #The channel-values in the key spikeTimes saves the channel number, but not the absolute
                #electrode index, which is necessary for the analysis. Zip the absolute indices and the
                #Corresponding time events into a dictionary.
                try:
                    spikeinfo = self.dict['proc0']['spikeTimes']['channel','frameno']
                    clean_abs_spike_ind = np.asarray([el_indices[i] for i in spikeinfo['channel'] if el_indices[i]!=-1])
                    clean_frameno = np.asarray([i[1] for i in spikeinfo if el_indices[i[0]]!=-1])
                    f['SpikeTimes']={}
                    f['SpikeTimes']['electrode_ind'] = clean_abs_spike_ind
                    f['SpikeTimes']['framenumber'] = clean_frameno
                except:
                    f['SpikeTimes']={}
                    f['SpikeTimes']['No_spikes_detected']=[]

            else:
                sys.exit('H5-file: No values in key ["mapping"]')

        if isinstance(self.dict,list):
            #Figure out all electrode numbers
            el_indices = [np.asarray(i['mapping']['electrode']) for i in self.dict]
            mask = [i!=-1 for i in el_indices]
            clean_abs_inds = [{'clean_abs_inds': i[0][i[1]]} for i in zip(el_indices,mask)]

            #Avoid empty mappings. Make sure there are none of them.
            for i in clean_abs_inds:
                if not i['clean_abs_inds'].size:
                    sys.exit('H5-file: No values in key ["mapping"]')

            #For each recording figure out the x and y coordinates per electrode
            x_clean=[{'x_clean':v['clean_abs_inds']%220} for v in clean_abs_inds]
            y_clean=[{'y_clean':v['clean_abs_inds']/220} for v in clean_abs_inds]

            #For each recordig build the elctrode array for visualization
            empt_arrays = [np.zeros((120,220),dtype=int) for _ in range(len(clean_abs_inds))]
            for i,j in enumerate(empt_arrays):
                j[y_clean[i]['y_clean'],x_clean[i]['x_clean']]=1

            el_array=[{'electrode_map':i} for i in empt_arrays]

            try:
                spikeinfo = [i['proc0']['spikeTimes']['channel','frameno'] for i in self.dict]
                clean_abs_spike_ind = [{'electrode_ind': np.asarray([j[0][i] for i in j[1]['channel'] if j[0][i]!=-1])} for j in zip(el_indices,spikeinfo)]
                clean_frameno = [{'framenumber': np.asarray([i[1] for i in j[1] if j[0][i[0]]!=-1])} for j in zip(el_indices,spikeinfo)]
                merged = [merge_two_dicts(q,k) for k,q in zip(clean_abs_spike_ind,clean_frameno)]
                SpikeTimes=[{'SpikeTimes': v} for v in merged]
            except:
                fakelist = np.zeros([len(self.dict)]).tolist()
                SpikeTimes=[{'NoSpikes': v} for v in fakelist]

            #One recording has five dicts to be merged.
            f = [merge_five_dicts(q,k,i,t,l) for i,k,q,t,l in zip(y_clean, x_clean, clean_abs_inds,el_array,SpikeTimes)]

        return f


class import_rawdata(import_metadata):
    '''
    An instance of this class takes a given directory containg multiple .h5 files
    or also a single .h5 file to load the data into working memory. The data is
    saved as instance.raw_data()
    '''
    def __init__(self,path,start,stop,modified):
        import_metadata.__init__(self,path)
        self.raw_data = self.read_in(path,start,stop,modified)

    def read_in(self,path,start,stop,modified):

        def array_slice(self,path,start,stop,modified):
            #Convert the (whole!) h5 dataset to an numpy array and slice immediately.
            #Check if input is only one h5 file or multiple
            if isinstance(self.dict,File):
                electrode_info = np.asarray(self.dict['mapping']['channel','electrode'])
                mask = electrode_info['electrode']!=-1
                clean_rel_inds = electrode_info['channel'][mask]

                if modified == True:
                    stop = 'end'
                    self.metadata['modified'] = True
                else:
                    self.metadata['modified'] = False

                if stop=='end':
                    if modified==False:
                        traces = np.asarray(self.dict['sig'])
                        self.metadata['DAC'] = traces[1024,start:]
                        traces = traces[clean_rel_inds,start:]#crop the traces
                        self.metadata['time'] = np.arange(0, (self.dict['sig'].shape[1]-start)/20000., 1/20000.)
                    else:
                        traces = np.asarray(self.dict['sig'])[:,start:]
                        self.metadata['time'] = self.dict['time'][:]
                else:
                    traces = np.asarray(self.dict['sig'])
                    self.metadata['DAC'] = traces[1024,start:stop]
                    traces = traces[clean_rel_inds,start:stop]
                    self.metadata['time'] = np.arange(0, (stop-start)/20000., 1/20000.)


            if isinstance(self.dict,list):
                electrode_info = [np.asarray(i['mapping']['channel','electrode']) for i in self.dict]
                mask = [i['electrode']!=-1 for i in electrode_info]
                clean_rel_inds = [i[0]['channel'][i[1]] for i in zip(electrode_info,mask)]

                traces=[]

                progr = FloatProgress(min=0, max=len(self.dict),description='Importing...',bar_style='success')
                display(progr)

                if modified == True:
                    stop = 'end'
                    for i in range(len(clean_rel_inds)):
                        self.metadata[i]['modified'] = True
                else:
                    for i in range(len(clean_rel_inds)):
                        self.metadata[i]['modified'] = False

                if stop =='end':
                    for i,v in enumerate(clean_rel_inds):

                        if modified == False:
                            raw_trace = np.asarray(self.dict[i]['sig'])
                            self.metadata[i]['DAC']=raw_trace[1024,start:]
                            traces.append(raw_trace[v,start:])
                            self.metadata[i]['time'] = np.arange(0, (self.dict[i]['sig'].shape[1]-start)/20000., 1/20000.)
                        else:
                            traces.append(np.asarray(self.dict[i]['sig'])[:,start:])
                            self.metadata[i]['time'] = self.dict[i]['time'][:]
                        progr.value += 1
                else:
                    for i,v in enumerate(clean_rel_inds):
                        raw_trace = np.asarray(self.dict[i]['sig'])
                        self.metadata[i]['DAC']=raw_trace[1024,start:]
                        traces.append(raw_trace[v,start:stop])
                        self.metadata[i]['time'] = np.arange(0, (stop-start)/20000., 1/20000.)
                        progr.value += 1

                progr.close()


            return traces

        def direct_read(self,path,start,stop,modified):
            # Generate an array with the length of the clean indices and broadcast the data directly into array.
            #-> The slicing is very time consuming for many datapoints (>100000)
            if isinstance(self.dict,File):
                electrode_info = np.asarray(self.dict['mapping']['channel','electrode'])
                mask = electrode_info['electrode']!=-1
                clean_rel_inds = electrode_info['channel'][mask]

                traces = np.empty((len(clean_rel_inds),stop-start))
                self.metadata['DAC'] = np.empty([stop-start])
                self.dict['sig'].read_direct(traces[:,:],source_sel=np.s_[clean_rel_inds,start:stop])
                self.dict['sig'].read_direct(self.metadata['DAC'][:],source_sel=np.s_[1024,start:stop])
                print self.metadata['DAC'].shape
                self.metadata['time'] = np.arange(0, (stop-start)/20000., 1/20000.)

            if isinstance(self.dict,list):
                electrode_info = [np.asarray(i['mapping']['channel','electrode']) for i in self.dict]
                mask = [i['electrode']!=-1 for i in electrode_info]
                clean_rel_inds = [i[0]['channel'][i[1]] for i in zip(electrode_info,mask)]

                traces = [np.empty((len(i),stop-start)) for i in clean_rel_inds]

                progr = FloatProgress(min=0, max=len(self.dict),description='Importing...',bar_style='success')
                display(progr)

                for i,v in enumerate(zip(traces,clean_rel_inds)):
                    self.dict[i]['sig'].read_direct(v[0],source_sel=np.s_[v[1],start:stop])
                    self.metadata[i]['DAC'] = np.empty([stop-start])
                    self.dict[i]['sig'].read_direct(self.metadata[i]['DAC'][:],source_sel=np.s_[1024,start:stop])
                    self.metadata[i]['time'] = np.arange(0, (stop-start)/20000., 1/20000.)
                    progr.value += 1
                progr.close()

            return traces

        #First find out wheter the data to read in...
        #is >0.8 GB and the stop != 'end' and ndim < 3 -> Fastest to direct_read with read_direct(...)
        #Note on ndim < 3: Default data has 1028xlenght_of_record -> ndim 2.
        #Else array slice the data
        if isinstance(self.dict,File):
            memory = os.path.getsize(path)
            if memory>1e9*0.8 and stop != 'end' and self.dict['time'].ndim==1:
                return direct_read(self,path,start,stop,modified)
            else:
                return array_slice(self,path,start,stop,modified)

        if isinstance(self.dict,list):
            onlyfiles = sorted([path+'/'+f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            memory = [os.path.getsize(i) for i in onlyfiles]
            mean_memory = sum(memory)/len(memory)

            if mean_memory>1e9*0.8 and stop != 'end' and self.dict[0]['time'].ndim==1:
                return direct_read(self,path,start,stop,modified)
            else:
                return array_slice(self,path,start,stop,modified)

        return traces
