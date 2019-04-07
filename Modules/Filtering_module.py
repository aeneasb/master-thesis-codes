import Import_module as mi
from scipy.signal import butter, filtfilt
import numpy as np
from ipywidgets import FloatProgress
from IPython.display import display
from scipy.ndimage.filters import uniform_filter1d
from itertools import groupby

class Filtering:
    '''
    This class inherits from import_rawdata (found in Import_module.py).
    butter_filter:
    Does the actual filtering. Raw traces are filtered using non-causal butterworth filter to bandpass the 
    recordings according to lowcut, highcut, the order of the filter and the frame rate.
    discard_out_of_bounds:
    Using a sliding standard deviation calculation function corrupt traces are
    identified and discarded.
    reshape:
    Reshape filtered traces into predefined bin-length
    '''

    def __init__(self,import_raw_data_instance):
        self.raw_data = import_raw_data_instance.raw_data
        self.metadata = import_raw_data_instance.metadata

    def butter_filter(self,lowcut,highcut=0,order=2,fs=20000.):

        nyq = 0.5*fs

        if highcut!=0:
            low = lowcut/nyq
            high = highcut/nyq
            b,a = butter(order,[low,high],btype='band',analog=False)
        else:
            low = lowcut/nyq
            b,a = butter(order,low,btype='high',analog=False)

        if isinstance(self.metadata,list):
            progr = FloatProgress(min=0, max=len(self.metadata),description='Filtering...',bar_style='success')
            display(progr)
            cut_data_butter = [None]*len(self.raw_data)
            for i,k in enumerate(self.raw_data):
                cut_data_butter[i]=np.empty(k.shape)
                for j,z in enumerate(self.raw_data[i]):
                    cut_data_butter[i][j] = filtfilt(b,a,z)
                progr.value += 1
            self.butter_data = cut_data_butter
            progr.close()

        if isinstance(self.metadata,dict):
            cut_data_butter = [None]*len(self.raw_data)
            for i,k in enumerate(self.raw_data):
                cut_data_butter[i] = filtfilt(b, a, k)
            self.butter_data = np.asarray(cut_data_butter)

    def discard_out_of_bounds(self):
        '''Replace all traces which cross the range [0,1023] with some exceptions'''
        def window_stdev(data, radius):
            '''Explanation on https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows'''
            c1 = uniform_filter1d(data.astype(np.float32), radius*2, mode='constant', origin=-radius)
            c2 = uniform_filter1d(data.astype(np.float32)*data.astype(np.float32), radius*2, mode='constant', origin=-radius)
            return ((c2 - c1*c1)**.5)[:-radius*2+1]

        def get_range_to_keep(critical_traces):
            ## Idea is to get the biggest intervall of data with slid_std>0.
            ## Discard traces where the interval is smaller than len_recording/3
            ## Attention: Function discards useful traces when trace has many stimuli
            delete =[]
            keep = []
            range_to_keep = []

            slid_std = np.asarray([window_stdev(o,20) for o in self.raw_data[critical_traces]])
            bool_arrays = np.array(slid_std>0.01,dtype=int)

            for p,bool_array in enumerate(bool_arrays):
                counts = np.asarray([[i,sum(1 for k in j)] for i,j in groupby(bool_array)])

                if bool_array[0]==1:
                    max_count = np.max(counts[range(0,counts.shape[0],2),1])
                    fake_count = np.copy(counts)
                    fake_count[range(1,counts.shape[0],2),1]=0
                    max_rel_ind = fake_count[:,1].tolist().index(max_count)
                elif bool_array[0]==0:
                    max_count = np.max(counts[range(1,counts.shape[0],2),1])
                    fake_count = np.copy(counts)
                    fake_count[range(0,counts.shape[0],2),1]=0
                    max_rel_ind = fake_count[:,1].tolist().index(max_count)

                abs_ind = 0
                for i,j in enumerate(counts[:,1]):
                    if i<max_rel_ind:
                        abs_ind+=j
                abs_range = range(abs_ind,abs_ind+max_count)

                if len(abs_range)>slid_std.shape[1]/3:
                    keep.append(critical_traces[p])
                    #range_to_keep.append(abs_range[500:-500])#To make sure that trace is within bounds
                    range_to_keep.append([abs_range[500],abs_range[-500]])
                else:
                    delete.append(critical_traces[p])

            return delete,keep,range_to_keep


        if isinstance(self.metadata,list):

            for i,j in enumerate(self.raw_data):
                #Get the indices of the traces containing 1023 or 0 -> Putative canditates to discard
                out_of_bounds = np.asarray([idx for idx in range(len(j)) if (1023 in j[idx,:]) or (0 in j[idx,:])])
                #print ', '.join('v{}: {}'.format(v, i) for v, i in enumerate(out_of_bounds))
                if not out_of_bounds.tolist():
                    self.metadata[i]['Out_of_bounds']='Zero traces out of bounds'
                    continue
                self.metadata[i]['Out_of_bounds']=out_of_bounds.tolist()

                #Replace all traces with std of raw_traces <1 with zeros in the filtered data
                raw_std=np.std(j,axis=1)
                mask = raw_std[out_of_bounds]<1
                #Alternatively replace with fake data:
                #data.butter_data[out_of_bounds[mask]]=np.random.normal(0,0.8,(1,len(filtdat.metadata['time'])))
                self.butter_data[i][out_of_bounds[mask]]=np.zeros([1,self.metadata[i]['time'].size])

                #For the remaining traces with std>1 check with moving std if there is at least one std-winddow<0.01. If yes -> discard
                mask = raw_std[out_of_bounds]>1
                print len(mask)
                if np.sum(mask)!=0:
                    slid_std = np.asarray([window_stdev(o,20) for o in j[out_of_bounds[mask]]])

                    false_pos = [out_of_bounds[mask][o[0]] for o in enumerate(slid_std) if np.sum(o[1]<0.01)!=0] #np.sum(o[1]<0.01) Count number of true elements
                    #print ', '.join('fp v{}: {}'.format(v, i) for v, i in enumerate(false_pos))
                    self.butter_data[i][false_pos]=np.zeros([1,self.metadata[i]['time'].size])

                    false_neg = list(set(out_of_bounds[mask])-set(false_pos)) #The remaining elements only touch the boundaries during the stimulus
                    #print ', '.join('fn v{}: {}'.format(v, i) for v, i in enumerate(false_neg))
                    for o in false_neg:
                        self.metadata[i]['Out_of_bounds'].remove(o)
                    self.metadata[i]['Critical_traces'] = false_neg

                x_tokill=[v%220 for v in self.metadata[i]['clean_abs_inds'][self.metadata[i]['Out_of_bounds']]]
                y_tokill=[v/220 for v in self.metadata[i]['clean_abs_inds'][self.metadata[i]['Out_of_bounds']]]
                self.metadata[i]['electrode_map'][y_tokill,x_tokill]=0

                print 'Recording {0:.1f}: Discarded {1:.1f} traces.'.format(i, len(self.metadata[i]['Out_of_bounds']))
                if 'Critical_traces' in self.metadata[i].keys():
                    print 'Recording {0:.1f}: {1:.1f} Critical traces.'.format(i, len(self.metadata[i]['Critical_traces']))

        if isinstance(self.metadata,dict):
            #Get the indices of the traces touchin the boundaries (1023 or 0) -> Putative canditates to discard
            out_of_bounds = np.asarray([idx for idx in range(len(self.raw_data)) if (1023 in self.raw_data[idx,:]) or (0 in self.raw_data[idx,:])])
            #print ', '.join('v{}: {}'.format(v, i) for v, i in enumerate(out_of_bounds))
            if not out_of_bounds.tolist():
                self.metadata['Out_of_bounds']='Zero traces out of bounds'
                print self.metadata['Out_of_bounds']
                return []
            self.metadata['Out_of_bounds']=out_of_bounds.tolist() #Save the indeces of the preliminary discarded traces

            #The easiest traces to eliminate are those with std < 1, ususally electrodes which don't record anything.
            raw_std=np.std(self.raw_data,axis=1)
            mask = raw_std[out_of_bounds]<1
            #Replace those traces with zeros in the filtered data
            self.butter_data[out_of_bounds[mask]]=np.zeros([1,self.metadata['time'].size])
            #Alternatively replace with fake data:
            #data.butter_data[out_of_bounds[mask]]=np.random.normal(0,0.8,(1,len(filtdat.metadata['time'])))

            #The next target are traces with std>1
            mask = raw_std[out_of_bounds]>1
            if np.sum(mask)!=0:
                #Calculate the std at each location using the moving std function
                slid_std = np.asarray([window_stdev(o,20) for o in self.raw_data[out_of_bounds[mask]]])

                #The next kind of traces to eliminate are those which only record the stimulus but otherwise are 0.
                #these traces have slid_std values<0.01 over large parts of the recording
                #Note: np.sum(o[1]<0.01) counts number of std bins which are <0.01
                #If the sum of these bins is bigger than half the total number of bins (~length of recording/2) -> discard trace
                false_pos = [out_of_bounds[mask][o[0]] for o in enumerate(slid_std) if np.sum(o[1]<0.01)>=len(o[1])/2]
                self.butter_data[false_pos]=np.zeros([1,self.metadata['time'].size])

                #The next kind of traces to separate ar those which just touch the boundaries during the stimulus
                #but immediately return to normal recording conditions.
                #They can be trated as false-negative and should be kept:
                false_neg = [out_of_bounds[mask][o[0]] for o in enumerate(slid_std) if np.sum(o[1]<0.1)==0]
                for o in false_neg:
                    self.metadata['Out_of_bounds'].remove(o)
                self.metadata['false_neg'] = false_neg

                #The last kind of traces to separate are the hard ones:
                #Some traces which get out of bounds due to stim. artefact get into the boundaries after a while.
                #They should not be discarded since they can contain useful information. They are given by the remaining traces.
                critical_traces = list(set(out_of_bounds[mask])-set(false_pos)-set(false_neg))
                [delete,keep,range_to_keep]= get_range_to_keep(critical_traces)
                for o in keep:
                    self.metadata['Out_of_bounds'].remove(o)
                keys = ['Keep','Range']
                self.metadata['Critical_traces']=keep
                self.metadata['Range']=range_to_keep
                self.butter_data[delete]=np.zeros([1,self.metadata['time'].size])


            x_tokill=[v%220 for v in self.metadata['clean_abs_inds'][self.metadata['Out_of_bounds']]]
            y_tokill=[v/220 for v in self.metadata['clean_abs_inds'][self.metadata['Out_of_bounds']]]
            self.metadata['electrode_map'][y_tokill,x_tokill]=0

            print 'Discarded {0:.1f} traces.'.format(len(self.metadata['Out_of_bounds']))
            if 'Critical_traces' in self.metadata.keys():
                print '{0:.1f} critical traces.'.format(len(self.metadata['Critical_traces']))

    def reshape(self,bin_length):
        # bin_length comes from script: find_best_stim_el.py

        if isinstance(self.metadata,dict):

            num_stim = self.raw_data.shape[1]/bin_length

            self.raw_data = self.raw_data.reshape((self.raw_data.shape[0],num_stim,bin_length))
            self.raw_data = np.swapaxes(self.raw_data,0,1)

            self.butter_data = self.butter_data.reshape((self.butter_data.shape[0],num_stim,bin_length))
            self.butter_data = np.swapaxes(self.butter_data,0,1)

        if isinstance(self.metadata,list):

            raw_reshaped = []
            filt_reshaped = []

            for i,t in enumerate(self.raw_data):
                num_stim = t.shape[1]/bin_length
                t = t.reshape((t.shape[0],num_stim,bin_length))
                t = np.swapaxes(t,0,1)
                raw_reshaped.append(t)

            for i,t in enumerate(self.butter_data):
                num_stim = t.shape[1]/bin_length
                t = t.reshape((t.shape[0],num_stim,bin_length))
                t = np.swapaxes(t,0,1)
                filt_reshaped.append(t)

            self.raw_data = raw_reshaped
            self.butter_data = filt_reshaped
