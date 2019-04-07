from scipy.spatial import cKDTree
import numpy as np


def spiker(threshold_factor,time_threshold,data,time,start=0,stop=-1,range_list=False):
    '''
    This spike detector looks at positive spikes only. You can do post-treatment to reconstruct the full-spike
    later from the returned time stamps. The function returns a list with all the spike_times ordered by electrode.
    Arguments:
    data: Python list of filtered (highpass/bandpass) traces, not raw. needs to be reasonably flat and centered around 0.
    threshold: factor which will be multiplied to std for thresholding. So any data point above 'threshold' is considered.
    time_threshold: time interval (in seconds) inside which the largest spike counts as 'the spike'.
    time: time  of data.
    start: start points to search for spikes
    stop: stop point
    '''

    spike_inds=[]
    for j,tr in enumerate(data):
        if range_list==False:
            std_filt = np.std(tr[start:stop])
            threshold = std_filt*threshold_factor
            over_threshold_islets=np.where(tr[start:stop]>threshold)[0]
        else:
            std_filt= np.std(tr[range_list[j][0]:range_list[j][1]])
            threshold = std_filt*threshold_factor
            over_threshold_islets=np.where(tr[range_list[j][0]:range_list[j][1]]>threshold)[0]
        #Return empty  list if no threshold crossings
        if (over_threshold_islets.size and over_threshold_islets.size != 1):
            #most-left indices of putative spikes
            islet_edges=np.hstack((0,np.asarray([1+el for el in np.where(np.diff(over_threshold_islets)!=1)[0]])))
            #Following processing can only be done if there is more than one spike
            if islet_edges.size != 1:
                islet_list=[over_threshold_islets[islet_edges[i]:islet_edges[i+1]] for i in range(len(islet_edges)-1)]
                islet_list.append(np.asarray(over_threshold_islets[islet_edges[-1]:]))
                if range_list==False:
                    maxima=np.asarray([np.argmax(tr[start:stop][el])+el[0] for el in islet_list])
                else:
                    maxima=np.asarray([np.argmax(tr[range_list[j][0]:range_list[j][1]][el])+el[0] for el in islet_list])
                val_tree=np.vstack( (np.zeros(len(maxima)),time[maxima] ) ).T
                stamp_kdTree=cKDTree(val_tree)
                too_close_tuplets=stamp_kdTree.query_ball_point(val_tree,r=time_threshold)
                if range_list==False:
                    real_maxima=maxima[np.unique([el[np.argmax(tr[start:stop][maxima[el]])] for el in too_close_tuplets])]+start
                else:
                    real_maxima=maxima[np.unique([el[np.argmax(tr[range_list[j][0]:range_list[j][1]][maxima[el]])] for el in too_close_tuplets])]+range_list[j][0]
                spike_inds.append(real_maxima.tolist())
            else:
                if range_list==False:
                    spike_inds.append([over_threshold_islets[0]+start])
                else:
                    spike_inds.append([over_threshold_islets[0]+range_list[j][0]])
        else:
            spike_inds.append([])


    return spike_inds
