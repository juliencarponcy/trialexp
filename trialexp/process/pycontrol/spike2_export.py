'''
Functions for exporting event data to spike2
'''

from sonpy import lib as sp
import time
import re
import numpy as np
import os

class Spike2Exporter:
    
    def __init__(self, smrx_filename, max_time_ms, verbose=False) -> None:
        if smrx_filename is None:
            raise Exception('smrx_filename is required')
        #TODO assert .smlx

        mtc = re.search('\.smrx$', smrx_filename)
        if mtc is None:
            raise Exception('smrx_filename has to end with .smrx')

        self.MyFile = sp.SonFile(smrx_filename, nChans = int(400)) # Allow up to 400 channels
        self.smrx_filename = smrx_filename
        self.CurChan = 0
        self.UsedChans = 0
        self.Scale = 65535/20
        self.Offset = 0
        self.ChanLow = 0
        self.ChanHigh = 5
        self.tFrom = 0
        self.tUpto = sp.MaxTime64()         # The maximum allowed time in a 64-bit SON file
        self.dTimeBase = 1e-6               # s = microseconds
        self.x86BufSec = 2.
        self.EventRate = 1/(self.dTimeBase*1e3)  # Hz, period is 1000 greater than the timebase
        self.SubDvd = 1                     # How many ticks between attached items in WaveMarks
        self.verbose = verbose

        # max_time_ms1 = np.max([np.max(self.times[k]) for k in keys if any(self.times[k])]) #TODO when no data 

        # list_of_match = [re.match('^\d+', L) for L in self.print_lines if re.match('^\d+', L) is not None]
        # max_time_ms2 = np.max([int(m.group(0)) for m in list_of_match])

        # max_time_ms = np.max([max_time_ms1, max_time_ms2])
        self.time_vec_ms = np.arange(0, max_time_ms, 1000/self.EventRate)
        # time_vec_micros = np.arange(0, max_time_ms*1000, 10**6 * 1/EventRate)

        samples_per_s = self.EventRate
        interval = 1/samples_per_s

        samples_per_ms = 1/1000 * self.EventRate
        interval = 1/samples_per_s
        self.MyFile.SetTimeBase(self.dTimeBase)  # Set timebase


    def write_event(self, X_ms, title, y_index):
        
        (hist, ___) = np.histogram(X_ms, bins=self.time_vec_ms) # time is 1000 too small

        eventfalldata = np.where(hist) #when there is event in that time bin

        self.MyFile.SetEventChannel(y_index, self.EventRate)
        self.MyFile.SetChannelTitle(y_index, title)
        if eventfalldata[0] is not []:
            self.MyFile.WriteEvents(int(y_index), eventfalldata[0]*1000) #dirty fix but works, unit should be the same as in timebase, i.e. microsecond
            time.sleep(0.05)# might help?

        if self.verbose:
            print(f'{y_index}, {title}:')
            nMax = 10
            # nMax = int(MyFile.ChannelMaxTime(int(y_index))/MyFile.ChannelDivide(int(y_index))) 
            print(self.MyFile.ReadEvents(int(y_index), nMax, self.tFrom, self.tUpto)) #TODO incompatible function arguments.
            # [-1] when failed

            # ReadEvents(self: sonpy.amd64.sonpy.SonFile, 
            #     chan: int, 
            #     nMax: int, # probably the end of the range to read in the unit of number of channel divide
            #     tFrom: int, 
            #     tUpto: int = 8070450532247928832, 
            #     Filter: sonpy.amd64.sonpy.MarkerFilter = <sonpy.MarkerFilter> in mode 'First', with trace column -1 and items
            #     Layer 1 [
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            
    def write_marker_for_state(self, X_ms, title, y_index, verbose=False):
        """
        Write a marker for the state in X_ms array.

        Parameters:
        X_ms (list): The x_ms format is expected to be [event1_onset, event1_offset, event2_onset, event2_offset] etc.
        title (str): Title of the marker
        y_index (int): Index of marker channel
        verbose (bool, optional): For verbosity purpose. Defaults to False.

        Returns:
        None
        """
        
        # remove NaN
        X_notnan_ms = [x for x in X_ms if not np.isnan(x)]

        (hist, ___) = np.histogram(X_notnan_ms, bins=self.time_vec_ms) # time is 1000 too small

        eventfalldata = np.where(hist)

        nEvents = len(eventfalldata[0])

        MarkData = np.empty(nEvents, dtype=sp.DigMark)
        for i in range(nEvents):
            if (i+1) % 2 == 0:
                MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 0) #offset, unit the same as timebase
            elif (i+1) % 2 == 1:
                MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 1) #onset
            else:
                raise Exception('oh no')
        self.MyFile.SetMarkerChannel(y_index, self.EventRate)
        self.MyFile.SetChannelTitle(y_index, title)
        if eventfalldata[0] is not []:
            self.MyFile.WriteMarkers(int(y_index), MarkData)
            time.sleep(0.05)# might help?

        if self.verbose:             
            print(f'{y_index}, {title}:')
            print(self.MyFile.ReadMarkers(int(y_index), nEvents, self.tFrom, self.tUpto)) #TODO failed Tick = -1

    def write_textmark(self, X_ms, title, y_index, txt, EventRate, time_vec_ms, verbose=False):
        """Writes text marks to file.

        Parameters
        ----------
        self : object
            Object of class
        X_ms : float
            location of the text in millesecond
        title : string
                Title of file
        y_index : int
                Index in y coordinate
        txt : list 
            List of strings for TextMarkers
        EventRate : int
                    Event Rate
        time_vec_ms : array-like, shape (n_bins, n_features)
                    Array of time in ms
        verbose : bool, optional (default=False)
                If true, print messages when writing or reading TextMarks.

        Returns
        -------
        Hist : array-like
            Array of Histogram values
        MarkData : array-like
                Array of DigMarker values
        TMrkData : array-like
                Array of TextMarker values
        """
            
        (hist, ___) = np.histogram(X_ms, bins=time_vec_ms) # time is 1000 too small

        eventfalldata = np.where(hist)

        nEvents = len(eventfalldata[0])

        MarkData = np.empty(nEvents, dtype=sp.DigMark)

        TMrkData = np.empty(nEvents, dtype=sp.TextMarker)

        for i in range(nEvents):
            if (i+1) % 2 == 0:
                MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 0) #offset
            elif (i+1) % 2 == 1:
                MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 1) #onset
            else:
                raise Exception('oh no')
            TMrkData[i] = sp.TextMarker(txt[i], MarkData[i]) #TODO
            
        self.MyFile.SetTextMarkChannel(y_index, EventRate, max(len(s) for s in txt)+1)
        self.MyFile.SetChannelTitle(y_index, title)
        if eventfalldata[0] is not []:
            self.MyFile.WriteTextMarks(y_index, TMrkData)
            time.sleep(0.05)# might help?

        if self.verbose:
            print(f'{y_index}, {title}:')
            try:
                print(self.yFile.ReadTextMarks(int(y_index), nEvents, self.tFrom, self.tUpto))
            except:
                print('error in print')
                
    def __del__(self):
        del self.MyFile
        # check if the file save is successful
        if os.path.isfile(self.smrx_filename):
            print(f'saved {self.smrx_filename}')
        else:
            print('save failed. Please make sure the folder path exists')