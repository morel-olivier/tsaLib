#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project TSA / tsalib
Created on Wed Sep 26 07:17:29 2018
To run in the 'tsa' environement (source activate tsa)

@author: pierre
"""


import sys, os, re, math
import scipy as sp
import numpy as np
import random
from scipy.io import wavfile


class cWave:
    def __init__(self):
        pass
        
    def readWaveFile(self, _filename, display=False):
        
        try:
            [self.samplingRate, self.samples] = wavfile.read(_filename)
        except:
            print('cWave::readWaveFile: unable to read ',_filename)
            return False
        
        self.numFrames=self.samples.shape[0]
        if(self.samples.size/self.numFrames>1):
            self.channels=self.samples.shape[1]
        else:
            self.channels=1
        self.duration=self.numFrames/self.samplingRate
        if display:
            if self.channels==1:
                print('sample type: ',type(self.samples[0]))
                if isinstance(self.samples[0], np.float64):
                    print('FLOAT')
            else:
                print('sample type: ',type(self.samples[0][0]))
            print('channels: ',self.channels)
            print('sampling rate: ',self.samplingRate)
            print('number of frames: ',self.numFrames)
            print('duration: ',self.duration)   
        return True
    
    def getWaveData(self):
        data=np.zeros( (self.numFrames,self.channels), dtype='float' )
        if(self.channels==2):
            for k in range(0,self.numFrames):
                data[k][0]=self.samples[k][0]
                data[k][1]=self.samples[k][1]
        else:
            for k in range(0,self.numFrames):
                data[k][0]=self.samples[k]
        return self.samplingRate, data


    def getLeftData(self):
        data=np.zeros( (self.numFrames), dtype='float' )
        if(self.channels==2):
            for k in range(0,self.numFrames):
                data[k]=self.samples[k][0]
        else:
            for k in range(0,self.numFrames):
                data[k]=self.samples[k]
        return self.samplingRate, data


    def writeWaveFile(self, _filename, _samplingRate, _frames):
        
        try:
            wavfile.write(_filename, _samplingRate, _frames)
        except:
            print('cWave::writeWaveFile: unable to write ',_filename)
            return False
        
        print('channels: ',_frames.shape[1])
        print('sampling rate: ',_samplingRate)
        print('number of frames: ',_frames.shape[0])
        print('duration: ',_frames.shape[0]/_samplingRate)   
