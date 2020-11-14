import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

from arff_helper import ArffHelper

class StatGenerator():

    def __init__(self, latency):
        self.latency = latency
        self.total = 0
        self.conf  = 0
        self.samples = {'F':0,'S':0,'P':0,'B':0}
        self.events  = {'F':0,'S':0,'P':0,'B':0}
        self.durations = {'F':[],'S':[],'P':[],'B':[]}


    def get_stats_folder(self, folder_path, histogram=False, verbose=True):
        for dirpath, dirnames, files in os.walk(folder_path):
            for f in files:
                src = os.path.join(dirpath, f)
                if verbose:
                    print(">>> Reading {}...".format(src))
                self.get_stats_file(src)
        self.print_stats()
        if histogram:
            self.build_histogram()
        


    def get_stats_file(self, file_path):
        arff_obj = ArffHelper.load(open(file_path, 'r'))
        data = arff_obj['data']
        attributes = arff_obj['attributes']
        _,_,conf,label = self._get_attr_window(attributes)
        last_ev = -1
        duration = 1
        val = -1
        for i in range(len(data)):
            val = int(data[i][label])
            self.total += 1
            self._get_data_label(val)
            self.conf += float(data[i][conf])
            if val == last_ev:
                duration += 1
            if val != last_ev:
                self._add_event(val)
                if last_ev != -1:
                    self._add_duration(last_ev, duration)
                    #if duration > 1 and duration < 6:
                    #    print('Aqui:', i+22)
                    #    input()
                    duration = 1
                last_ev = val
        self._add_duration(val, duration)


    def build_histogram(self):
        colors = ['g','b','r','y']
        pattern = {'F': 'fixations',
                   'S': 'saccades',
                   'P': 'smooth pursuits',
                   'B': 'blinks'}
        i = 0
        for k in self.durations.keys():
            durations = np.array(self.durations[k]) * self.latency
            plt.hist(durations, bins=50, 
                     facecolor=colors[i], range=(0,2000))
            plt.xlabel("Pattern duration (ms)")
            plt.ylabel("Pattern occurrence")
            plt.title('Histogram of '+pattern[k])
            plt.grid(True)
            plt.show()
            i += 1



    def print_stats(self):
        m = self.latency
        print('\n\n>> Total number of samples: {}\n'.format(self.total))
        print('>> Percentages:\n     Fixations: {:2f}%\n     Saccades: {:2f}%'.format(                 
                 self.samples['F']/self.total*100,
                 self.samples['S']/self.total*100))
        print('     Pursuits: {:2f}%\n     Blinks: {:2f}%\n'.format(
                 self.samples['P']/self.total*100,
                 self.samples['B']/self.total*100))
        print('>> Events:\n     Fixations: {}\n     Saccades: {}'.format(
                 self.events['F'],
                 self.events['S']))
        print('     Pursuits: {}\n     Blinks: {}\n'.format(
                 self.events['P'],
                 self.events['B']))
        print('>> Confidence average: {}\n'.format(self.conf/self.total))
        print('>> Average pattern length:')
        print('     Fixations: {} [+-{}]\n     Saccades: {} [+-{}]'.format(
                 np.mean(self.durations['F']), np.std(self.durations['F']),
                 np.mean(self.durations['S']), np.std(self.durations['S'])))
        print('     Pursuits: {} [+-{}]\n     Blinks: {} [+-{}]\n'.format(
                 np.mean(self.durations['P']), np.std(self.durations['P']),
                 np.mean(self.durations['B']), np.std(self.durations['B'])))
        print('>> Average pattern length (in ms):')
        print('     Fixations: {} [+-{}]\n     Saccades: {} [+-{}]'.format(
                 np.mean(self.durations['F'])*m, np.std(self.durations['F'])*m,
                 np.mean(self.durations['S'])*m, np.std(self.durations['S'])*m))
        print('     Pursuits: {} [+-{}]\n     Blinks: {} [+-{}]\n'.format(
                 np.mean(self.durations['P'])*m, np.std(self.durations['P'])*m,
                 np.mean(self.durations['B'])*m, np.std(self.durations['B'])*m))
        # print('>> Pattern min/max:')
        # print('     Fixations: {}(min) {}(max)\n     Saccades: {}(min) {}(max)'.format(
        #          np.min(self.durations['F']), np.max(self.durations['F']),
        #          np.min(self.durations['S']), np.max(self.durations['S'])))
        # print('     Pursuits: {}(min) {}(max)\n     Blinks: {}(min) {}(max)\n'.format(
        #          np.min(self.durations['P']), np.max(self.durations['P']),
        #          np.min(self.durations['B']), np.max(self.durations['B'])))
        

    def _get_attr_window(self, attributes):
        x, y, conf, label = 0,0,0,0
        for i in range(len(attributes)):
            if attributes[i][0] == 'x':
                x = i
            elif attributes[i][0] == 'y':
                y = i
            elif attributes[i][0] == 'confidence':
                conf = i
            elif attributes[i][0] == 'handlabeller_final':
                label = i
        return x, y, conf, label


    def _add_duration(self, v, duration):
        if v == 1:
            self.durations['F'].append(duration)
        elif v == 2:
            self.durations['S'].append(duration)
        elif v == 3:
            self.durations['P'].append(duration)
        elif v == 4:
            self.durations['B'].append(duration)


    def _add_event(self, v):
        if v == 1:
            self.events['F'] += 1
        elif v == 2:
            self.events['S'] += 1
        elif v == 3:
            self.events['P'] += 1
        elif v == 4:
            self.events['B'] += 1


    def _get_data_label(self, v):
        if v == 1:
            self.samples['F'] += 1
        elif v == 2:
            self.samples['S'] += 1
        elif v == 3:
            self.samples['P'] += 1
        elif v == 4:
            self.samples['B'] += 1



if __name__=="__main__":
    statgen = StatGenerator(4)
    #base_folder = "../our_dataset_arff/200Hz"
    base_folder = "../data/inputs/GazeCom_ground_truth/"
    #statgen.get_stats_folder(base_folder)

    #treating each folder separately
    for x, y, z in os.walk(base_folder):
        if x != base_folder:
            statgen = StatGenerator(4)
            print('\n-----------\n>>> SHOWING STATS FOR:', x)
            statgen.get_stats_folder(x, verbose=False)
        

