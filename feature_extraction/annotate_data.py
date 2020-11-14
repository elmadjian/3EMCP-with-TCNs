import numpy as np
import os
import re
import sys

from arff_helper import ArffHelper

class Annotator():
    BASEPATH = "../our_dataset_arff"
    OUTPATH  = "../our_dataset_features"

    def __init__(self, basepath=BASEPATH, outpath=OUTPATH):
        self.base_path = basepath
        self.out_path = outpath
        self.ppd_f = -1


    def load_arff_file(self, file_path):
        arff_obj = ArffHelper.load(open(file_path, 'r'))
        self.ppd_f = self.calculate_ppd(arff_obj)
        #print('PPD:', self.ppd_f)
        return arff_obj

    def save_arff_file(self, arff_obj, file_path):
        output = ArffHelper.dumps(arff_obj)
        with open(file_path, 'w') as f:
            f.write(output)
            #ArffHelper.dump(arff_obj, f)

    def annotate_all(self):
        for dirpath, dirnames, files in os.walk(self.base_path):
            for f in files:
                src = os.path.join(dirpath, f)
                patt = re.findall(".*\/(.*)\/[A-Z]*", src)
                outpath = os.path.join(self.out_path, patt[0])
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                outfile = os.path.join(self.out_path, patt[0], f)
                print(">>> extracting features from {}...".format(src))
                output = self.annotate_data(src)    
                self.save_arff_file(output, outfile)


    def annotate_data(self, arff_file):
        window_size = [1, 2, 4, 8, 16, 32, 64, 128]
        arff = self.load_arff_file(arff_file)
        comment = "The number after speed, direction denotes the step size"\
                +" that was used for the calculation.\n"\
                +"Acceleration was calculated between two adjacent samples"\
                +" of the already low pass filtered velocity.\n"
        arff['description'] += comment
        for s in window_size:
            speed_att_name = 'speed_' + str(s)
            dir_att_name   = 'direction_' + str(s)
            acc_att_name   = 'acceleration_' + str(s)
            dev_att_name   = 'standard_dev_' + str(s)
            disp_att_name  = 'displacement_' + str(s)
            ArffHelper.add_column(arff, speed_att_name, 'NUMERIC', 0)
            ArffHelper.add_column(arff, dir_att_name, 'NUMERIC', 0)
            ArffHelper.add_column(arff, acc_att_name, 'NUMERIC', 0)
            ArffHelper.add_column(arff, dev_att_name, 'NUMERIC', 0)
            ArffHelper.add_column(arff, disp_att_name, 'NUMERIC', 0)
            self._get_velocity(arff, s)
            self._get_deviation(arff, s)
            self._get_acceleration(arff, s)
        return arff


    def annotate_window(self, data, feat_size, feat_list, latency):
        '''
        Method to be used with online classification
        data: bi-dimensional array containing x coord values,
            y coord values, and confidence values within a fixed window. 
            E.g.: data = [[0.1, 0.2, 0.3], [0.5, 0.5, 0.7], [1.0,1.0,1.0]]
        feat_size: number of features
        feat_list: an array with one or more of the following:
            'acc', 'velocity', 'direction'
        latency: timestep between two samples (in microseconds)
        '''
        windows = [1,2,4,8,16,32,64,128][:feat_size]
        for feat in feat_list:
            if feat == 'acc':
                pass
            elif feat == 'speed':
                v, d = self._get_velocity_window(data, windows, latency)
        tensor = np.hstack((v, d))
        return np.array([tensor])


    def _get_start_end_window(self, i, step, win_width, c_minconf, data):
        x_list, y_list, conf = data
        #getting initial interval
        if step == win_width:
            start_pos = i - step
            end_pos = i
        else:
            start_pos = i - step
            end_pos = i + step
        #finetuning interval
        if start_pos < 0 or conf[start_pos] < c_minconf: 
            start_pos = i
        if end_pos >= len(x_list) or conf[end_pos] < c_minconf:
            end_pos = i
        return start_pos, end_pos


    def _get_velocity_window(self, data, windows, latency):
        x_list, y_list, conf = data
        c_minconf  = 0.75
        tensor_vel = np.zeros((len(x_list), len(windows)))
        tensor_dir = np.zeros((len(x_list), len(windows))) 
        for i in range(len(x_list)):
            for j in range(len(windows)):
                w = windows[j]
                step = np.math.ceil(w/2)
                if conf[i] < c_minconf:
                    continue
                start_pos, end_pos = self._get_start_end_window(i,step,w,c_minconf,data)
                if start_pos == end_pos: #invalid interval
                    continue
                diff_x = x_list[end_pos] - x_list[start_pos]
                diff_y = y_list[end_pos] - y_list[start_pos]
                ampl = np.math.sqrt(diff_x**2 + diff_y**2)
                time = ((end_pos - start_pos)*latency)/1000000
                tensor_vel[i][j] = (ampl/time)/self.ppd_f
                tensor_dir[i][j] = np.math.atan2(diff_y, diff_x)
        return tensor_vel, tensor_dir    


    def get_attr_window(self, attributes):
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


    def _get_attr_position(self, attributes, window):
        time, x, y, conf, speed, direction, acc, dev, disp = 0,0,0,0,0,0,0,0,0
        speed_name = 'speed_' + str(window)
        dir_name = 'direction_' + str(window)
        acc_name = 'acceleration_' + str(window)
        dev_name = 'standard_dev_' + str(window)
        disp_name = 'displacement_'+ str(window)
        for i in range(len(attributes)):
            if attributes[i][0] == 'time':
                time = i
            elif attributes[i][0] =='x':
                x = i
            elif attributes[i][0] =='y':
                y = i
            elif attributes[i][0] == 'confidence':
                conf = i
            elif attributes[i][0] == speed_name:
                speed = i
            elif attributes[i][0] == dir_name:
                direction = i
            elif attributes[i][0] == acc_name:
                acc = i
            elif attributes[i][0] == dev_name:
                dev = i
            elif attributes[i][0] == disp_name:
                disp = i
        return time, x, y, conf, speed, direction, acc, dev, disp


    def _get_start_end(self, i, step, win_width, conf, c_minconf, data):
        #getting initial interval
        if step == win_width:
            start_pos = i - step
            end_pos = i
        else:
            start_pos = i - step
            end_pos = i + step
        #finetuning interval
        if start_pos < 0 or data[start_pos][conf] < c_minconf: 
            start_pos = i
        if end_pos >= len(data) or data[end_pos][conf] < c_minconf:
            end_pos = i
        return start_pos, end_pos


    def _get_velocity(self, arff_obj, window):
        data = arff_obj['data']
        attributes = arff_obj['attributes']
        c_minconf = 0.75
        step = np.math.ceil(window/2)
        t,x,y,conf,s,d,_,_,_ = self._get_attr_position(attributes, window)
        for i in range(len(data)):
            if data[i][conf] < c_minconf:
                continue
            start_pos, end_pos = self._get_start_end(i,step,window,conf,c_minconf,data)
            if start_pos == end_pos: #invalid interval
                continue
            diff_x = data[end_pos][x] - data[start_pos][x]
            diff_y = data[end_pos][y] - data[start_pos][y]
            ampl = np.math.sqrt(diff_x**2 + diff_y**2)
            time = (data[end_pos][t] - data[start_pos][t])/1000000
            arff_obj['data'][i][s] = ampl/time
            arff_obj['data'][i][d] = np.math.atan2(diff_y, diff_x)

    
    def _get_deviation(self, arff_obj, window):
        data = arff_obj['data']
        attributes = arff_obj['attributes']
        c_minconf = 0.75
        step = np.math.ceil(window/2)
        t,x,y,conf,_,_,_,s,d = self._get_attr_position(attributes, window)
        for i in range(len(data)):
            if data[i][conf] < c_minconf:
                continue
            start_pos, end_pos = self._get_start_end(i,step,window,conf,c_minconf,data)
            if start_pos == end_pos: #invalid interval
                continue
            list_x, list_y = [], []
            disp_x, disp_y = 0, 0
            for j in range(start_pos, end_pos+1):
                list_x.append(data[j][x])
                list_y.append(data[j][y])
            for j in range(start_pos, end_pos):
                disp_x += data[j+1][x] - data[j][x]
                disp_y += data[j+1][y] - data[j][y]
            ampl = np.math.sqrt(disp_x**2 + disp_y**2)
            std_x = np.std(list_x)
            std_y = np.std(list_y)
            std = np.mean([std_x, std_y])
            arff_obj['data'][i][s] = std
            arff_obj['data'][i][d] = ampl


    def _get_acceleration(self, arff_obj, window):
        w = window
        window = 1
        data = arff_obj['data']
        attributes = arff_obj['attributes']
        c_minconf = 0.75
        step = np.math.ceil(window/2)
        t,_,_,conf,s,d,a,_,_ = self._get_attr_position(attributes, w)
        for i in range(len(data)):
            if data[i][conf] < c_minconf:
                continue
            start_pos, end_pos = self._get_start_end(i,step,window,conf,c_minconf,data)
            if start_pos == end_pos:
                continue
            vel_startx = data[start_pos][s]*np.math.cos(data[start_pos][d])
            vel_starty = data[start_pos][s]*np.math.sin(data[start_pos][d])
            vel_endx = data[end_pos][s]*np.math.cos(data[end_pos][d])
            vel_endy = data[end_pos][s]*np.math.sin(data[end_pos][d])
            delta_t = (data[end_pos][t]-data[start_pos][t])/1000000
            acc_x = (vel_endx-vel_startx)/delta_t
            acc_y = (vel_endy-vel_starty)/delta_t
            arff_obj['data'][i][a] = np.math.sqrt(acc_x**2 + acc_y**2)


    def calculate_ppd(self, arff_object, skip_consistency_check=False):
        """
        Pixel-per-degree value is computed as an average of pixel-per-degree values for each dimension (X and Y).

        :param arff_object: arff object, i.e. a dictionary that includes the 'metadata' key.
                    @METADATA in arff object must include "width_px", "height_px", "distance_mm", "width_mm" and
                    "height_mm" keys for successful ppd computation.
        :param skip_consistency_check: if True, will not check that the PPD value for the X axis resembles that of
                                    the Y axis
        :return: pixel per degree.

        """
        # Previous version of @METADATA keys, now obsolete
        OBSOLETE_METADATA = {
            'PIXELX': ('width_px', lambda val: val),
            'PIXELY': ('height_px', lambda val: val),
            'DIMENSIONX': ('width_mm', lambda val: val * 1e3),
            'DIMENSIONY': ('height_mm', lambda val: val * 1e3),
            'DISTANCE': ('distance_mm', lambda val: val * 1e3)
        }

        for obsolete_key, (new_key, value_modifier) in OBSOLETE_METADATA.items():
            if obsolete_key in arff_object['metadata'] and new_key not in arff_object['metadata']:
                arff_object['metadata'][new_key] = value_modifier(arff_object['metadata'].pop(obsolete_key))

        theta_w = 2*np.math.atan(arff_object['metadata']['width_mm'] /
                                (2 * arff_object['metadata']['distance_mm']))*180/np.math.pi
        theta_h = 2*np.math.atan(arff_object['metadata']['height_mm'] /
                                (2 * arff_object['metadata']['distance_mm']))*180/np.math.pi

        ppdx = arff_object['metadata']['width_px'] / theta_w
        ppdy = arff_object['metadata']['height_px'] / theta_h

        ppd_relative_diff_thd = 0.2
        if not skip_consistency_check and abs(ppdx - ppdy) > ppd_relative_diff_thd * (ppdx + ppdy) / 2:
            warnings.warn('Pixel-per-degree values for x-axis and y-axis differ '
                        'by more than {}% in source file {}! '
                        'PPD-x = {}, PPD-y = {}.'.format(ppd_relative_diff_thd * 100,
                                                        arff_object['metadata'].get('filename', ''),
                                                        ppdx, ppdy))
        return (ppdx + ppdy) / 2


    #def _get_distance(self, arff_obj, window):




if __name__=="__main__":
    #basepath = "../our_dataset_arff/25Hz"
    #outpath  = "../our_dataset_features/25Hz"
    basepath = "../data/inputs/GazeCom_ground_truth"
    outpath = "../data/inputs/GazeCom_new_features"
    annotator = Annotator(basepath, outpath)
    annotator.annotate_all()
    #ata = annotator.annotate_data("input.arff")
    #nnotator.save_arff_file(data, 'output.arff')
    
    
    #annotator.save_arff_file(data, 'copia.arff')
