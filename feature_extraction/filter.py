import os
import re
import sys
import numpy as np

from arff_helper import ArffHelper

class OutputFilter():

    def __init__(self, basepath, outpath):
        self.basepath = basepath
        self.outpath = outpath


    def filter_folder(self, verbose=True):
        for dirpath, dirnames, files in os.walk(self.basepath):
            for f in files:
                if "eval.json" in f:
                    continue
                src = os.path.join(dirpath, f)
                patt = re.findall(".*\/(.*)\/[A-Z]*", src)
                outpath = os.path.join(self.outpath, patt[0])
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                outfile = os.path.join(self.outpath, patt[0], f)
                if verbose:
                    print(">>> Reading {}...".format(src))
                arff_obj = self.filter_file(src)
                self.save_file(arff_obj, outfile)



    def filter_file(self, file_path):
        arff_obj = ArffHelper.load(open(file_path, 'r'))
        data = arff_obj['data']
        attributes = arff_obj['attributes']
        _,_,conf,label = self._get_attr_window(attributes)
        last_ev = "NONE"
        duration = 1
        val = "NONE"
        to_change = []
        for i in range(len(data)):
            val = data[i][label]
            if val == last_ev:
                duration += 1
                to_change.append(i)
            if val != last_ev:
                if last_ev != "NONE":
                    preserve_changes = False
                    if duration < 4:
                        preserve_changes = True
                        for j in to_change:
                            arff_obj['data'][j][label] = val
                    duration = 1
                    if preserve_changes:
                        to_change.append(i)
                    else:
                        to_change = [i]
                last_ev = val
        return arff_obj


    def save_file(self, arff_obj, file_path):
        output = ArffHelper.dumps(arff_obj)
        with open(file_path, 'w') as f:
            f.write(output)


    

    def _get_attr_window(self, attributes):
        x, y, conf, label = 0,0,0,0
        for i in range(len(attributes)):
            if attributes[i][0] == 'x':
                x = i
            elif attributes[i][0] == 'y':
                y = i
            elif attributes[i][0] == 'confidence':
                conf = i
            elif attributes[i][0] == 'EYE_MOVEMENT_TYPE':
                label = i
        return x, y, conf, label





if __name__=="__main__":
    if len(sys.argv) != 3:
        print('usage: <filter> <basepath> <outpath>')
        sys.exit(-1)
    else:
        basepath = sys.argv[1]
        outpath = sys.argv[2]
        out_filter = OutputFilter(basepath, outpath)
        out_filter.filter_folder()
    #output = out_filter.filter_file("AAF_beach.arff")
    #out_filter.save_file(output, 'AAF_beach_mod.arff')
        

