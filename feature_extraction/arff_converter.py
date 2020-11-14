import os
import re

class ArffConverter():

    def __init__(self, resolution, dimension, distance, latency):
        '''
        (int, int), (float, float), (float), (int) -> None
        Resolution should be provided in pixels
        Dimension and Distance are in mm
        Latency between samples from eye tracker (in ms)
        e.g. 200Hz ---> 5 ms
        '''
        self.width_px = resolution[0]
        self.height_px = resolution[1]
        self.width_mm = dimension[0]
        self.height_mm = dimension[1]
        self.distance = distance
        self.latency = latency * 1000


    def convert_folder(self, folder_path, out_path):
        for dirpath, dirnames, files in os.walk(folder_path):
            for f in files:
                src = os.path.join(dirpath, f)
                patt = re.findall(".*\/(.*)\/[a-z]*", src)
                outpath = os.path.join(out_path, patt[0])
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                outfile = os.path.join(out_path, patt[0], f)
                print(">>> Converting {} to ARFF...".format(src))
                output = self.convert_file(src)
                self.save_file(output, outfile)


    def convert_file(self, file_path):
        converted  = self._add_default_comment()
        converted += self._add_metadata()
        converted += self._add_attribute()
        converted += "@DATA\n"
        time = 1000
        with open(file_path, 'r') as f:
            f.readline()
            for line in f.readlines():
                data = line.split('\t')
                x = float(data[0]) * self.width_px
                y = float(data[1]) * self.height_px
                c = data[2]
                l = self._convert_label(data[3])
                converted += str(time) + "," + str(x) + ","
                converted += str(y) + "," +c+ "," +l+ "\n"
                time += self.latency
        return converted


    def save_file(self, file_data, name):
        file_name = name[:-4] + ".arff"
        with open(file_name, 'w') as f:
            f.write(file_data)


    def _add_default_comment(self):
        comment = "% The handlabeller_final column contains the "\
                + "final labels. Individual labels are not provided. "\
                + "The attribute time is in microseconds.\n"\
                + "% Labels in these columns are to be interpreted as follows:\n"\
                + "%   - 0 is UNKNOWN\n"\
                + "%   - 1 is FIX (fixation)\n"\
                + "%   - 2 is SACCADE\n"\
                + "%   - 3 is SP (smooth pursuit)\n"\
                + "%   - 4 is BLINK\n%\n" 
        return comment

    
    def _add_metadata(self):
        metadata  = "%@METADATA width_px {}\n".format(self.width_px)
        metadata += "%@METADATA height_px {}\n".format(self.height_px)
        metadata += "%@METADATA width_mm {}\n".format(self.width_mm)
        metadata += "%@METADATA height_mm {}\n".format(self.height_mm)
        metadata += "%@METADATA distance_mm {}\n".format(self.distance)
        metadata += "@RELATION gaze_labels\n\n"
        return metadata


    def _add_attribute(self):
        attribute  = "@ATTRIBUTE time INTEGER\n"
        attribute += "@ATTRIBUTE x NUMERIC\n"
        attribute += "@ATTRIBUTE y NUMERIC\n"
        attribute += "@ATTRIBUTE confidence NUMERIC\n"
        attribute += "@ATTRIBUTE handlabeller_final INTEGER\n\n"
        return attribute


    def _convert_label(self, val):
        if val.startswith("F"):
            return "1"
        elif val.startswith("S"):
            return "2"
        elif val.startswith("P"):
            return "3"
        elif val.startswith("B"):
            return "4"


if __name__=="__main__":
    converter = ArffConverter((1920,1080), (564.896,342.9), 500, 5)
    input_folder = "../our_dataset"
    output_folder = "../our_dataset_arff"
    converter.convert_folder(input_folder, output_folder)
        

