import subprocess
import shlex
import sys

EPOCHS = 20
MINIBATCH = 128
WINDOW = 257
SAMPLES = 50000
OVERLAP = 192
FEATURES = "speed direction"
FEAT_FOLDER = "data/inputs/GazeCom_new_features"
NUM_FEAT_SCALES = 6
CONV_MOD = "same"
NUM_CONV = 4
NUM_DENSE = 0
NUM_BLSTM = 2
MODEL_FOLDER = "models/tcn_257_6/"
MODEL = "tcn_model.py"
OUT_FOLDER = "outputs/outputs_tcn_257_6"
SPTOOL = "./sp_tool"


def run_command(args):
    try :
        subprocess.run(args)
    except KeyboardInterrupt:
        print('>>> Exiting...')
        sys.exit(-1)

if __name__=="__main__":
    output = subprocess.check_output("ls {} | wc -l".format(FEAT_FOLDER), shell=True)
    movies = int(output.decode()[:-1]) 
    command  = "python3 {} --batch-size {} --epochs {} --window {} ".format(MODEL, MINIBATCH, EPOCHS, WINDOW)
    command += "-o " #run each video as it comes
    command += "--overlap {} --features {} --feat-folder {} ".format(OVERLAP, FEATURES, FEAT_FOLDER)
    command += "--num-feature-scales {} --conv-padding-mode {} ".format(NUM_FEAT_SCALES, CONV_MOD)
    command += "--num-conv {} --num-dense {} --num-blstm {} ".format(NUM_CONV, NUM_DENSE, NUM_BLSTM)
    command += "--training-samples {} --model-folder {}".format(SAMPLES, MODEL_FOLDER)
    args = shlex.split(command)
    for i in range(movies):
        run_command(args)
    cmd_final  = command.replace(" -o ", " --final ")
    cmd_final += " --output-folder {} --sp-tool-folder {}".format(OUT_FOLDER, SPTOOL)
    args = shlex.split(cmd_final)
    run_command(args)
            
