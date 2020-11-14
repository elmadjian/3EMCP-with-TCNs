import sys, os

OUT_FOLDER = "outputs_tcn_385_7_plus_filtered"
GT_FOLDER = "data/inputs/GazeCom_ground_truth"
SPTOOL = "./sp_tool"


def run_command(args):
    try :
        os.system(args)
    except KeyboardInterrupt:
        print('>>> Exiting...')
        sys.exit(-1)

if __name__=="__main__":
    if len(sys.argv) != 2:
        print('usage: <evaluate> <out_folder>')
        sys.exit(-1)
    else:
        OUT_FOLDER = sys.argv[1]
        print("Running sp_tool eval --> {}/eval.json".format(OUT_FOLDER))
        command  = 'python3 {}/examples/run_evaluation.py '.format(SPTOOL)
        command += '--in "{}" --hand "{}" > "{}/eval.json"'.format(OUT_FOLDER, GT_FOLDER, OUT_FOLDER)
        run_command(command)

