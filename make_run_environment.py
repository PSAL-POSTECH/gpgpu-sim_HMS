#python file that make run environment
#get arguments from command line
import sys
import os
import argparse
import math

#used for HMS
NUM_SCM_LAYERS=4
NUM_DRAM_LAYERS=4
DENSITY=4


def make_run_environment(memory, trace, footprintratio, output, configdir):
    #read memory footprint of trace
    size_file_path = os.path.join(trace, 'num_pages.txt')
    if not os.path.exists(size_file_path):
        print("num_pages.txt does not exist")
        sys.exit(1)
    else:
        num_pages = 0
        with open(size_file_path, 'r') as f:
            num_pages = int(f.readline().strip().split()[0])
            print("num_pages: ", num_pages)
    
    #calculate memory size
    num_hold_pages = math.ceil(int(num_pages) * (int(footprintratio) / 100.0))
    
    if "HMS" in memory:
        per_layer_pages = math.ceil(num_hold_pages / (NUM_SCM_LAYERS+NUM_DRAM_LAYERS))
        num_dram_pages = math.ceil(per_layer_pages * NUM_DRAM_LAYERS) + 2
        num_total_pages = math.ceil(per_layer_pages * NUM_SCM_LAYERS * DENSITY) + 2
    elif memory == "PCM":
        num_total_pages = math.ceil(num_hold_pages * DENSITY) + 2
    elif memory == "HBM":
        num_total_pages = num_hold_pages + 2
    
    #make run environment
    
    #make output directory
    #check if output directory exists
    if not os.path.exists(output):
        os.makedirs(output)
    #make directory of run environment (memory/benchmark)
    os.makedirs(os.path.join(output, (memory + "_" + footprintratio)))
    benchmark = trace.split('/')[-2]
    os.makedirs(os.path.join(output, (memory + "_" + footprintratio), benchmark))
    os.makedirs(os.path.join(output, (memory + "_" + footprintratio), benchmark, "output"))
    
    #copy config files to run environment
    config_files = os.listdir(configdir)
    for config_file in config_files:
        os.system("cp " + os.path.join(configdir, config_file) + " " + os.path.join(output, (memory + "_" + footprintratio), benchmark))
    
    #append page size to config file (gpgpusim.config) in run environment
    #check if gpgpusim.config exists
    if not os.path.exists(os.path.join(output, (memory + "_" + footprintratio), benchmark, "gpgpusim.config")):
        print("gpgpusim.config does not exist")
        sys.exit(1)
    else:
        with open(os.path.join(output, (memory + "_" + footprintratio), benchmark, "gpgpusim.config"), 'a') as f:
            f.write("\n")
            if "HMS" in memory:
                f.write("-num_dram_pages " + str(num_dram_pages) + "\n")
            f.write("-num_total_pages " + str(num_total_pages) + "\n")
    
    #return output directory
    return os.path.join(output, (memory + "_" + footprintratio), benchmark)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make run environment")
    parser.add_argument("-m", "--memory", help="memory type")
    parser.add_argument("-t", "--trace", help="trace directory")
    parser.add_argument("-f", "--footprintratio", help="footprint ratio that baseline HBM can hold")
    parser.add_argument("-o", "--output", help="output directory")
    parser.add_argument("-c", "--configdir", help="config directory")
    args = parser.parse_args()
        
    output_dir = make_run_environment(args.memory, args.trace, args.footprintratio, args.output, args.configdir)
    
    with open("run_environment.txt", 'w') as f:
        f.write(output_dir)