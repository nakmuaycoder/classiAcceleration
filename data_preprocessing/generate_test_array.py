"""
Generate test array

$python3 generate_test_array.py nothing_path, walk_path, run_path, output_path, observations

author: nakmuaycoder
date: 01/2021

"""

import argparse
import os
from random import randint

def generate_test(nothing_path, walk_path, run_path, output_path, observations):
    with open(nothing_path, "r") as f:
        nothing = f.readlines()[1:]

    with open(walk_path, "r") as f:
        walk = f.readlines()[1:]
    
    with open(run_path, "r") as f:
        run = f.readlines()[1:]
    
    with open(output_path, "w") as f:
        for source in [("nothing", nothing), ("walk", walk), ("run", run)]:
            first_line = randint(0, len(source[1])- observations)
            f.write("\n\nfloat {}[] = {{".format(source[0]))
            for i in range(observations):
                f.write("\n\t")
                z = 0
                for _ in source[1][i + first_line].split(",")[-3:]:
                    z += 1
                    if z == 3 and i == observations-1:
                        f.write(_.replace("\n", ""))
                    else:
                        f.write(_.replace("\n", "") + ",")
                    
            f.write("\n};\n\n")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation of arrays_test.h")
    parser.add_argument("nothing_path", type=str, help="path of the nothing acceleration", default=None)
    parser.add_argument("walk_path", type=str, help="path of the walking acceleration", default=None)
    parser.add_argument("run_path", type=str, help="Path of the running acceleration", default=None)
    parser.add_argument("output_path", type=str, help="Path of output", default=os.path.join(os.getcwd(), "arrays_test.h"))
    parser.add_argument("observations", type=int, help="Number of observation", default=60)
    args = parser.parse_args()

    generate_test(args.nothing_path, args.walk_path, args.run_path, args.output_path, args.observations)
