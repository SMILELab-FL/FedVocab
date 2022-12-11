import os
import sys

run_dir = "/".join(os.path.abspath(sys.argv[0]).split("/")[0:-3])
sys.path.append(run_dir)

