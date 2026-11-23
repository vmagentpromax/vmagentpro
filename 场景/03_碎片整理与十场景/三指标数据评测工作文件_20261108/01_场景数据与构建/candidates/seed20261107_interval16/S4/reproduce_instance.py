from pathlib import Path
import sys
here=Path(__file__).resolve().parent
root=next(p for p in here.parents if (p/'source_audit').exists() and (p/'engine.py').exists())
sys.path.insert(0,str(root))
import argparse
from reproduce import instance
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();instance(root,here,args.output.resolve())
