from pathlib import Path
import sys
here=Path(__file__).resolve().parent
root=next(p for p in here.parents if (p/'source_audit').exists() and (p/'engine.py').exists())
sys.path.insert(0,str(root))
from verify_suite import verify
print(verify(root,here)['status'])
