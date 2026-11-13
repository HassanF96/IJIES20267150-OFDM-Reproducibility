import sys,os
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'revision/scripts'))
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        if writing and not Path(os.fsdecode(args[0])).resolve().is_relative_to(ROOT):raise PermissionError('Clean-room write outside package')
sys.addaudithook(guard)
import audit_data_receiver as audit
assert audit.ROOT==ROOT
audit.receiver()
print('CLEAN_ROOM_RECEIVER_PASS',flush=True)
