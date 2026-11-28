"""Scientific I/O invariants: physical stride, clock quarantine, unsupported mode."""
import pathlib
import struct
import tempfile
from process import frames,layout


def fixture(j,clock_fault=False):
    b=bytearray(1024+149440*8)
    struct.pack_into('<I',b,0,0xaaaa5555);struct.pack_into('<I',b,1020,0xaa5555aa)
    b[12:18]=bytes([11,26,8,25,54+int(clock_fault),59])
    struct.pack_into('<H',b,18,1000+j*400);b[36]=0x35;b[764]=1
    for o,v in [(668,9),(766,333),(936,50),(942,j//2),(980,10000),(988,320)]: struct.pack_into('<H',b,o,v)
    struct.pack_into('<I',b,768,149436)
    return b


with tempfile.TemporaryDirectory() as directory:
    p=pathlib.Path(directory)/'fixture.dat';p.write_bytes(b''.join(fixture(j,j in (2,3)) for j in range(6)))
    h=layout(p);assert h['extra_complex_samples']==4 and h['stride']==1196544
    exclusions=[];result=list(frames([p],h['time'],h['time']+60,h,exclusions))
    assert len(result)==4 and len(exclusions)==2
    assert [x[1] for x in exclusions]==[2*h['stride'],3*h['stride']]
    assert [round((r[0]['time']-h['time'])*100) for r in result]==[0,1,4,5]
    raw=bytearray(p.read_bytes())
    for j in range(6):raw[j*h['stride']+34]=30 if j%2==0 else 70
    p.write_bytes(raw)
    low=layout(p);high=dict(low,receiver_center_hz=444e6)
    lower=list(frames([p],low['time'],low['time']+60,low,[]));upper=list(frames([p],low['time'],low['time']+60,high,[]))
    assert [x[3] for x in lower]==[0,4*h['stride']]
    assert [x[3] for x in upper]==[h['stride'],5*h['stride']]
    assert all(x[0]['receiver_center_hz']==436e6 for x in lower)
    assert all(x[0]['receiver_center_hz']==444e6 for x in upper)
    raw=bytearray(p.read_bytes());raw[36]=0x34;p.write_bytes(raw)
    try: layout(p)
    except ValueError: pass
    else: raise AssertionError('unsupported mode accepted')
print('PASS: four extra IQ pairs; bad clock fragment excluded without time repair; exact offsets; unsupported mode fails')
