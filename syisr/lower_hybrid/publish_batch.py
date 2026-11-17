"""Copy finished products to the data disk and relocate internal product links."""
import argparse
from pathlib import Path
import shutil
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('scratch',type=Path);p.add_argument('destination',type=Path);q=p.parse_args()
    assert q.scratch.resolve()!=q.destination.resolve()
    q.destination.mkdir(parents=True,exist_ok=True)
    # Reproducible movie frames stay in scratch. Movies, previews, decoded
    # QA frames, scientific data, and manifests are the delivered products.
    shutil.copytree(q.scratch,q.destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('figures'))
    before=str(q.scratch.resolve());after=str(q.destination.resolve())
    def relocate(value):
        if isinstance(value,str):return value.replace(before,after)
        if isinstance(value,bytes):return value.decode().replace(before,after)
        return value
    for path in q.destination.rglob('*.h5'):
        if path.name.startswith('._'):continue # macOS AppleDouble, not scientific data
        with h5py.File(path,'r+') as h:
            def patch(name,obj):
                for key,value in list(obj.attrs.items()):
                    replacement=relocate(value)
                    if isinstance(value,(str,bytes)) and replacement!=value:obj.attrs[key]=replacement
                if isinstance(obj,h5py.Dataset) and h5py.check_string_dtype(obj.dtype) is not None:
                    values=obj.asstr()[()];replacement=np.vectorize(relocate,otypes=[object])(values)
                    if np.any(replacement!=values):obj[...]=replacement
            patch('',h);h.visititems(patch)
    print('Copied products and relocated internal links to',q.destination,flush=True)


if __name__=='__main__':main()
