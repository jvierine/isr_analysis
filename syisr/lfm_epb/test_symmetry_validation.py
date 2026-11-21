"""Independent synthetic coefficient recovery, including held-out observations."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import numpy as np
from scipy import sparse
# Numerical unit tests need no MPI launcher or MPI installation.
try:import mpi4py
except ImportError:sys.modules['mpi4py']=types.SimpleNamespace(MPI=None)
spec=importlib.util.spec_from_file_location('symmetry_validation',Path(__file__).with_name('symmetry_validation.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Recovery(unittest.TestCase):
    def test_separate_along_field_and_time_terms(self):
        old=m.N
        try:
            m.N=12;rng=np.random.default_rng(42);n=600;row=np.arange(n);cell=rng.integers(m.N,size=n)
            A=sparse.csr_matrix((np.ones(n),(row,cell)),shape=(n,m.N))
            S=A.multiply(rng.uniform(-1,1,n)[:,None]).tocsr();T=A.multiply(rng.uniform(-1,1,n)[:,None]).tocsr()
            y=A@rng.normal(size=m.N)+S@rng.normal(size=m.N)+T@rng.normal(size=m.N)
            models=m.matrices(A,S,T);train=row%3!=0;test=~train;errors=[]
            for X in models:
                x,res=m.solve(X,y,np.full(n,.001),train,0,sparse.csr_matrix((m.N,m.N)))
                errors.append(np.sqrt(np.mean((X[test]@x-y[test])**2)))
                self.assertLess(res,1e-5)
            self.assertLess(errors[3],1e-5)
            self.assertGreater(min(errors[:3]),.1)
        finally:m.N=old
if __name__=='__main__':unittest.main()
