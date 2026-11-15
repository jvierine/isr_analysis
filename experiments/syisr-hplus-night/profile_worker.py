"""Process orchestration only; every likelihood optimization is native Rust."""
import concurrent.futures,json,os,pathlib,subprocess,time
r=pathlib.Path(__file__).resolve().parent
jobs=json.loads((r/'jobs.json').read_text());out=r/'profiles';out.mkdir(exist_ok=True)
def run(j):
 k,h=j;name=f'k{k}-h{h:.2f}';p=out/(name+'.json');start=time.time()
 with (out/(name+'.log')).open('w') as log:
  subprocess.run([str(r/'hplus_coupled_fit'),str(r/'warm-16.json'),str(p),'250',f'profile-{k}-{h:.2f}'],env=dict(os.environ,RAYON_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
 d=json.loads(p.read_text());b=min(d['runs'],key=lambda x:x['chi_square'])
 return dict(knot=k,h_fraction=h,chi_square=b['chi_square'],status=b['status'],normalized_gradient=b['normalized_gradient'],nodes=b['nodes'],host=os.uname().nodename,elapsed_s=time.time()-start,start_objectives=[x['chi_square']for x in d['runs']])
with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,(os.cpu_count()or 2)-1))as pool:
 rows=list(pool.map(run,jobs))
(r/'profile-summary.json').write_text(json.dumps(rows));print('COMPLETE',len(rows),flush=True)
