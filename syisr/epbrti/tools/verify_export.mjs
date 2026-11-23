// Physical round trips use the actual exported eleven-beam geometry.
import fs from 'node:fs/promises';
import init,{map_hover,cross} from '../web/pkg/epbrti_core.js';
await init({module_or_path:await fs.readFile(new URL('../web/pkg/epbrti_core_bg.wasm',import.meta.url))});
const dir=process.argv[2]||'data/public';const m=JSON.parse(await fs.readFile(`${dir}/manifest.json`));let maxR=0,maxT=0,count=0;
for(let b of m.beams){let raw=await fs.readFile(`${dir}/geom${String(b.id).padStart(2,'0')}.bin`);let geom=new Float32Array(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));let r=Float32Array.from({length:b.nr},(_,i)=>b.r0+i*b.dr);
 for(let j=1;j<20;j++){let range=r[0]+(r.at(-1)-r[0])*j/20,t=m.tmin+300+j*280;let uv=map_hover(range,t,r,geom,.12,0,m.tref);let roots=cross(uv[0],uv[1],r,geom,.12,0,m.tref,m.tmin,m.tmax);if(!roots.length)throw Error(`Missing crossing ${b.id}`);let e=[];for(let k=0;k<roots.length;k+=3)e.push([Math.abs(roots[k]-t),Math.abs(roots[k+1]-range)]);e.sort((a,b)=>a[0]-b[0]);maxT=Math.max(maxT,e[0][0]);maxR=Math.max(maxR,e[0][1]);count++;}
 let measured=await fs.stat(`${dir}/beam${String(b.id).padStart(2,'0')}.bin`);if(measured.size!==b.nt*b.nr)throw Error('Texture dimensions');
}
if(maxR>1e-4||maxT>1e-3)throw Error('Mapping tolerance exceeded');
console.log(JSON.stringify({beam_count:m.beams.length,crossings:count,max_range_residual_km:maxR,max_time_residual_s:maxT,native_texture_dimensions_verified:true}));
const {projectedWindow,centeredBox}=await import('../web/magnetic-view.js');
let centering=0;
for(let b of m.beams){let raw=await fs.readFile(`${dir}/geom${String(b.id).padStart(2,'0')}.bin`);let geom=new Float32Array(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));let ranges=Float32Array.from({length:b.nr},(_,i)=>b.r0+i*b.dr);let beam={meta:b,geom,ranges};for(let u of [-60,0,80]){let v=20,hit=cross(u,v,ranges,geom,.12,0,m.tref,-1e20,1e20);let wide=projectedWindow(beam,centeredBox(u,v,400,240),.12,0,m.tref,cross),tight=projectedWindow(beam,centeredBox(u,v,200,120),.12,0,m.tref,cross);if(!hit.length||!wide||!tight)throw Error('Missing magnetic viewport');if(Math.abs((tight[0]+tight[1])/2-hit[0])>1e-6||Math.abs((tight[2]+tight[3])/2-hit[1])>1e-6)throw Error('Zoom centre is not the mapped field line');if(tight[1]-tight[0]>=wide[1]-wide[0]||tight[3]-tight[2]>=wide[3]-wide[2])throw Error('Magnetic zoom did not narrow');centering++;}}
console.log(JSON.stringify({magnetic_zoom_centres_verified:centering,all_beams_have_individual_time_range_windows:true}));
