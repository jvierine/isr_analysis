// Zoom is defined in the co-moving IGRF reference plane. Each beam receives its
// own time/range window centred on the SAME (u,v) field-line label.
export function projectedWindow(beam,box,vx,vy,tref,cross){
 const u=(box[0]+box[1])/2,v=(box[2]+box[3])/2;
 const hit=(u,v)=>cross(u,v,beam.ranges,beam.geom,vx,vy,tref,-1e20,1e20);
 const center=hit(u,v);if(center.length!==3)return null;
 const tc=center[0],rc=center[1];let rLo=rc,rHi=rc,timeShear=0;
 // Sampling edges also handles curvature and partial beam coverage. No ray is extrapolated.
 for(let j=0;j<=32;j++){
  const a=hit(u,box[2]+(box[3]-box[2])*j/32);if(a.length!==3)continue;
  rLo=Math.min(rLo,a[1]);rHi=Math.max(rHi,a[1]);timeShear=Math.max(timeShear,Math.abs(a[0]-tc));
 }
 const hr=Math.max(beam.meta.dr,(rc-rLo),(rHi-rc));
 const ht=Math.max(beam.meta.dt,(box[1]-box[0])/(2*Math.abs(vx))+timeShear);
 return [tc-ht,tc+ht,rc-hr,rc+hr];
}
export function centeredBox(u,v,width,height){return [u-width/2,u+width/2,v-height/2,v+height/2];}
