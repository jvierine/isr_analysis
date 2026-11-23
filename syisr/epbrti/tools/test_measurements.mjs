import assert from 'node:assert/strict';import {encodeLinear,averageLinear} from '../web/measurements.js';
// Signed values participate in averaging; the logarithm comes afterwards.
let a=averageLinear(new Float32Array([-8,12,NaN,4]),2,2,2,2);assert.ok(Math.abs(a.values[0]-8/3)<1e-6); // f32 roundoff handled below
assert.deepEqual([...encodeLinear(new Float32Array([-1,0,1e10,1e14,1e16,NaN]),'density')],[1,1,1,255,255,0]);
let data=new Float32Array([1e10,1e14]);let mean=averageLinear(data,2,1,2,1);let p=encodeLinear(mean.values,'density');assert.ok(p[0]>230); // A log-space average would be near the scale midpoint.
console.log('SIGNED_LINEAR_AVERAGING_AND_DENSITY_SCALE_VERIFIED');
