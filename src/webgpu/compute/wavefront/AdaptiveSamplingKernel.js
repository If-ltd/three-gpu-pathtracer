import { pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { bindPipelineShader } from './pipeline-shader.js';
import { StorageBufferAttribute, StorageTexture } from 'three/webgpu';
import { storage, textureStore, uniform, globalId } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { pixelQueueStruct } from './structs.js';
import { SAMPLE_COUNT_MASK } from '../../constants.js';

// Called only after EVERY active path reached the preceding stage cap. Thus no
// in-flight path owns a pixel when rebuilding this unique compact work list.
export class AdaptiveSamplingKernel extends ComputeKernel {
 constructor(shaderSource) {
  const params = {
   moments: textureStore(new StorageTexture(1, 1)).toReadWrite(),
   risk: storage(new StorageBufferAttribute(new Uint32Array(1), 1), 'uint'),
   targets: storage(new StorageBufferAttribute(new Uint32Array(1), 1), 'uint'),
   pixelQueue: storage(new StorageBufferAttribute(1, 1), pixelQueueStruct),
   sampleCountTarget: textureStore(new StorageTexture(1, 1)).toReadOnly(),
   finalSamples: uniform(64, 'uint'),
   relativeError: uniform(0.025),
   absoluteError: uniform(0.006),
   globalId,
  };
  const legacy = pipelineWgslTagFn(65)`
   fn compute(finalSamples: u32, relativeError: f32, absoluteError: f32, globalId: vec3u) -> void {
    let dims = textureDimensions(${params.sampleCountTarget});
    if (any(globalId.xy >= dims)) { return; }
    let index = globalId.y * dims.x + globalId.x;
    let count = textureLoad(${params.sampleCountTarget}, globalId.xy).r & ${SAMPLE_COUNT_MASK}u;
    if (${params.targets}[index] <= count) { return; }
    let mean = textureLoad(${params.moments}, vec2u(globalId.x * 2u, globalId.y));
    let m2 = textureLoad(${params.moments}, vec2u(globalId.x * 2u + 1u, globalId.y));
    let error = 1.96 * sqrt(max(m2, vec4f(0.0)) / max(f32(count * (count - 1u)), 1.0));
    let tolerance = vec4f(absoluteError) + relativeError * abs(mean);
    var risk = ${params.risk}[index];
    for (var y = -1; y <= 1; y++) {
     for (var x = -1; x <= 1; x++) {
      let uv = vec2u(clamp(vec2i(globalId.xy) + vec2i(x,y), vec2i(0), vec2i(dims) - vec2i(1)));
      risk = max(risk, ${params.risk}[uv.y * dims.x + uv.x]);
     }
    }
    let floor = select(select(16u, 32u, risk >= 1u), finalSamples, risk >= 2u);
    // Reject non-finite statistics too: NaN comparisons cannot report convergence.
    if (count >= floor && all(error <= tolerance)) {
     ${params.targets}[index] = count;
     return;
    }
    let queue = &${params.pixelQueue};
    let position = atomicAdd(&queue.elementCount, 1u);
    atomicStore(&queue.elements[position], (globalId.x << 16u) | globalId.y);
   }
  `;
        const fn = shaderSource === undefined ? legacy : bindPipelineShader(shaderSource, { sampleCountTarget: params.sampleCountTarget, sampleCountMask: SAMPLE_COUNT_MASK, targets: params.targets, moments: params.moments, risk: params.risk, pixelQueue: params.pixelQueue });
  super(fn(params)); this.defineUniformAccessors(params);
 }
}
