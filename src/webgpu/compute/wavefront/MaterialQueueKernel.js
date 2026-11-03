import { pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { bindPipelineShader } from './pipeline-shader.js';
import { StorageBufferAttribute } from 'three/webgpu';
import { storage, uniform, globalId } from 'three/tsl';
import { proxy, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { rayDataStruct } from './structs.js';
import { createPathStateAccessors } from './path-state.js';

export const MATERIAL_QUEUE_COUNT = 4;

// Coarse ordering only: exact PBR evaluation, including maps and mixed lobes,
// remains in MaterialKernel. Every path appears once, even dormant/recycled ones.
// No mesh names, object omissions, or altered material parameters are involved.
export class MaterialQueueKernel extends ComputeKernel {
    constructor(soa = false, shaderSource) {
        const params = {
            bvhData: { value: null },
            rayDataStorage: storage(new StorageBufferAttribute(1, 1), soa ? 'uint' : rayDataStruct),
            workQueue: storage(new StorageBufferAttribute(new Uint32Array(1), 1), 'uint').toAtomic(),
            rayCount: uniform(0, 'uint'), globalId,
        };
        const path = createPathStateAccessors(params.rayDataStorage, soa);
        const legacy = pipelineWgslTagFn(79)`
            fn classifyMaterial(rayCount: u32, globalId: vec3u) -> void {
                let index = globalId.x;
                if (index >= rayCount) { return; }
                let objectIndex = ${path.read.objectIndex}(index);
                let transforms = &${proxy('bvhData.value.storage.transforms', params)};
                let materials = &${proxy('bvhData.value.storage.materials', params)};
                var group = 0u;
                if (objectIndex >= 0) {
                    let material = materials[transforms[u32(objectIndex)].materialIndex];
                    group = 3u;
                    if (material.transmission > 0.0) { group = 1u; }
                    else if (material.metalness > 0.0) { group = 2u; }
                }
                let queue = &${params.workQueue};
                let capacity = (arrayLength(queue) - ${MATERIAL_QUEUE_COUNT}u) / ${MATERIAL_QUEUE_COUNT}u;
                let position = atomicAdd(&queue[group], 1u);
                atomicStore(&queue[${MATERIAL_QUEUE_COUNT}u + group * capacity + position], index);
            }
        `;
        const fn = shaderSource === undefined ? legacy : bindPipelineShader(shaderSource, { readObjectIndex: path.read.objectIndex, transforms: proxy('bvhData.value.storage.transforms', params), materials: proxy('bvhData.value.storage.materials', params), workQueue: params.workQueue, queueCount: MATERIAL_QUEUE_COUNT });
        super(fn(params)); this.defineUniformAccessors(params);
    }
}
