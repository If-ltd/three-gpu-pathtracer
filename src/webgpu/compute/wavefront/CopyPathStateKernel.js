import { pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { bindPipelineShader } from './pipeline-shader.js';
import { StorageBufferAttribute } from 'three/webgpu';
import { uniform, storage, globalId } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { PATH_STATE_WORDS } from './path-state.js';

// A capacity change changes the stride of every component plane. Preserve live
// paths at the same logical slot instead of copying an interleaved prefix.
export class CopyPathStateKernel extends ComputeKernel {
    constructor(shaderSource) {
        const params = {
            source: storage(new StorageBufferAttribute(1, 1), 'uint').toReadOnly(),
            target: storage(new StorageBufferAttribute(1, 1), 'uint'),
            count: uniform(0, 'uint'), globalId,
        };
        const legacy = pipelineWgslTagFn(66)`
            fn copyPathState(count: u32, globalId: vec3u) -> void {
                let index = globalId.x;
                if (index >= count) { return; }
                let source = &${params.source}; let destination = &${params.target};
                let sourceStride = arrayLength(source) / ${PATH_STATE_WORDS}u;
                let targetStride = arrayLength(destination) / ${PATH_STATE_WORDS}u;
                for (var word = 0u; word < ${PATH_STATE_WORDS}u; word++) {
                    destination[word * targetStride + index] = source[word * sourceStride + index];
                }
            }
        `;
        const fn = shaderSource === undefined ? legacy : bindPipelineShader(shaderSource, { source: params.source, target: params.target, pathStateWords: PATH_STATE_WORDS });
        super(fn(params)); this.defineUniformAccessors(params);
    }
}
