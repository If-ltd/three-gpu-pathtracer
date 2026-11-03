import { pipelineWgslFn } from 'three-mesh-bvh/webgpu';
import { wgslFn } from 'three/tsl';

export const resetResultFn = pipelineWgslFn(195, ["\n\n\tfn resetBuffers(\n\t\tresultBuffer: ptr<storage, array<vec4f>, read_write>,\n\t\tsample_count_buffer: ptr<storage, array<u32>, read_write>,\n\t\tdimensions: vec2u,\n\n\t\tglobalId: vec2u,\n\t) -> void {\n\n\t\tlet offset = globalId.x + globalId.y * dimensions.x;\n\t\tsample_count_buffer[offset] = 0;\n\t\tresultBuffer[offset] = vec4f(0.0);\n\n\t}\n\n"], [], []);

export default resetResultFn;
