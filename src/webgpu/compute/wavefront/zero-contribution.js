import { pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';

export function createHasContributionFn(enabled) {
        // Test channels rather than multiplying: preserve tiny nonzero contributions
        // that could otherwise underflow before MIS weighting.
    return enabled ? pipelineWgslTagFn(88)`
            fn hasContribution(throughput: vec3f, emission: vec3f, bsdf: vec3f) -> bool {
                return (throughput.r != 0.0 && emission.r != 0.0 && bsdf.r != 0.0)
                    || (throughput.g != 0.0 && emission.g != 0.0 && bsdf.g != 0.0)
                    || (throughput.b != 0.0 && emission.b != 0.0 && bsdf.b != 0.0);
            }
        ` : pipelineWgslTagFn(89)`fn keepUnprunedContribution(throughput: vec3f, emission: vec3f, bsdf: vec3f) -> bool { return true; }`;
}
