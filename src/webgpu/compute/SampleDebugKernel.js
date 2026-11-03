import { pipelineWgslFn } from 'three-mesh-bvh/webgpu';
import { StorageTexture } from 'three/webgpu';
import { ComputeKernel } from './ComputeKernel.js';
import { textureStore, wgslFn, globalId, uniform } from 'three/tsl';
import { SAMPLE_ACTIVE_FLAG, SAMPLE_COUNT_MASK } from '../constants.js';

// Kernel for copying count + active flag to an output target for debug visualizations
export class SampleDebugKernel extends ComputeKernel {

	constructor() {

		const params = {
			globalId: globalId,
			displaySamples: uniform( true ),
			inputTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadOnly(),
			outputTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadWrite(),
		};

		const fn = pipelineWgslFn(58, ["\n\n\t\t\tfn compute(\n\t\t\t\tglobalId: vec3u,\n\t\t\t\tinputTarget: texture_storage_2d<r32uint, read>,\n\t\t\t\toutputTarget: texture_storage_2d<rgba8unorm, read_write>,\n\t\t\t\tdisplaySamples: u32\n\t\t\t) -> void {\n\n\t\t\t\tlet combined = textureLoad( inputTarget, globalId.xy ).r;\n\t\t\t\tlet isActive = ( ","u & combined ) != 0u;\n\t\t\t\tlet samples = combined & ","u;\n\n\t\t\t\tif ( displaySamples != 0 ) {\n\n\t\t\t\t\ttextureStore( outputTarget, globalId.xy, vec4f( f32( samples ) * 0.01, 0, 0, 1.0 ) );\n\n\t\t\t\t} else {\n\n\t\t\t\t\tlet v = f32( isActive );\n\t\t\t\t\ttextureStore( outputTarget, globalId.xy, vec4f( v, v, v, 1.0 ) );\n\n\t\t\t\t}\n\n\t\t\t}\n\t\t"], [SAMPLE_ACTIVE_FLAG, SAMPLE_COUNT_MASK], [])( params );

		super( fn );

		this.defineUniformAccessors( params );

	}

}
