import { pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { StorageTexture } from 'three/webgpu';
import { ComputeKernel } from './ComputeKernel.js';
import { textureStore, globalId } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';

export class ZeroOutKernel extends ComputeKernel {

	constructor() {

		const params = {
			globalId: globalId,
			outputTarget: textureStore( new StorageTexture( 1, 1 ) ).toWriteOnly(),
		};

		const fn = pipelineWgslTagFn(64)`
			fn compute( globalId: vec3u ) -> void {

				textureStore( ${ params.outputTarget }, globalId.xy, vec4( 0, 0, 0, 1 ) );

			}
		`;

		super( fn( params ) );

		this.defineUniformAccessors( {
			target: params.outputTarget,
		} );

	}

}
