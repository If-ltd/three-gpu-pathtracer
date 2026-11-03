import { pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { bindPipelineShader } from './pipeline-shader.js';
import { createPathStateAccessors } from './path-state.js';
import { StorageBufferAttribute } from 'three/webgpu';
import { uniform, storage, globalId } from 'three/tsl';
import { ComputeKernel } from '../ComputeKernel.js';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rayDataStruct, pixelQueueStruct } from './structs.js';

// Moves pixels between a range of path slots and the queue when the pool is resized. Slots added
// by a grow take a pixel from the queue tail and start idle, as MaterialKernel expects. Slots
// dropped by a shrink push their pixel onto the tail before the pool is discarded. Runs between
// frames, so nothing else touches the queue meanwhile.
export class ResetSlotsKernel extends ComputeKernel {

	constructor(soa = false, shaderList) {

		const params = {
			rayDataStorage: storage( new StorageBufferAttribute( 1, 1 ), soa ? 'uint' : rayDataStruct ),
			pixelQueue: storage( new StorageBufferAttribute( 1, 1 ), pixelQueueStruct ),
			start: uniform( 0, 'uint' ),
			end: uniform( 0, 'uint' ),
			addingSlots: uniform( 0, 'uint' ),
			globalId: globalId,
		};

        const path = createPathStateAccessors(params.rayDataStorage, soa);
		const fn = (shaderList === undefined ? pipelineWgslTagFn(84)`
			fn compute( start: u32, end: u32, addingSlots: u32, globalId: vec3u ) -> void {

				
				let pixelQueue = &${ params.pixelQueue };

				let index = start + globalId.x;
				if ( index >= end ) {

					return;

				}

				if ( addingSlots == 0u ) {

					// slots being removed hand their pixel back before the pool is discarded
					let queueIndex = atomicAdd( &pixelQueue.elementCount, 1u );
					atomicStore( &pixelQueue.elements[ queueIndex ], ${path.read.pixelIndex}(index) );

				} else {

					// the pool never exceeds the pixel count, so the queue holds enough for every new slot
					let previousCount = atomicSub( &pixelQueue.elementCount, 1u );
					${path.write.pixelIndex}(index, atomicLoad( &pixelQueue.elements[ previousCount - 1u ] ));
					${path.write.resultColor}(index, vec4f( 0.0 ));
					${path.write.throughputColor}(index, vec3f( 0.0 ));
					${path.write.objectIndex}(index, - 1);
					${path.write.alphaDepth}(index, 0u);
					${path.write.rayIntersectionIndex}(index, - 1);
					${path.write.shadowRayIntersectionIndex}(index, - 1);

				}

			}
		` : bindPipelineShader(shaderList.resetSlots, { pixelQueue: params.pixelQueue, readPixelIndex: path.read.pixelIndex, writePixelIndex: path.write.pixelIndex, writeResultColor: path.write.resultColor, writeThroughputColor: path.write.throughputColor, writeObjectIndex: path.write.objectIndex, writeAlphaDepth: path.write.alphaDepth, writeRayIntersectionIndex: path.write.rayIntersectionIndex, writeShadowRayIntersectionIndex: path.write.shadowRayIntersectionIndex }));

		super( fn( params ) );

		this.defineUniformAccessors( params );

	}

}
