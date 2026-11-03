import { pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { bindPipelineShader } from './pipeline-shader.js';
import { createPathStateAccessors } from './path-state.js';
import { Vector2 } from 'three';
import { StorageBufferAttribute } from 'three/webgpu';
import { uniform, storage, globalId } from 'three/tsl';
import { ComputeKernel } from '../ComputeKernel.js';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rayDataStruct, pixelQueueNonAtomicStruct } from './structs.js';

// Runs once per reset: assigns each of the first rayData-pool-count pixels to a path slot and parks
// the overflow pixel indices in the pixel queue. Slots are initialized so LogicKernel skips them and
// MaterialKernel immediately generates fresh camera rays. Later budget changes are handled by
// resizing the pool in the render loop, with ResetSlotsKernel returning retired slots' pixels.
export class PopulatePixelIndicesKernel extends ComputeKernel {

	constructor(adaptiveSampling = false, soa = false, shaderList) {

		const params = {
			rayDataStorage: storage( new StorageBufferAttribute( 1, 1 ), soa ? 'uint' : rayDataStruct ),
			pixelQueue: storage( new StorageBufferAttribute( 1, 1 ), pixelQueueNonAtomicStruct ),
			targetDimensions: uniform( new Vector2() ),
			frameBudget: uniform( 0, 'uint' ),
			risk: storage( new StorageBufferAttribute( new Uint32Array( 1 ), 1 ), 'uint' ),
			targets: storage( new StorageBufferAttribute( new Uint32Array( 1 ), 1 ), 'uint' ),
			finalSamples: uniform( 0, 'uint' ),
			globalId: globalId,
		};

        const initializeAdaptive = adaptiveSampling ? (shaderList === undefined ? pipelineWgslTagFn(80)`
            fn initializeAdaptive(index: u32, samples: u32) -> void {
                ${params.risk}[index] = 0u; ${params.targets}[index] = samples;
            }
        ` : bindPipelineShader(shaderList.initializeAdaptiveTargets, { risk: params.risk, targets: params.targets })) : (shaderList === undefined ? pipelineWgslTagFn(81)`fn initializeAdaptive(index: u32, samples: u32) -> void {}` : bindPipelineShader(shaderList.skipAdaptiveTargets, {  }));
        const path = createPathStateAccessors(params.rayDataStorage, soa);
		const fn = (shaderList === undefined ? pipelineWgslTagFn(82)`
			fn compute( targetDimensions: vec2u, frameBudget: u32, finalSamples: u32, globalId: vec3u ) -> void {

				
				let pixelQueue = &${ params.pixelQueue };

				if ( globalId.x >= targetDimensions.x || globalId.y >= targetDimensions.y ) {

					return;

				}

				let rayCount = min( frameBudget, ${path.capacity}() );
				let pixelCount = targetDimensions.x * targetDimensions.y;
				if ( globalId.x == 0u && globalId.y == 0u ) {

					pixelQueue.current = 0u;
					pixelQueue.elementCount = select( 0u, pixelCount - rayCount, pixelCount >= rayCount );

				}

				let pixelHash = ( globalId.x << 16 ) | globalId.y;
				let pixelIndex = globalId.x + globalId.y * targetDimensions.x;
                ${initializeAdaptive}(pixelIndex, finalSamples);
				if ( pixelIndex < rayCount ) {

					${path.write.pixelIndex}(pixelIndex, pixelHash);
					${path.write.resultColor}(pixelIndex, vec4f( 0.0 ));
					${path.write.throughputColor}(pixelIndex, vec3f( 0.0 ));
					${path.write.objectIndex}(pixelIndex, - 1);
					${path.write.alphaDepth}(pixelIndex, 0u);
					${path.write.rayIntersectionIndex}(pixelIndex, - 1);
					${path.write.shadowRayIntersectionIndex}(pixelIndex, - 1);

				} else {

					pixelQueue.elements[ pixelIndex - rayCount ] = pixelHash;

				}

			}
		` : bindPipelineShader(shaderList.populatePixels, { pixelQueue: params.pixelQueue, capacity: path.capacity, initializeAdaptive: initializeAdaptive, writePixelIndex: path.write.pixelIndex, writeResultColor: path.write.resultColor, writeThroughputColor: path.write.throughputColor, writeObjectIndex: path.write.objectIndex, writeAlphaDepth: path.write.alphaDepth, writeRayIntersectionIndex: path.write.rayIntersectionIndex, writeShadowRayIntersectionIndex: path.write.shadowRayIntersectionIndex }));

		super( fn( params ) );

		this.defineUniformAccessors( params );

	}

}
