import { pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { bindPipelineShader } from './pipeline-shader.js';
import { createHasContributionFn } from './zero-contribution.js';
import { createPathStateAccessors } from './path-state.js';
import { Vector2 } from 'three';
import { StorageBufferAttribute, StorageTexture } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { uniform, storage, textureStore, globalId } from 'three/tsl';
import { proxy, proxyFn, rayStruct, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rngInit, rand1, rand2, RNG_INDEX_RAY_JITTER, RNG_INDEX_ALPHA_TEST, RNG_INDEX_RUSSIAN_ROULETTE, RNG_INDEX_DISPERSION_WAVELENGTH } from '../../nodes/random.wgsl.js';
import { rayDataStruct, rayQueueAtomicStruct, pixelQueueStruct } from './structs.js';
import { SAMPLE_ACTIVE_FLAG, SAMPLE_COUNT_MASK, SAMPLE_DISPATCHED_FLAG } from '../../constants.js';
import { applyDispersionFunc, dispersionColorWeightFunc, DISPERSION_MIN_WAVELENGTH, DISPERSION_MAX_WAVELENGTH, transmissionAttenuationFunc } from '../../nodes/material.wgsl.js';
import { isTerminatingScatterFunc, offsetRayOriginFunc } from '../../nodes/utils.wgsl.js';
import { surfaceRecordStruct } from '../../nodes/structs.wgsl.js';
import { LIGHT_EPSILON } from '../../nodes/lights.wgsl.js';

// Pure material evaluation and ray generation: terminated slots pull a recycled pixel and emit a
// fresh camera ray; live slots evaluate the surface staged by LogicKernel, sample the bsdf, and
// enqueue the next bounce ray plus the NEE shadow ray toward the light LogicKernel selected.
export class MaterialKernel extends ComputeKernel {

	constructor(adaptiveSampling = false, soa = false, sortMaterialPaths = false, pruneZeroContribution = false, shaderList, captureDiffuse = false) {

		const params = {
            workQueue: storage(new StorageBufferAttribute(new Uint32Array(1), 1), 'uint'),
			bvhData: { value: null },
			material: { value: null },

			risk: storage( new StorageBufferAttribute( new Uint32Array( 1 ), 1 ), 'uint' ),
			seed: uniform( 0, 'uint' ),
			targetDimensions: uniform( new Vector2() ),
			maxSamples: uniform( 0, 'uint' ),
			rayCount: uniform( 0, 'uint' ),
			filterGlossy: uniform( 1 ),
			maxTransparentBounces: uniform( 5, 'uint' ),
			maxBounces: uniform( 5, 'uint' ),

			sampleCountTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadWrite(),

			rayDataStorage: storage( new StorageBufferAttribute( 1, 1 ), soa ? 'uint' : rayDataStruct ),
			rayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueAtomicStruct ),
			shadowRayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueAtomicStruct ),
			pixelQueue: storage( new StorageBufferAttribute( 1, 1 ), pixelQueueStruct ),

			globalId: globalId,
		};

		const getCameraRayFn = proxyFn( 'bvhData.value.fns.getCameraRay', params );
		const sampleTrianglePointFn = proxyFn( 'bvhData.value.fns.sampleTrianglePoint', params );
		const getSurfaceRecordFn = proxyFn( 'bvhData.value.fns.getSurfaceRecord', params );
		const bsdfSampleFn = proxyFn( 'material.value.bsdfSample', params );
		const bsdfEvalPdfFn = proxyFn( 'material.value.bsdfEvalPdf', params );

		const recordRisk = adaptiveSampling ? (shaderList === undefined ? pipelineWgslTagFn(74)`
            fn recordRisk(indexUV: vec2u, width: u32, risk: u32) -> void {
                let index = indexUV.y * width + indexUV.x;
                ${params.risk}[index] = max(${params.risk}[index], risk);
            }
        ` : bindPipelineShader(shaderList.recordRisk, { risk: params.risk })) : (shaderList === undefined ? pipelineWgslTagFn(75)`fn recordRisk(indexUV: vec2u, width: u32, risk: u32) -> void {}` : bindPipelineShader(shaderList.skipRisk, {  }));

        const path = createPathStateAccessors(params.rayDataStorage, soa);
		const workIndex = sortMaterialPaths ? (shaderList === undefined ? pipelineWgslTagFn(76)`
            fn materialWorkIndex(workIndex: u32) -> u32 {
                let queue = &${params.workQueue};
                let capacity = (arrayLength(queue) - 4u) / 4u;
                var offset = workIndex;
                for (var group = 0u; group < 4u; group++) {
                    if (offset < queue[group]) { return queue[4u + group * capacity + offset]; }
                    offset -= queue[group];
                }
                return workIndex;
            }
        ` : bindPipelineShader(shaderList.materialWorkIndex, { workQueue: params.workQueue })) : (shaderList === undefined ? pipelineWgslTagFn(77)`fn materialWorkIndex(workIndex: u32) -> u32 { return workIndex; }` : bindPipelineShader(shaderList.directMaterialWorkIndex, {  }));
        const hasContribution = createHasContributionFn(pruneZeroContribution);
        const diffuseCoefficient = captureDiffuse ? wgslTagFn`
            fn diffuseCoefficient(coefficient: vec3f, surface: ${surfaceRecordStruct}, bounce: u32) -> vec3f {
                if (bounce != 0u || surface.metalness > 0.0 || surface.transmission > 0.0 || surface.opacity < 1.0) { return vec3f(0.0); }
                return coefficient;
            }
        ` : wgslTagFn`fn diffuseCoefficient(coefficient: vec3f, surface: ${surfaceRecordStruct}, bounce: u32) -> vec3f { return vec3f(0.0); }`;
        const fn = (shaderList === undefined ? pipelineWgslTagFn(78)`

			fn compute(
				seed: u32,
				targetDimensions: vec2u,
				maxSamples: u32,
				rayCount: u32,
				filterGlossy: f32,
				maxTransparentBounces: u32,
				maxBounces: u32,

				globalId: vec3u
			) -> void {


				let rayQueue = &${ params.rayQueue };
				let shadowRayQueue = &${ params.shadowRayQueue };
				let pixelQueue = &${ params.pixelQueue };

				let materials = &${ proxy( 'bvhData.value.storage.materials', params ) };
				let transforms = &${ proxy( 'bvhData.value.storage.transforms', params ) };

				// bound by "rayCount" rather than the pool length. The dispatch rounds up to the
				// workgroup size and those extra slots hold a zeroed pixel index
				let workItem = globalId.x;
				if ( workItem >= rayCount ) {

					return;

				}

				let index = ${ workIndex }(workItem);
                let input = ${ path.load }(index);
				if ( input.objectIndex < 0 ) {

					// the slot's path has terminated: recycle the pixel through the overflow queue and
					// generate a fresh camera ray
					var pixelIndex = input.pixelIndex;
					let elementCount = atomicLoad( &pixelQueue.elementCount );
					if ( elementCount > 0u ) {

						// TODO: If we've pulled off a pixel that's already finished we currently just
						// write a no-op ray, wasting a frame. It may be better to iterate over a few
						// points in the queue to see if we can find one we can use.
						let queueIndex = atomicAdd( &pixelQueue.current, 1u ) % elementCount;
						pixelIndex = atomicExchange( &pixelQueue.elements[ queueIndex ], pixelIndex );

					}

					let indexUV = vec2u( pixelIndex >> 16, pixelIndex & 0xFFFF );

					// skip the pixel if it has hit the sample limit
					let combinedField = textureLoad( ${ params.sampleCountTarget }, indexUV ).r;
					let samples = ( ${ SAMPLE_COUNT_MASK }u & combinedField );
					let isComplete = maxSamples != 0u && samples >= maxSamples;

					if ( isComplete ) {

						${ path.write.pixelIndex }(index, pixelIndex);
						${ path.write.rayIntersectionIndex }(index, - 1);
						${ path.write.shadowRayIntersectionIndex }(index, - 1);
						return;

					}

					${ rngInit }( indexUV, seed + samples, 0 );

					let uv = vec2f( indexUV ) / vec2f( targetDimensions );
					let jitteredUv = uv + ${ rand2 }( ${ RNG_INDEX_RAY_JITTER } ) / vec2f( targetDimensions );
					var ray: ${ rayStruct };
					if ( ! ${ getCameraRayFn }( jitteredUv, vec2f( targetDimensions ), &ray ) ) {

						// the camera declined the pixel, so leave the slot dormant for this round
						// TODO: same as above - this work is a bit wasteful and leaves slots empty for
						// a frame. Is it possible to quickly skip rays outside the mask or are finished?
						${ path.write.pixelIndex }(index, pixelIndex);
						${ path.write.rayIntersectionIndex }(index, - 1);
						${ path.write.shadowRayIntersectionIndex }(index, - 1);
						return;

					}

					ray.direction = normalize( ray.direction );

					let rayIndex = atomicAdd( &rayQueue.length, 1u );
					rayQueue.elements[ rayIndex ].origin = ray.origin;
					rayQueue.elements[ rayIndex ].direction = ray.direction;
					rayQueue.elements[ rayIndex ].pixelIndex = pixelIndex;
					rayQueue.elements[ rayIndex ].currentBounce = 0u;
					rayQueue.elements[ rayIndex ].seed = seed + samples;
					rayQueue.elements[ rayIndex ].alphaDepth = 0u;
					rayQueue.elements[ rayIndex ].maxDist = ray.maxDist;

					${ path.write.origin }(index, ray.origin);
					${ path.write.direction }(index, ray.direction);
					${ path.write.pixelIndex }(index, pixelIndex);
					${ path.write.seed }(index, seed + samples);
					${ path.write.currentBounce }(index, 0u);
					${ path.write.throughputColor }(index, vec3f( 1.0 ));
                    ${ path.write.diffuseThroughput }(index, vec3f(0.0));
                    ${ path.write.diffuseResult }(index, vec3f(0.0));
                    ${ path.write.diffuseScatter }(index, vec3f(0.0));
                    ${ path.write.diffuseLight }(index, vec3f(0.0));
					${ path.write.resultColor }(index, vec4f( 0.0, 0.0, 0.0, 1.0 ));
					${ path.write.scatterColor }(index, vec3f( 1.0 ));
					${ path.write.scatterPdf }(index, 1.0);
					${ path.write.minPdf }(index, 1.0);
					${ path.write.isFullyTransmissive }(index, 1u);
					${ path.write.emission }(index, vec3f( 0.0 ));
					${ path.write.lightPdf }(index, 0.0);
					${ path.write.alphaDepth }(index, 0u);
					${ path.write.maxDist }(index, ray.maxDist);
					${ path.write.rayIntersectionIndex }(index, i32( rayIndex ));
					${ path.write.shadowRayIntersectionIndex }(index, - 1);
					${ path.write.dispersionWavelength }(index, - mix( ${ DISPERSION_MIN_WAVELENGTH }.0, ${ DISPERSION_MAX_WAVELENGTH }.0, ${ rand1 }( ${ RNG_INDEX_DISPERSION_WAVELENGTH } ) ));

					// write the active params & dispatched flag
					textureStore( ${ params.sampleCountTarget }, indexUV, vec4( ${ SAMPLE_ACTIVE_FLAG }u | ${ SAMPLE_DISPATCHED_FLAG }u | samples ) );

				} else {

					// evaluate the surface staged by LogicKernel
					let indexUV = vec2u( input.pixelIndex >> 16, input.pixelIndex & 0xFFFF );
					${ rngInit }( indexUV, input.seed, input.currentBounce + input.alphaDepth );

					let objectInfo = transforms[ u32( input.objectIndex ) ];
					var materialInfo = ( *materials )[ objectInfo.materialIndex ];

					// a matte surface hit by the camera ray renders as a fully transparent
					let isMatte = materialInfo.matte != 0 && input.currentBounce == 0u;
					if ( isMatte ) {

						${ path.write.resultColor }(index, vec4f( 0.0 ));
						${ path.write.throughputColor }(index, vec3f( 0.0 ));
						${ path.write.emission }(index, vec3f( 0.0 ));
						${ path.write.lightPdf }(index, 0.0);
						${ path.write.shadowRayIntersectionIndex }(index, - 1);
						return;

					}

					// apply per-object colors
					materialInfo.color *= objectInfo.color.rgb;
					materialInfo.opacity *= objectInfo.color.a;

					var vertexData = ${ sampleTrianglePointFn }( input.barycoord, input.indices );
					vertexData.normal = normalize( transpose( objectInfo.inverseMatrixWorld ) * vertexData.normal );
					vertexData.tangent = vec4f( ( objectInfo.matrixWorld * vec4f( vertexData.tangent.xyz, 0.0 ) ).xyz, vertexData.tangent.w );
					vertexData.position = objectInfo.matrixWorld * vertexData.position;

					let view = - input.direction;

					// blur glossy surfaces after low-probability bounces to suppress fireflies,
					// from the Cycles "filter glossy" approach in integrator/surface_shader.h
					let blurRoughness = sqrt( clamp( 1.0 - filterGlossy * input.minPdf, 0.0, 1.0 ) ) * 0.5;

					var surface = ${ getSurfaceRecordFn }( materialInfo, vertexData, input.side, input.normal, view, blurRoughness );

                    if (input.currentBounce == 0u) {
                        // Evaluate maps too. Names, part IDs and model identity play no role.
                        let reflective = surface.metalness > 0.05 || surface.roughness < 0.4 || surface.clearcoat > 0.05;
                        let risk = select(select(0u, 1u, reflective), 2u, surface.transmission > 0.01);
                        ${ recordRisk }(indexUV, targetDimensions.x, risk);
                    }

					// Stochastically pass through partially transparent surfaces by re-enqueueing
					// the ray at the hit point, advancing the alpha depth but not the bounce count.
					let passesThrough = ${ rand1 }( ${ RNG_INDEX_ALPHA_TEST } ) > surface.opacity;
					if ( passesThrough ) {

						// out of transparent bounces, so stop rather than shading a surface that
						// should be invisible. A zeroed throughput terminates in LogicKernel.
						if ( input.alphaDepth >= maxTransparentBounces ) {

							${ path.write.throughputColor }(index, vec3f( 0.0 ));
							${ path.write.emission }(index, vec3f( 0.0 ));
							${ path.write.lightPdf }(index, 0.0);
							${ path.write.shadowRayIntersectionIndex }(index, - 1);
							return;

						}

						let alphaIndex = atomicAdd( &rayQueue.length, 1u );
						rayQueue.elements[ alphaIndex ].origin = ${ offsetRayOriginFunc }( vertexData.position.xyz, input.direction, input.normal );
						rayQueue.elements[ alphaIndex ].direction = input.direction;
						rayQueue.elements[ alphaIndex ].pixelIndex = input.pixelIndex;
						rayQueue.elements[ alphaIndex ].currentBounce = input.currentBounce;
						rayQueue.elements[ alphaIndex ].seed = input.seed;
						rayQueue.elements[ alphaIndex ].alphaDepth = input.alphaDepth + 1u;
						// the origin advanced to the hit point, so the remaining budget shrinks by the
						// distance already traced
						rayQueue.elements[ alphaIndex ].maxDist = max( input.maxDist - input.dist, 0.0 );

						// the surface is skipped, so no scatter or emission is staged for LogicKernel.
						// "pdf" is left alone so the previous scatter still weights the forward MIS,
						// and "bsdf" matches it so applying the scatter leaves the throughput as is.
						${ path.write.alphaDepth }(index, input.alphaDepth + 1u);
						${ path.write.emission }(index, vec3f( 0.0 ));
						${ path.write.scatterColor }(index, vec3f( input.scatterPdf ));
						${ path.write.lightPdf }(index, 0.0);
						${ path.write.origin }(index, rayQueue.elements[ alphaIndex ].origin);
						${ path.write.rayIntersectionIndex }(index, i32( alphaIndex ));
						${ path.write.shadowRayIntersectionIndex }(index, - 1);
						return;

					}

					// apply the hero wavelength to dispersive surfaces, folding the spectral weight
					// into the throughput at the path's first dispersive interaction
					var throughputColor = input.throughputColor;
					let isDispersive = materialInfo.dispersion > 0.0 && surface.ior > 1.0 && surface.transmission > 0.0 && ! surface.thinWall;
					if ( isDispersive ) {

						let wavelength = abs( input.dispersionWavelength );
						${ applyDispersionFunc }( &surface, materialInfo.dispersion, wavelength );
						if ( input.dispersionWavelength < 0.0 ) {

							${ path.write.dispersionWavelength }(index, wavelength);
							throughputColor *= ${ dispersionColorWeightFunc }( wavelength );
							${ path.write.throughputColor }(index, throughputColor);
                        ${ path.write.diffuseThroughput }(index, input.diffuseThroughput * ${ dispersionColorWeightFunc }(wavelength));

						}

					}

					// attenuate the light transmitted through the volume when exiting a backface so
					// the surface's emission and NEE resolve against the attenuated throughput
					if ( input.side < 0.0 && materialInfo.transmission > 0.0 ) {

						throughputColor *= ${ transmissionAttenuationFunc }( input.dist, materialInfo.attenuationColor, materialInfo.attenuationDistance );
						${ path.write.throughputColor }(index, throughputColor);
                        ${ path.write.diffuseThroughput }(index, input.diffuseThroughput * ${ transmissionAttenuationFunc }(input.dist, materialInfo.attenuationColor, materialInfo.attenuationDistance));

					}

					// sample the next bounce direction and stage the scatter state for LogicKernel
					var scatterRec = ${ bsdfSampleFn }( view, surface );
					let newBounce = input.currentBounce + 1u;
                    ${ path.write.diffuseScatter }(index, ${ diffuseCoefficient }(scatterRec.diffuseCoefficient, surface, input.currentBounce));
                    ${ path.write.diffuseLight }(index, vec3f(0.0));

					// decide termination now so finished paths skip the bounce trace entirely - a
					// zeroed pdf reads as a terminating scatter in LogicKernel, which still resolves
					// the surface's emission and NEE before freeing the slot
					var isTerminated = newBounce >= maxBounces || all( scatterRec.color == vec3f( 0.0 ) ) || ${ isTerminatingScatterFunc }( scatterRec );

					// russian roulette early out:
					// Matches Cycles path_state_continuation_probability in integrator/path_state.h
					if ( ! isTerminated && newBounce >= 3u ) {

						let rrThroughput = throughputColor * scatterRec.color / scatterRec.pdf;
						let rrProb = saturate( sqrt( max( max( rrThroughput.r, rrThroughput.g ), rrThroughput.b ) ) );
						isTerminated = rrProb <= 0.0 || ${ rand1 }( ${ RNG_INDEX_RUSSIAN_ROULETTE } ) > rrProb;
						if ( ! isTerminated ) {

							// fold the survival boost into the scatter color so LogicKernel's
							// throughput update applies it without a separate division
							scatterRec.color /= rrProb;

						}

					}

					// Write the ray storage content here since things like the emissive value is read
					// in the logic kernel on the subsequent frame.
					${ path.write.scatterColor }(index, scatterRec.color);
					${ path.write.scatterPdf }(index, select( scatterRec.pdf, 0.0, isTerminated ));
					${ path.write.minPdf }(index, min( input.minPdf, scatterRec.pdf ));
					${ path.write.isFullyTransmissive }(index, input.isFullyTransmissive & select( 0u, 1u, scatterRec.isTransmissive ));
					${ path.write.emission }(index, surface.emission);
					${ path.write.currentBounce }(index, newBounce);

					// the NEE shadow ray below still resolves the surface's direct light, so only
					// the bounce segment is skipped for finished paths
					if ( ! isTerminated ) {

						let rayIndex = atomicAdd( &rayQueue.length, 1u );
						rayQueue.elements[ rayIndex ].origin = ${ offsetRayOriginFunc }( vertexData.position.xyz, scatterRec.direction, input.normal );
						rayQueue.elements[ rayIndex ].direction = scatterRec.direction;
						rayQueue.elements[ rayIndex ].pixelIndex = input.pixelIndex;
						rayQueue.elements[ rayIndex ].currentBounce = newBounce;
						rayQueue.elements[ rayIndex ].seed = input.seed;
						rayQueue.elements[ rayIndex ].alphaDepth = input.alphaDepth;
						rayQueue.elements[ rayIndex ].maxDist = 0.0;
						${ path.write.rayIntersectionIndex }(index, i32( rayIndex ));

						${ path.write.origin }(index, rayQueue.elements[ rayIndex ].origin);
						${ path.write.direction }(index, scatterRec.direction);

					}

					// evaluate the bsdf toward the light LogicKernel selected and enqueue the shadow ray.
					// the light pdf will be 0 if NEE is disabled.
					var lightPdf = input.lightPdf;
                    if (!${ hasContribution }(throughputColor, input.lightEmission, vec3f(1.0))) {
                        lightPdf = 0.0;
                    }
					if ( lightPdf > 0.0 ) {

						let evalRec = ${ bsdfEvalPdfFn }( view, input.lightDirection, surface );
						if ( evalRec.pdf > 0.0 && ${ hasContribution }(throughputColor, input.lightEmission, evalRec.color) ) {

							${ path.write.lightBsdf }(index, evalRec.color);
                            ${ path.write.diffuseLight }(index, ${ diffuseCoefficient }(evalRec.diffuseCoefficient, surface, input.currentBounce));
							${ path.write.lightBsdfPdf }(index, evalRec.pdf);

							let shadowIndex = atomicAdd( &shadowRayQueue.length, 1u );
							shadowRayQueue.elements[ shadowIndex ].origin = ${ offsetRayOriginFunc }( vertexData.position.xyz, input.lightDirection, input.normal );
							shadowRayQueue.elements[ shadowIndex ].direction = input.lightDirection;
							shadowRayQueue.elements[ shadowIndex ].pixelIndex = input.pixelIndex;
							shadowRayQueue.elements[ shadowIndex ].currentBounce = input.currentBounce;
							shadowRayQueue.elements[ shadowIndex ].seed = input.seed;
							shadowRayQueue.elements[ shadowIndex ].alphaDepth = input.alphaDepth;
							shadowRayQueue.elements[ shadowIndex ].maxDist = input.lightDist * ( 1.0 - ${ LIGHT_EPSILON } );
							${ path.write.shadowRayIntersectionIndex }(index, i32( shadowIndex ));

						} else {

							lightPdf = 0.0;

						}

					}

					if ( lightPdf <= 0.0 ) {

						${ path.write.lightPdf }(index, 0.0);
						${ path.write.shadowRayIntersectionIndex }(index, - 1);

					}

				}

			}
` : bindPipelineShader(shaderList.materialScatter, { diffuseCoefficient, writeDiffuseThroughput: path.write.diffuseThroughput, writeDiffuseResult: path.write.diffuseResult, writeDiffuseScatter: path.write.diffuseScatter, writeDiffuseLight: path.write.diffuseLight, rayQueue: params.rayQueue, shadowRayQueue: params.shadowRayQueue, pixelQueue: params.pixelQueue, materials: proxy( 'bvhData.value.storage.materials', params ), transforms: proxy( 'bvhData.value.storage.transforms', params ), workIndex: workIndex, load: path.load, sampleCountTarget: params.sampleCountTarget, sampleCountMask: SAMPLE_COUNT_MASK, writePixelIndex: path.write.pixelIndex, writeRayIntersectionIndex: path.write.rayIntersectionIndex, writeShadowRayIntersectionIndex: path.write.shadowRayIntersectionIndex, rngInit: rngInit, rand2: rand2, rngIndexRayJitter: RNG_INDEX_RAY_JITTER, rayStruct: rayStruct, getCameraRayFn: getCameraRayFn, writeOrigin: path.write.origin, writeDirection: path.write.direction, writeSeed: path.write.seed, writeCurrentBounce: path.write.currentBounce, writeThroughputColor: path.write.throughputColor, writeResultColor: path.write.resultColor, writeScatterColor: path.write.scatterColor, writeScatterPdf: path.write.scatterPdf, writeMinPdf: path.write.minPdf, writeIsFullyTransmissive: path.write.isFullyTransmissive, writeEmission: path.write.emission, writeLightPdf: path.write.lightPdf, writeAlphaDepth: path.write.alphaDepth, writeMaxDist: path.write.maxDist, writeDispersionWavelength: path.write.dispersionWavelength, dispersionMinWavelength: DISPERSION_MIN_WAVELENGTH, dispersionMaxWavelength: DISPERSION_MAX_WAVELENGTH, rand1: rand1, rngIndexDispersionWavelength: RNG_INDEX_DISPERSION_WAVELENGTH, sampleActiveFlag: SAMPLE_ACTIVE_FLAG, sampleDispatchedFlag: SAMPLE_DISPATCHED_FLAG, sampleTrianglePointFn: sampleTrianglePointFn, getSurfaceRecordFn: getSurfaceRecordFn, recordRisk: recordRisk, rngIndexAlphaTest: RNG_INDEX_ALPHA_TEST, offsetRayOriginFunc: offsetRayOriginFunc, applyDispersionFunc: applyDispersionFunc, dispersionColorWeightFunc: dispersionColorWeightFunc, transmissionAttenuationFunc: transmissionAttenuationFunc, bsdfSampleFn: bsdfSampleFn, isTerminatingScatterFunc: isTerminatingScatterFunc, rngIndexRussianRoulette: RNG_INDEX_RUSSIAN_ROULETTE, hasContribution: hasContribution, bsdfEvalPdfFn: bsdfEvalPdfFn, writeLightBsdf: path.write.lightBsdf, writeLightBsdfPdf: path.write.lightBsdfPdf, lightEpsilon: LIGHT_EPSILON }));

		super( fn( params ) );

		this.defineUniformAccessors( params );

	}

}
