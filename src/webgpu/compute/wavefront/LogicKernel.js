import { pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { bindPipelineShader } from './pipeline-shader.js';
import { createPathStateAccessors } from './path-state.js';
import { StorageBufferAttribute, StorageTexture } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { uniform, storage, textureStore, globalId } from 'three/tsl';
import { proxy, proxyFn, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { misHeuristicFn, weightedAlphaBlendFn } from '../../nodes/sampling.wgsl.js';
import { clampPathContributionFunc, isTerminatingScatterFunc } from '../../nodes/utils.wgsl.js';
import { TRANSMISSIVE_BACKGROUND_ENVIRONMENT, TRANSMISSIVE_BACKGROUND_OVERLAY, TRANSMISSIVE_BACKGROUND_TRANSPARENT } from '../../constants.js';
import {
	rngInit, rand2, rand3,
	RNG_INDEX_BACKGROUND_SAMPLE,
	RNG_INDEX_DIRECT_LIGHT_SAMPLE,
} from '../../nodes/random.wgsl.js';
import { ENVIRONMENT_LIGHT_TYPE, LIGHT_FAR_DISTANCE, isMISWeightLightFn } from '../../nodes/lights.wgsl.js';
import { lightRecordStruct, scatterRecordStruct } from '../../nodes/structs.wgsl.js';
import { rayDataStruct, intersectionResultStruct, rayQueueAtomicStruct } from './structs.js';
import { SAMPLE_COUNT_MASK, SAMPLE_DISPATCHED_FLAG } from '../../constants.js';

// Path logic over the persistent slot pool: resolves the previous frame's shadow and bounce trace
// results, accumulates emission and NEE / forward MIS contributions, terminates finished paths into
// the output, and samples the next NEE light for MaterialKernel. Touches no BVH data.
export class LogicKernel extends ComputeKernel {

	constructor(sparseAccumulation = false, adaptiveSampling = false, soa = false, shaderList, captureDiffuse = false) {

		const params = {
            diffuseTarget: textureStore(new StorageTexture(1, 1)).toReadWrite(),
			envInfo: { value: null },
			backgroundInfo: { value: null },
			lightsInfo: { value: null },

			moments: textureStore(new StorageTexture(1, 1)).toReadWrite(),
			// targets
			outputStorage: storage( new StorageBufferAttribute( 1, 4 ), 'vec4' ),
			prevOutputTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadOnly(),
			outputTarget: sparseAccumulation ? textureStore(new StorageTexture(1, 1)).toReadWrite() : textureStore(new StorageTexture(1, 1)).toWriteOnly(),
			sampleCountTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadWrite(),

			// settings
			misEnabled: uniform( 1, 'uint' ),
			rayCount: uniform( 0, 'uint' ),
			resetRayQueues: uniform( 0, 'uint' ),
			rayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueAtomicStruct ),
			shadowRayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueAtomicStruct ),
			maxBounces: uniform( 5, 'uint' ),
			clampDirect: uniform( 0 ),
			clampIndirect: uniform( 10 ),
			transmissiveBackground: uniform( TRANSMISSIVE_BACKGROUND_OVERLAY ),

			rayDataStorage: storage( new StorageBufferAttribute( 1, 1 ), soa ? 'uint' : rayDataStruct ),
			rayIntersectionsStorage: storage( new StorageBufferAttribute( 1, 1 ), intersectionResultStruct ),
			shadowRayIntersectionsStorage: storage( new StorageBufferAttribute( 1, 1 ), intersectionResultStruct ),

			globalId: globalId,
		};

		// environment + background resources pulled off their providers (embedded functions)
		const envTotalSumNode = proxy( 'envInfo.value.totalSumNode', params );
		const envIntensityNode = proxy( 'envInfo.value.intensityNode', params );
		const sampleEnvColor = proxy( 'envInfo.value.sampleColor', params );
		const sampleEnvDir = proxy( 'envInfo.value.sampleDir', params );
		const getEnvDirPdf = proxy( 'envInfo.value.getDirPdf', params );
		const getEnvWeight = proxy( 'envInfo.value.getWeight', params );
		const sampleBackground = proxy( 'backgroundInfo.value.sampleColor', params );

		// analytic scene lights pulled off the lightsInfo provider (LightsInfoNode)
		const lightsCountNode = proxy( 'lightsInfo.value.countNode', params );
		const randomLightSampleFn = proxyFn( 'lightsInfo.value.randomLightSample', params );
		const intersectLightAtIndexFn = proxyFn( 'lightsInfo.value.intersectLightAtIndex', params );
		const getLightWeightFn = proxyFn( 'lightsInfo.value.getLightWeight', params );
		const getLightsWeightFn = proxyFn( 'lightsInfo.value.getLightsWeight', params );
		const selectLightFn = proxyFn( 'lightsInfo.value.selectLight', params );
		const isLightVisibleToCameraFn = proxyFn( 'lightsInfo.value.isLightVisibleToCamera', params );

		const readPreviousColorFn = sparseAccumulation ? (shaderList === undefined ? pipelineWgslTagFn(67)`
			fn readPreviousColor( indexUV: vec2u, colorIndex: vec2u, storedSamples: u32 ) -> vec4f {

				// Each pixel has one owner in the path pool. A zero sample count
				// invalidates storage from an earlier camera or image reset.
				if ( storedSamples == 0u ) { return vec4f( 0.0 ); }
				let index = indexUV.y * textureDimensions( ${ params.outputTarget } ).x + indexUV.x;
				return textureLoad(${params.outputTarget}, colorIndex);

			}
		` : bindPipelineShader(shaderList.readSparseColor, { outputTarget: params.outputTarget })) : (shaderList === undefined ? pipelineWgslTagFn(68)`
			fn readPreviousColor( indexUV: vec2u, colorIndex: vec2u, storedSamples: u32 ) -> vec4f {

				return textureLoad( ${ params.prevOutputTarget }, colorIndex );

			}
		` : bindPipelineShader(shaderList.readBufferedColor, { prevOutputTarget: params.prevOutputTarget }));
		const storeColorFn = sparseAccumulation ? (shaderList === undefined ? pipelineWgslTagFn(69)`
			fn storeColor( indexUV: vec2u, colorIndex: vec2u, color: vec4f ) -> void {

				let index = indexUV.y * textureDimensions( ${ params.outputTarget } ).x + indexUV.x;
				textureStore( ${ params.outputTarget }, colorIndex, color );

			}
		` : bindPipelineShader(shaderList.storeSparseColor, { outputTarget: params.outputTarget })) : (shaderList === undefined ? pipelineWgslTagFn(70)`
			fn storeColor( indexUV: vec2u, colorIndex: vec2u, color: vec4f ) -> void {

				textureStore( ${ params.outputTarget }, colorIndex, color );

			}
		` : bindPipelineShader(shaderList.storeBufferedColor, { outputTarget: params.outputTarget }));

		const recordMoments = adaptiveSampling ? (shaderList === undefined ? pipelineWgslTagFn(71)`
            fn recordMoments(indexUV: vec2u, count: u32, color: vec4f) -> void {
                let uv = vec2u(indexUV.x * 2u, indexUV.y);
                var mean = vec4f(0.0); var m2 = vec4f(0.0);
                if (count > 1u) { mean = textureLoad(${params.moments}, uv); m2 = textureLoad(${params.moments}, uv + vec2u(1u, 0u)); }
                let delta = color - mean;
                mean += delta / f32(count);
                m2 += delta * (color - mean);
                textureStore(${params.moments}, uv, mean); textureStore(${params.moments}, uv + vec2u(1u, 0u), m2);
            }
        ` : bindPipelineShader(shaderList.recordMoments, { moments: params.moments })) : (shaderList === undefined ? pipelineWgslTagFn(72)`fn recordMoments(indexUV: vec2u, count: u32, color: vec4f) -> void {}` : bindPipelineShader(shaderList.skipMoments, {  }));

        const path = createPathStateAccessors(params.rayDataStorage, soa);
        const storeDiffuse = captureDiffuse ? wgslTagFn`
            fn storeDiffuse(pixel: vec2u, samples: u32, value: vec4f) -> void {
                var previous = vec4f(0.0);
                if (samples > 1u) { previous = textureLoad(${params.diffuseTarget}, pixel); }
                textureStore(${params.diffuseTarget}, pixel, mix(previous, value, 1.0 / f32(samples)));
            }
        ` : wgslTagFn`fn storeDiffuse(pixel: vec2u, samples: u32, value: vec4f) -> void {}`;

		const fn = (shaderList === undefined ? pipelineWgslTagFn(73)`

			fn compute(
				// settings
				misEnabled: u32,
				rayCount: u32,
				resetRayQueues: u32,
				maxBounces: u32,
				clampDirect: f32,
				clampIndirect: f32,
				transmissiveBackground: u32,

				globalId: vec3u
			) -> void {


				let rayIntersectionsStorage = &${ params.rayIntersectionsStorage };
				let shadowRayIntersectionsStorage = &${ params.shadowRayIntersectionsStorage };

				// bound by "rayCount" rather than the pool length. See MaterialKernel
				let index = globalId.x;
				// Logic reads intersections, never the trace queues. Reset both queue
				// headers here; the next dispatch is the storage visibility boundary.
				if ( index == 0u && resetRayQueues != 0u ) {

					let rayQueue = &${ params.rayQueue };
					let shadowRayQueue = &${ params.shadowRayQueue };
					atomicStore( &rayQueue.length, 0u );
					atomicStore( &shadowRayQueue.length, 0u );

				}
				if ( index >= rayCount ) {

					return;

				}

				// skip slots that have never spawned a ray
				let input = ${ path.load }(index);
				if ( input.rayIntersectionIndex < 0 ) {

					return;

				}

				let indexUV = vec2u( input.pixelIndex >> 16, input.pixelIndex & 0xFFFF );
				${ rngInit }( indexUV, input.seed, input.currentBounce + input.alphaDepth );

				// one-sample NEE chooses between the analytic lights and the environment, matched with
				// the megakernel
				let envActive = ${ envTotalSumNode } > 0.0 && ${ envIntensityNode } > 0.0;
				let lightsCount = ${ lightsCountNode };
				var envWeight = 0.0;
				if ( envActive ) {

					envWeight = ${ getEnvWeight }();

				}

				var resultColor = input.resultColor;
				var throughputColor = input.throughputColor;
                var diffuseThroughput = input.diffuseThroughput;
                var diffuseResult = input.diffuseResult;

				// added last, since a camera ray miss overwrites the pixel with the background
				var lightHits = vec3f( 0.0 );
				var cameraHitLight = false;

				// resolve the previous surface's NEE shadow ray (pre-scatter throughput). The index
				// is negative when no shadow ray was enqueued last frame
				if ( input.shadowRayIntersectionIndex >= 0 && input.lightPdf > 0.0 ) {

					let shadowHit = ( *shadowRayIntersectionsStorage )[ u32( input.shadowRayIntersectionIndex ) ];
					let occluded = shadowHit.objectIndex >= 0;
					if ( ! occluded ) {

						// env + area lights are also bsdf-sampled, so MIS-weight them; punctual take full weight,
						// or when a bsdf ray could never reach the light
						let isMISWeighted = ${ isMISWeightLightFn }( input.lightType ) && shadowHit.forwardBlocked == 0u;
						let misWeight = select( 1.0, ${ misHeuristicFn }( input.lightPdf, input.lightBsdfPdf ), isMISWeighted );
						let directLight = throughputColor * input.lightEmission * input.lightBsdf * misWeight / input.lightPdf;
                        if (input.currentBounce == 1u) {
                            diffuseResult += throughputColor * input.lightEmission * input.diffuseLight * misWeight / input.lightPdf;
                        }
						let contribution = ${ clampPathContributionFunc }( directLight, input.currentBounce, clampDirect, clampIndirect );
						resultColor += vec4f( contribution, 0.0 );
                        diffuseResult += diffuseThroughput * input.lightEmission * input.lightBsdf * misWeight / input.lightPdf;

					}

				}

				// emission gathered at the previous surface (pre-scatter throughput). Found light clamps
				// at the depth of the segment that found it, as in Cycles, so one less than the current
				let emission = ${ clampPathContributionFunc }( throughputColor * input.emission, max( input.currentBounce, 1u ) - 1u, clampDirect, clampIndirect );
				resultColor += vec4f( emission, 0.0 );
                diffuseResult += diffuseThroughput * input.emission;

				// reconstruct the scatter record staged by MaterialKernel
				var scatterRec: ${ scatterRecordStruct };
				scatterRec.color = input.scatterColor;
				scatterRec.pdf = input.scatterPdf;

				// the bounce limit, russian roulette, and terminating scatter checks all ran in
				// MaterialKernel, which stages a zeroed pdf and skips the bounce trace when they fire
				var isTerminated = all( throughputColor == vec3f( 0.0 ) ) || input.currentBounce >= maxBounces || ${ isTerminatingScatterFunc }( scatterRec );

				if ( ! isTerminated ) {

					// apply the scatter across the traced segment
					if (input.currentBounce == 1u) {
                        diffuseThroughput = throughputColor * input.diffuseScatter / scatterRec.pdf;
                    } else {
                        diffuseThroughput *= scatterRec.color / scatterRec.pdf;
                    }
                    throughputColor *= scatterRec.color / scatterRec.pdf;

					let hitResult = ( *rayIntersectionsStorage )[ u32( input.rayIntersectionIndex ) ];
					let didHit = hitResult.objectIndex >= 0;
					let surfaceDist = select( ${ LIGHT_FAR_DISTANCE }, hitResult.dist, didHit );

					// NEE's total weight at the vertex this segment left, for MIS. Camera segments have none
					let isMISWeighted = misEnabled != 0u && input.currentBounce > 0u;
					var lightTotalWeight = 0.0;
					if ( isMISWeighted ) {

						lightTotalWeight = ${ getLightsWeightFn }( input.origin ) + envWeight;

					}

					// forward hits on area lights. Camera rays only see visibleToCameraRays lights, at full weight
					for ( var li = 0u; li < lightsCount; li ++ ) {

						if ( input.currentBounce == 0u && ! ${ isLightVisibleToCameraFn }( li ) ) {

							continue;

						}

						var lightRec: ${ lightRecordStruct };
						if ( ${ intersectLightAtIndexFn }( input.origin, input.direction, li, &lightRec ) && lightRec.dist < surfaceDist ) {

							var misWeight = 1.0;
							if ( isMISWeighted ) {

								let selectionPdf = ${ getLightWeightFn }( li, input.origin ) / lightTotalWeight;
								misWeight = ${ misHeuristicFn }( input.scatterPdf, lightRec.pdf * selectionPdf );

							}

							lightHits += ${ clampPathContributionFunc }( lightRec.emission * throughputColor * misWeight, input.currentBounce, clampDirect, clampIndirect );
                            diffuseResult += lightRec.emission * diffuseThroughput * misWeight;
							cameraHitLight = cameraHitLight || input.currentBounce == 0u;

						}

					}

					if ( didHit ) {

						// stage the hit for MaterialKernel
						${ path.write.barycoord }(index, hitResult.barycoord);
						${ path.write.normal }(index, hitResult.normal);
						${ path.write.side }(index, hitResult.side);
						${ path.write.indices }(index, hitResult.indices);
						${ path.write.objectIndex }(index, hitResult.objectIndex);
						${ path.write.dist }(index, hitResult.dist);

						// next event estimation: choose one light or the environment by its estimated
						// contribution. MaterialKernel evaluates the bsdf and enqueues the shadow ray.
						var lightPdf = 0.0;
						if ( misEnabled != 0u ) {

							let ruv = ${ rand3 }( ${ RNG_INDEX_DIRECT_LIGHT_SAMPLE } );
							var selectionPdf = 0.0;
							let lightIndex = ${ selectLightFn }( hitResult.position, envWeight, ruv.x, &selectionPdf );
							if ( selectionPdf > 0.0 ) {

								var lightRec: ${ lightRecordStruct };
								if ( envActive && lightIndex == lightsCount ) {

									// the environment, sampled from its CDF, as a light of kind ENVIRONMENT
									let envSample = ${ sampleEnvDir }( ruv.yz );
									lightRec.direction = envSample.direction;
									lightRec.emission = envSample.color;
									lightRec.pdf = envSample.pdf;
									lightRec.dist = ${ LIGHT_FAR_DISTANCE };
									lightRec.lightType = ${ ENVIRONMENT_LIGHT_TYPE };

								} else {

									lightRec = ${ randomLightSampleFn }( lightIndex, hitResult.position, ruv.yz );

								}

								lightPdf = lightRec.pdf * selectionPdf;
								${ path.write.lightDirection }(index, lightRec.direction);
								${ path.write.lightEmission }(index, lightRec.emission);
								${ path.write.lightDist }(index, lightRec.dist);
								${ path.write.lightType }(index, lightRec.lightType);

							}

						}

						${ path.write.lightPdf }(index, lightPdf);

					} else {

						// the segment escaped the scene: gather the environment for opaque paths, or
						// the background for camera segments and fully transmissive paths
						if ( input.currentBounce > 0u && input.isFullyTransmissive == 0u ) {

							var misWeight = 1.0;
							if ( isMISWeighted && envActive ) {

								// match the env pdf scaling used by the NEE selection so the two estimators balance
								let envPdf = ${ getEnvDirPdf }( input.direction ) * envWeight / lightTotalWeight;
								misWeight = ${ misHeuristicFn }( input.scatterPdf, envPdf );

							}

							let environment = ${ sampleEnvColor }( input.direction ).rgb * throughputColor * misWeight;
							let contribution = ${ clampPathContributionFunc }( environment, input.currentBounce, clampDirect, clampIndirect );
							resultColor += vec4f( contribution, 0.0 );
                        diffuseResult += ${ sampleEnvColor }(input.direction).rgb * diffuseThroughput * misWeight;

						} else {

							// hit the background
							// support multiple transparent background blending techniques
							let rng = ${ rand2 }( ${ RNG_INDEX_BACKGROUND_SAMPLE } );
							let bg = ${ sampleBackground }( input.direction, rng );
							if ( input.currentBounce == 0u ) {

								// sample the background directly if this is the primary ray
								let background = ${ clampPathContributionFunc }( bg.a * bg.rgb, input.currentBounce, clampDirect, clampIndirect );
								resultColor = vec4f( background, bg.a );

							} else {

								// transmissive ray handling
								let env = ${ sampleEnvColor }( input.direction );
								let avg = saturate( dot( throughputColor, vec3f( 1.0 / 3.0 ) ) );
								let transparency = ( 1.0 - bg.a ) * avg;

								var misWeight = 1.0;
								if ( misEnabled != 0u && envActive ) {

									let envPdf = ${ getEnvDirPdf }( input.direction );
									misWeight = ${ misHeuristicFn }( input.scatterPdf, envPdf );

								}

								if ( transmissiveBackground == ${ TRANSMISSIVE_BACKGROUND_ENVIRONMENT }u ) {

									// display the env map through transmissive surfaces
									let background = ${ clampPathContributionFunc }( env.rgb * throughputColor * misWeight, input.currentBounce, clampDirect, clampIndirect );
									resultColor = vec4f(
										resultColor.rgb + background,
										1.0,
									);

								} else if ( transmissiveBackground == ${ TRANSMISSIVE_BACKGROUND_TRANSPARENT }u ) {

									// fade the background by the throughput color average
									let background = ${ clampPathContributionFunc }( bg.a * bg.rgb * throughputColor * misWeight, input.currentBounce, clampDirect, clampIndirect );
									resultColor = vec4f(
										resultColor.rgb + background,
										1.0 - transparency,
									);

								} else {

									// fade the background by the throughput color average, mixing in env lighting
									var light = mix( env.rgb, bg.rgb, bg.a ) * misWeight;
									let background = ${ clampPathContributionFunc }( light * throughputColor, input.currentBounce, clampDirect, clampIndirect );
									resultColor = vec4f(
										resultColor.rgb + background,
										1.0 - transparency,
									);

								}

							}

						}

						isTerminated = true;

					}

				}

				// visible lights are opaque against a transparent background
				resultColor += vec4f( lightHits, 0.0 );
				if ( cameraHitLight ) {

					resultColor.a = 1.0;

				}

				// the color rows are stored top down to match a rasterized render target
				let colorIndex = vec2u( indexUV.x, textureDimensions( ${ params.outputTarget } ).y - 1u - indexUV.y );
				let storedSamples = textureLoad( ${ params.sampleCountTarget }, indexUV ).r & ${ SAMPLE_COUNT_MASK }u;

				if ( isTerminated ) {

					// blend the finished sample into the output and free the slot for a new camera ray
					let sampleCount = storedSamples + 1;
					${ recordMoments }( indexUV, sampleCount, resultColor );
					let prevColor = ${ readPreviousColorFn }( indexUV, colorIndex, storedSamples );
					let blendedColor = ${ weightedAlphaBlendFn }( prevColor, resultColor, 1.0 / f32( sampleCount ) );
					textureStore( ${ params.sampleCountTarget }, indexUV, vec4( ${ SAMPLE_DISPATCHED_FLAG }u | sampleCount ) );
					${ storeColorFn }( indexUV, colorIndex, blendedColor );
                    ${ storeDiffuse }(colorIndex, sampleCount, vec4f(diffuseResult, resultColor.a));

					${ path.write.objectIndex }(index, - 1);

				} else {

					// show the path so far, which the first sample overwrites
					if ( storedSamples == 0u ) {

						textureStore( ${ params.outputTarget }, colorIndex, resultColor );

					}

					${ path.write.resultColor }(index, resultColor);
					${ path.write.throughputColor }(index, throughputColor);
                    ${ path.write.diffuseThroughput }(index, diffuseThroughput);
                    ${ path.write.diffuseResult }(index, diffuseResult);

				}

			}
` : bindPipelineShader(shaderList.pathLogic, { storeDiffuse, writeDiffuseThroughput: path.write.diffuseThroughput, writeDiffuseResult: path.write.diffuseResult, rayIntersectionsStorage: params.rayIntersectionsStorage, shadowRayIntersectionsStorage: params.shadowRayIntersectionsStorage, rayQueue: params.rayQueue, shadowRayQueue: params.shadowRayQueue, load: path.load, rngInit: rngInit, envTotalSumNode: envTotalSumNode, envIntensityNode: envIntensityNode, lightsCountNode: lightsCountNode, getEnvWeight: getEnvWeight, isMISWeightLightFn: isMISWeightLightFn, misHeuristicFn: misHeuristicFn, clampPathContributionFunc: clampPathContributionFunc, scatterRecordStruct: scatterRecordStruct, isTerminatingScatterFunc: isTerminatingScatterFunc, lightFarDistance: LIGHT_FAR_DISTANCE, getLightsWeightFn: getLightsWeightFn, isLightVisibleToCameraFn: isLightVisibleToCameraFn, lightRecordStruct: lightRecordStruct, intersectLightAtIndexFn: intersectLightAtIndexFn, getLightWeightFn: getLightWeightFn, writeBarycoord: path.write.barycoord, writeNormal: path.write.normal, writeSide: path.write.side, writeIndices: path.write.indices, writeObjectIndex: path.write.objectIndex, writeDist: path.write.dist, rand3: rand3, rngIndexDirectLightSample: RNG_INDEX_DIRECT_LIGHT_SAMPLE, selectLightFn: selectLightFn, sampleEnvDir: sampleEnvDir, environmentLightType: ENVIRONMENT_LIGHT_TYPE, randomLightSampleFn: randomLightSampleFn, writeLightDirection: path.write.lightDirection, writeLightEmission: path.write.lightEmission, writeLightDist: path.write.lightDist, writeLightType: path.write.lightType, writeLightPdf: path.write.lightPdf, getEnvDirPdf: getEnvDirPdf, sampleEnvColor: sampleEnvColor, rand2: rand2, rngIndexBackgroundSample: RNG_INDEX_BACKGROUND_SAMPLE, sampleBackground: sampleBackground, transmissiveBackgroundEnvironment: TRANSMISSIVE_BACKGROUND_ENVIRONMENT, transmissiveBackgroundTransparent: TRANSMISSIVE_BACKGROUND_TRANSPARENT, outputTarget: params.outputTarget, sampleCountTarget: params.sampleCountTarget, sampleCountMask: SAMPLE_COUNT_MASK, recordMoments: recordMoments, readPreviousColorFn: readPreviousColorFn, weightedAlphaBlendFn: weightedAlphaBlendFn, sampleDispatchedFlag: SAMPLE_DISPATCHED_FLAG, storeColorFn: storeColorFn, writeResultColor: path.write.resultColor, writeThroughputColor: path.write.throughputColor }));

		super( fn( params ) );

		this.defineUniformAccessors( params );

	}

}
