import { MaterialQueueKernel, MATERIAL_QUEUE_COUNT } from './compute/wavefront/MaterialQueueKernel.js';
import { PATH_STATE_WORDS } from './compute/wavefront/path-state.js';
import { CopyPathStateKernel } from './compute/wavefront/CopyPathStateKernel.js';
import { AdaptiveSamplingKernel } from './compute/wavefront/AdaptiveSamplingKernel.js';
import { Matrix4, StorageBufferAttribute, StorageTexture, FloatType, Vector2 } from 'three/webgpu';
import { PopulatePixelIndicesKernel } from './compute/wavefront/PopulatePixelIndicesKernel.js';
import { LogicKernel } from './compute/wavefront/LogicKernel.js';
import { MaterialKernel } from './compute/wavefront/MaterialKernel.js';
import { TraceRayKernel } from './compute/wavefront/TraceRayKernel.js';
import { TraceShadowRayKernel } from './compute/wavefront/TraceShadowRayKernel.js';
import { QueueLengthToDispatchKernel } from './compute/wavefront/QueueLengthToDispatchKernel.js';
import { ResetSlotsKernel } from './compute/wavefront/ResetSlotsKernel.js';
import { ZeroOutBufferKernel } from './compute/ZeroOutBufferKernel.js';
import { CopyBufferKernel } from './compute/CopyBufferKernel.js';
import { EquirectHdrInfoNode } from './EquirectHdrInfoNode.js';
import { EquirectBackgroundInfo } from './EquirectBackgroundInfo.js';
import { LightsInfoNode } from './LightsInfoNode.js';
import { rayDataStruct, traceQueuedRayStruct, intersectionResultStruct, rayQueueStruct, pixelQueueStruct } from './compute/wavefront/structs.js';
import { PathTracerBackend } from './PathTracerBackend.js';
import { FILTER_GLOSSY_DISABLED } from './nodes/material.wgsl.js';
import {
	PrimeSampleCountersKernel,
	TallySampleCountsKernel,
	SAMPLE_COUNTER_LENGTH,
	SAMPLE_COUNTER_CONVERGED_MIN,
	SAMPLE_COUNTER_MAX,
	SAMPLE_COUNTER_MIN,
	SAMPLE_COUNTER_PIXEL_COUNT,
	SAMPLE_COUNTER_TOTAL_HI,
	SAMPLE_COUNTER_TOTAL_LO,
	U32_RANGE,
} from './compute/TallySampleCountsKernel.js';

// set the buffers to the max possible size supported by default (128MB)
// TODO: this can be increased based on platform.
const MAX_BUFFER_SIZE = 134217728;

// the persistent path-slot pool is sized off the ray data struct; every other per-slot buffer
// ( queues, intersections ) is sized to the same count so a full pool can never overflow them
const MAX_RAY_DATA_COUNT = Math.floor( MAX_BUFFER_SIZE / ( rayDataStruct.getLength() * 4 ) );

const LOW_RES_ITERATIONS = 5;
const RESERVED_PATH_CAPACITY = 65536;

export class WaveFrontPathTracer extends PathTracerBackend {

	constructor( renderer, { sparseAccumulation = false, adaptiveSampling = false, pathStateLayout = 'aos', sortMaterialPaths = false, pruneZeroContribution = false, captureDiffuse = false, pipelineShaders } = {} ) {

		super( renderer, { doubleBuffered: ! sparseAccumulation } );
		this.sparseAccumulation = sparseAccumulation;
        this.captureDiffuse = captureDiffuse;
        this.diffuseTarget = null;
        this.pathStateLayout = pathStateLayout;
        this.sortMaterialPaths = sortMaterialPaths;
        this.materialWorkQueue = null;
        const soa = pathStateLayout === 'soa';
        this.adaptiveSampling = adaptiveSampling;
        this._adaptiveGeneration = 0;
        this.adaptiveStage = 0;
        this.activePixelCount = Infinity;
        this.adaptiveMoments = null;
        this.adaptiveRisk = null;
        this.adaptiveTargets = null;
        this.adaptiveSamplingKernel = new AdaptiveSamplingKernel(pipelineShaders?.adaptiveSampling).setWorkgroupSize(8, 8, 1);
        this.clearAdaptiveQueueKernel = new ZeroOutBufferKernel().setWorkgroupSize(1, 1, 1);
		this.outputStorage = null;

		// options
		this.batchDispatches = false;
		this.seed = 0;
		this.envInfo = new EquirectHdrInfoNode();
		this.backgroundInfo = new EquirectBackgroundInfo();
		this.lightsInfo = new LightsInfoNode();

		// The path slot pool, sized to the frame budget and resized in the render loop as the budget
		// changes. Every per-slot buffer is allocated to the same count so a full pool can never
		// overflow them.
		this.rayCount = 0;
		this.rayCapacity = 0;
		this._compilePromiseList = null;

		// persistent per-path state, one slot per in-flight path
		this.rayDataStorage = null;

		// append-only trace queues, prefixed by an atomic length header
		this.rayQueue = null;
		this.shadowRayQueue = null;

		// per-queue-slot trace results, indexed by the ray's position in its queue
		this.rayIntersectionsStorage = null;
		this.shadowRayIntersectionsStorage = null;

		// pixel indices waiting for a free path slot, sized to the resolution
		this.pixelQueue = null;

		// reduction target for the per pixel sample counts, read back asynchronously
		this.sampleCountersStorage = new StorageBufferAttribute( new Uint32Array( SAMPLE_COUNTER_LENGTH ), SAMPLE_COUNTER_LENGTH );
		this.sampleCountersStorage.name = 'Sample Counters';
		this._samplesPromise = null;

		// kernels
        this.materialQueueKernel = new MaterialQueueKernel(soa, pipelineShaders?.materialQueue).setWorkgroupSize(64, 1, 1);
        this.clearMaterialQueueKernel = new ZeroOutBufferKernel().setWorkgroupSize(1, 1, 1);
		this.populatePixelIndicesKernel = new PopulatePixelIndicesKernel(adaptiveSampling, soa, pipelineShaders?.wavefront).setWorkgroupSize( 8, 8, 1 );
		this.logicKernel = new LogicKernel( sparseAccumulation, adaptiveSampling, soa, pipelineShaders?.wavefront, captureDiffuse ).setWorkgroupSize( 64, 1, 1 );
		this.materialKernel = new MaterialKernel(adaptiveSampling, soa, sortMaterialPaths, pruneZeroContribution, pipelineShaders?.wavefront, captureDiffuse).setWorkgroupSize( 64, 1, 1 );
		this.traceRayKernel = new TraceRayKernel(pipelineShaders?.wavefront).setWorkgroupSize( 64, 1, 1 );
		this.traceShadowRayKernel = new TraceShadowRayKernel(pipelineShaders?.wavefront).setWorkgroupSize( 64, 1, 1 );
		this.rayDispatchConverter = new QueueLengthToDispatchKernel(undefined, undefined, pipelineShaders?.wavefront).setWorkgroupSize( 1, 1, 1 );
		this.shadowDispatchConverter = new QueueLengthToDispatchKernel(undefined, undefined, pipelineShaders?.wavefront).setWorkgroupSize( 1, 1, 1 );
		this.primeSampleCountersKernel = new PrimeSampleCountersKernel().setWorkgroupSize( 1, 1, 1 );
		this.tallySampleCountsKernel = new TallySampleCountsKernel(adaptiveSampling).setWorkgroupSize( 8, 8, 1 );

		// pool resizing: carry live slots into a reallocated pool and idle the rest
		this.resetSlotsKernel = new ResetSlotsKernel(soa, pipelineShaders?.wavefront).setWorkgroupSize( 64, 1, 1 );
		this.copyRayDataKernel = (soa ? new CopyPathStateKernel(pipelineShaders?.copyPathState) : new CopyBufferKernel( rayDataStruct )).setWorkgroupSize( 64, 1, 1 );

		// bind the shared env / lights providers so the kernels' proxies resolve even before they're set
		this.logicKernel.envInfo = this.envInfo;
		this.logicKernel.backgroundInfo = this.backgroundInfo;
		this.logicKernel.lightsInfo = this.lightsInfo;

		// clear kernels
		this.zeroDispatchKernel = new ZeroOutBufferKernel().setWorkgroupSize( 1, 1, 1 );
		this.zeroShadowDispatchKernel = new ZeroOutBufferKernel().setWorkgroupSize( 1, 1, 1 );

		// later
		this.volumeKernel = null;

	}

	_compute( node, dispatchSize ) {

		if ( this._compilePromiseList ) {

			this._compilePromiseList.push( this.renderer.compileComputeAsync( node ) );

		} else {

			this.renderer.compute( node, dispatchSize );

		}

	}

	async prepareAsync() {

		const seed = this.seed;
		const promises = [];
		this._compilePromiseList = promises;
		let preparationError;
		try {

			// Configure the real graph without executing path integration. Compile
			// every bound pipeline asynchronously before any expensive submission.
			this.createRenderTask().next();
            this.primeSampleCountersKernel.counters = this.sampleCountersStorage;
            this._compute(this.primeSampleCountersKernel.kernel, [1]);
            this.tallySampleCountsKernel.counters = this.sampleCountersStorage;
            this.tallySampleCountsKernel.sampleCountTarget = this.sampleCountTarget;
            this._compute(this.tallySampleCountsKernel.kernel, [1, 1]);
            if (this.adaptiveSampling) {
                const kernel = this.adaptiveSamplingKernel;
                kernel.moments = this.adaptiveMoments; kernel.risk = this.adaptiveRisk;
                kernel.targets = this.adaptiveTargets; kernel.pixelQueue = this.pixelQueue;
                kernel.sampleCountTarget = this.sampleCountTarget;
                this._compute(kernel.kernel, [1, 1]);
                this.clearAdaptiveQueueKernel.target = this.pixelQueue;
                this._compute(this.clearAdaptiveQueueKernel.kernel, [2]);
            }

		} catch ( error ) {

			preparationError = error;

		} finally {

			this._compilePromiseList = null;

		}
		const results = await Promise.allSettled( promises );
		this.seed = seed;
		this.reset();
		if ( preparationError ) throw preparationError;
		const failed = results.find( result => result.status === 'rejected' );
		if ( failed ) throw failed.reason;

	}

	resetSeed() {

		this.seed = 0;

	}

	reset() {

		super.reset();
        this._adaptiveGeneration++;
		this._samplesPromise = null;

	}

	setBVHData( bvhData ) {

		this.materialKernel.bvhData = bvhData;
        this.materialQueueKernel.bvhData = bvhData;
        this.materialQueueKernel.needsUpdate = true;
		this.materialKernel.needsUpdate = true;

		this.traceRayKernel.bvhData = bvhData;
		this.traceRayKernel.needsUpdate = true;

		this.traceShadowRayKernel.bvhData = bvhData;
		this.traceShadowRayKernel.needsUpdate = true;

		this.reset();

	}

	rebuild() {

		super.rebuild();

		// MaterialKernel bakes in the camera ray function, so a camera swap needs a recompile
		this.materialKernel.needsUpdate = true;
		this.reset();

	}

	setRandom( random ) {

		this.logicKernel.context.random = random;
		this.logicKernel.needsUpdate = true;

		this.materialKernel.context.random = random;
		this.materialKernel.needsUpdate = true;

		this.traceRayKernel.context.random = random;
		this.traceRayKernel.needsUpdate = true;

		this.traceShadowRayKernel.context.random = random;
		this.traceShadowRayKernel.needsUpdate = true;

		this.reset();

	}

	setMaterial( material ) {

		this.materialKernel.material = material.getData();
		this.materialKernel.needsUpdate = true;
		this.reset();

	}

	setFilterGlossy( value ) {

		// the kernel takes the inverted threshold, mirroring Cycles
		this.materialKernel.filterGlossy = value === 0 ? FILTER_GLOSSY_DISABLED : 1 / value;
		this.reset();

	}

	setClamping( direct, indirect ) {

		this.logicKernel.clampDirect = direct;
		this.logicKernel.clampIndirect = indirect;
		this.reset();

	}

	setTransmissiveBackground( value ) {

		this.logicKernel.transmissiveBackground = value;
		this.reset();

	}

	setEnvironment( envMap ) {

		this.envInfo.updateFrom( envMap );

	}

	setLights( lights ) {

		this.lightsInfo.updateFrom( this.renderer, lights );
		this.reset();

	}

	setEnvironmentParams( envMapIntensity, envMapRotation ) {

		const { envInfo } = this;
		const rotationMatrix = new Matrix4().makeRotationFromEuler( envMapRotation ).invert();
		envInfo.rotationNode.value.setFromMatrix4( rotationMatrix );
		envInfo.intensityNode.value = envMapIntensity;

	}

	setMultipleImportanceSampling( enabled ) {

		this.logicKernel.misEnabled = enabled ? 1 : 0;
		this.reset();

	}

	setBackground( background ) {

		const { backgroundInfo } = this;
		backgroundInfo.dispose();
		backgroundInfo.map = background;

	}

	setBackgroundParams(
		backgroundIntensity,
		backgroundRotation,
		backgroundBlurriness,
	) {

		const { backgroundInfo } = this;
		const rotationMatrix = new Matrix4().makeRotationFromEuler( backgroundRotation ).invert();
		backgroundInfo.rotationNode.value.copy( rotationMatrix );
		backgroundInfo.intensity = backgroundIntensity;
		backgroundInfo.blur = backgroundBlurriness;

	}

	dispose() {
        this.diffuseTarget?.dispose();

		super.dispose();

		this.envInfo.dispose();
		this.lightsInfo.dispose();

		this.rayDataStorage?.dispose();
		this.rayQueue?.dispose();
		this.shadowRayQueue?.dispose();
		this.rayIntersectionsStorage?.dispose();
		this.shadowRayIntersectionsStorage?.dispose();
		this.sampleCountersStorage.dispose();
		this.pixelQueue?.dispose();
		this.outputStorage?.dispose();
        this.adaptiveMoments?.dispose(); this.adaptiveRisk?.dispose(); this.adaptiveTargets?.dispose();
        this.adaptiveSamplingKernel.dispose(); this.clearAdaptiveQueueKernel.dispose();

		// TODO: the dispatch converters own their indirect dispatch buffers, they should dispose of them themselves
		this.rayDispatchConverter.outputDispatch.dispose();
		this.shadowDispatchConverter.outputDispatch.dispose();

		this.populatePixelIndicesKernel.dispose();
		this.logicKernel.dispose();
		this.materialKernel.dispose();
        this.materialQueueKernel.dispose();
        this.clearMaterialQueueKernel.dispose();
        this.materialWorkQueue?.dispose();
		this.traceRayKernel.dispose();
		this.traceShadowRayKernel.dispose();
		this.rayDispatchConverter.dispose();
		this.shadowDispatchConverter.dispose();
		this.primeSampleCountersKernel.dispose();
		this.tallySampleCountsKernel.dispose();
		this.resetSlotsKernel.dispose();
		this.copyRayDataKernel.dispose();
		this.zeroDispatchKernel.dispose();
		this.zeroShadowDispatchKernel.dispose();

	}

	// The queue holds every pixel not owned by a slot. It is sized to the full pixel count so slots
	// retired by a shrink can always return theirs.
	_updatePixelQueue( width, height ) {

		const size = pixelQueueStruct.getLength() + Math.max( width * height, 1 );
		if ( ! this.pixelQueue || this.pixelQueue.array.length < size ) {

			this.pixelQueue?.dispose();
			this.pixelQueue = new StorageBufferAttribute( new Float32Array( size ), size );
			this.pixelQueue.name = 'Pixel Queue';

			// the queue buffer object changed, so kernels bound to it must rebuild
			this.materialKernel.pixelQueue = this.pixelQueue;
			this.materialKernel.needsUpdate = true;
			this.populatePixelIndicesKernel.pixelQueue = this.pixelQueue;
			this.populatePixelIndicesKernel.needsUpdate = true;
			this.resetSlotsKernel.pixelQueue = this.pixelQueue;
			this.resetSlotsKernel.needsUpdate = true;

		}

	}

	// Reallocates the per slot buffers to "count" slots. Slots that survive carry their ray data
	// over, dropped slots return their pixel to the queue, and added slots take one from it. The
	// trace queues and results only live within a frame, so they start empty.
	_resizePool( count ) {

		const { renderer, resetSlotsKernel, copyRayDataKernel } = this;
		const previousCount = this.rayCount;

		resetSlotsKernel.pixelQueue = this.pixelQueue;
		if ( count < previousCount ) {

			resetSlotsKernel.rayDataStorage = this.rayDataStorage;
			resetSlotsKernel.start = count;
			resetSlotsKernel.end = previousCount;
			resetSlotsKernel.addingSlots = false;
			this._compute( resetSlotsKernel.kernel, resetSlotsKernel.getDispatchSize( previousCount - count, 1, 1 ) );

		}

		// Retain buffer identities while the active pool changes. This avoids
		// copying live paths and recompiling size-dependent WGSL on every adjustment.
		if ( this.rayDataStorage && count <= this.rayCapacity ) {

			this.rayCount = count;
			if ( count > previousCount ) {

				resetSlotsKernel.rayDataStorage = this.rayDataStorage;
				resetSlotsKernel.start = previousCount;
				resetSlotsKernel.end = count;
				resetSlotsKernel.addingSlots = true;
				this._compute( resetSlotsKernel.kernel, resetSlotsKernel.getDispatchSize( count - previousCount, 1, 1 ) );

			}
			return;

		}
		const pixelCount = this.outputTarget.width * this.outputTarget.height;
		const capacity = Math.min( MAX_RAY_DATA_COUNT, Math.max( count, Math.min( RESERVED_PATH_CAPACITY, pixelCount ) ) );

		const rayDataStorage = this.pathStateLayout === 'soa'
            ? new StorageBufferAttribute(new Uint32Array(capacity * PATH_STATE_WORDS), 1)
            : new StorageBufferAttribute( capacity, rayDataStruct.getLength() );
		rayDataStorage.name = 'Ray Data';

		const queueSize = rayQueueStruct.getLength() + capacity * traceQueuedRayStruct.getLength();
		const rayQueue = new StorageBufferAttribute( new Float32Array( queueSize ), queueSize );
		rayQueue.name = 'Ray Queue';

		const shadowRayQueue = new StorageBufferAttribute( new Float32Array( queueSize ), queueSize );
		shadowRayQueue.name = 'Shadow Ray Queue';

		const rayIntersectionsStorage = new StorageBufferAttribute( capacity, intersectionResultStruct.getLength() );
		rayIntersectionsStorage.name = 'Ray Intersections';

		const shadowRayIntersectionsStorage = new StorageBufferAttribute( capacity, intersectionResultStruct.getLength() );
		shadowRayIntersectionsStorage.name = 'Shadow Ray Intersections';

		const copyCount = Math.min( count, previousCount );
		if ( copyCount > 0 ) {

			copyRayDataKernel.source = this.rayDataStorage;
			copyRayDataKernel.target = rayDataStorage;
			copyRayDataKernel.count = copyCount;
			this._compute( copyRayDataKernel.kernel, copyRayDataKernel.getDispatchSize( copyCount, 1, 1 ) );

		}

		this.rayDataStorage?.dispose();
		this.rayQueue?.dispose();
		this.shadowRayQueue?.dispose();
		this.rayIntersectionsStorage?.dispose();
		this.shadowRayIntersectionsStorage?.dispose();

		this.rayDataStorage = rayDataStorage;
		this.rayQueue = rayQueue;
		this.shadowRayQueue = shadowRayQueue;
		this.rayIntersectionsStorage = rayIntersectionsStorage;
		this.shadowRayIntersectionsStorage = shadowRayIntersectionsStorage;
		this.rayCount = count;
		this.rayCapacity = capacity;
        if (this.sortMaterialPaths) {
            this.materialWorkQueue?.dispose();
            this.materialWorkQueue = new StorageBufferAttribute(new Uint32Array(MATERIAL_QUEUE_COUNT * (capacity + 1)), 1);
            this.materialQueueKernel.needsUpdate = true;
        }

		// the buffer objects changed, so kernels bound to them must rebuild
		this.logicKernel.needsUpdate = true;
		this.materialKernel.needsUpdate = true;
		this.traceRayKernel.needsUpdate = true;
		this.traceShadowRayKernel.needsUpdate = true;
		this.populatePixelIndicesKernel.needsUpdate = true;
		this.resetSlotsKernel.needsUpdate = true;

		if ( count > previousCount ) {

			resetSlotsKernel.rayDataStorage = rayDataStorage;
			resetSlotsKernel.start = previousCount;
			resetSlotsKernel.end = count;
			resetSlotsKernel.addingSlots = true;
			this._compute( resetSlotsKernel.kernel, resetSlotsKernel.getDispatchSize( count - previousCount, 1, 1 ) );

		}

	}

	// The slot count the current budget asks for, capped by the pool limit and the pixel count
	_getRequestedSlotCount( pixelCount ) {

		return Math.min( MAX_RAY_DATA_COUNT, this.activePixelCount, pixelCount, Math.max( 1, Math.floor( this.frameBudget ) ) );

	}

	// Resizes the pool when the budget changes. Paths in flight past a smaller budget are dropped.
	_applyBudget( pixelCount ) {

		const requested = this._getRequestedSlotCount( pixelCount );
		if ( requested !== this.rayCount ) {

			this._resizePool( requested );

		}

	}

	*createRenderTask() {

		const {
			renderer,
			maxTransparentBounces,

			logicKernel,
			materialKernel,
			traceRayKernel,
			traceShadowRayKernel,
			rayDispatchConverter,
			shadowDispatchConverter,
			zeroDispatchKernel,
			zeroShadowDispatchKernel,
			populatePixelIndicesKernel,
		} = this;

		// a resize resets, recreating this task, so the dimensions and pixel queue hold for its
		// lifetime. The slot pool starts tight, at the budget, and is resized in the loop below.
		const targetDimensions = new Vector2();
		this.getSize( targetDimensions );

		// referenced via "this" since the buffers may be replaced here and in the loop
		this._updatePixelQueue( targetDimensions.x, targetDimensions.y );

		// the populate pass below overwrites whatever the resize carried over
		const pixelCount = targetDimensions.x * targetDimensions.y;
        if (this.captureDiffuse) {
            if (!this.diffuseTarget || this.diffuseTarget.width !== targetDimensions.x || this.diffuseTarget.height !== targetDimensions.y) {
                this.diffuseTarget?.dispose();
                this.diffuseTarget = new StorageTexture(targetDimensions.x, targetDimensions.y);
                this.diffuseTarget.type = FloatType;
                logicKernel.needsUpdate = true;
            }
            logicKernel.diffuseTarget = this.diffuseTarget;
        }
        this.activePixelCount = pixelCount;
        this.adaptiveStage = this.adaptiveSampling ? Math.min(16, this.maxSamples) : this.maxSamples;
        if (this.adaptiveSampling) {
            if (!this.adaptiveMoments || this.adaptiveMoments.width !== targetDimensions.x * 2 || this.adaptiveMoments.height !== targetDimensions.y) {
                if (pixelCount * 32 > MAX_BUFFER_SIZE) throw new Error('Adaptive moments exceed the storage buffer limit.');
                this.adaptiveMoments?.dispose(); this.adaptiveRisk?.dispose(); this.adaptiveTargets?.dispose();
                this.adaptiveMoments = new StorageTexture(targetDimensions.x * 2, targetDimensions.y);
                this.adaptiveMoments.type = FloatType;
                this.adaptiveRisk = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
                this.adaptiveTargets = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
                for (const kernel of [logicKernel, materialKernel, populatePixelIndicesKernel, this.tallySampleCountsKernel, this.adaptiveSamplingKernel]) kernel.needsUpdate = true;
            }
            logicKernel.moments = this.adaptiveMoments;
            materialKernel.risk = this.adaptiveRisk;
            populatePixelIndicesKernel.risk = this.adaptiveRisk;
            populatePixelIndicesKernel.targets = this.adaptiveTargets;
            populatePixelIndicesKernel.finalSamples = this.maxSamples;
            this.tallySampleCountsKernel.targets = this.adaptiveTargets;
        }
		this._applyBudget( pixelCount );

		// reset the trace queues — only the length header needs zeroing
		zeroDispatchKernel.target = this.rayQueue;
		this._compute( zeroDispatchKernel.kernel, [ 1 ] );

		zeroDispatchKernel.target = this.shadowRayQueue;
		this._compute( zeroDispatchKernel.kernel, [ 1 ] );

		// assign every path slot a pixel and park the overflow pixels in the pixel queue
		populatePixelIndicesKernel.rayDataStorage = this.rayDataStorage;
		populatePixelIndicesKernel.pixelQueue = this.pixelQueue;
		populatePixelIndicesKernel.frameBudget = this.rayCount;
		populatePixelIndicesKernel.targetDimensions.copy( targetDimensions );
		this._compute( populatePixelIndicesKernel.kernel, populatePixelIndicesKernel.getDispatchSize( targetDimensions.x, targetDimensions.y, 1 ) );

		// advance the sequence once per task so each render pass starts from fresh
		// noise unless "stableNoise" has reset it
		this.seed ++;

		while ( true ) {
            if (this.activePixelCount === 0) { yield; continue; }

			// TODO: this only makes sense if we can cap the number rays per iteration to the full
			// frame
			const iter = this.lowResMode ? Math.min( this.maxBounces, LOW_RES_ITERATIONS ) : 1;

			for ( let i = 0; i < iter; i ++ ) {

				if ( ! this.sparseAccumulation ) {

					renderer.copyTextureToTexture( this.outputTarget, this.prevOutputTarget );
					[ this.outputTarget, this.prevOutputTarget ] = [ this.prevOutputTarget, this.outputTarget ];
					logicKernel.prevOutputTarget = this.prevOutputTarget;

				}
				logicKernel.outputTarget = this.outputTarget;
				logicKernel.sampleCountTarget = this.sampleCountTarget;

				// Step 1: resolve last frame's trace results — accumulate NEE / emission / env, terminate
				// finished paths into the output, and pick the next NEE light for each live path
				logicKernel.rayDataStorage = this.rayDataStorage;
				logicKernel.rayIntersectionsStorage = this.rayIntersectionsStorage;
				logicKernel.shadowRayIntersectionsStorage = this.shadowRayIntersectionsStorage;
				logicKernel.maxBounces = this.maxBounces;
				logicKernel.rayCount = this.rayCount;
				logicKernel.rayQueue = this.rayQueue;
				logicKernel.shadowRayQueue = this.shadowRayQueue;
				logicKernel.resetRayQueues = this.batchDispatches ? 1 : 0;
				// A pool resize must consume the old intersections before replacing buffers.
				const batch = this.batchDispatches && this._getRequestedSlotCount( pixelCount ) === this.rayCount;
				logicKernel.kernel.dispatchSize = logicKernel.getDispatchSize( this.rayCount, 1, 1 );
				if ( ! batch ) {

					this._compute( logicKernel.kernel );
					this._applyBudget( pixelCount );

				}

				const {
					rayDataStorage,
					rayQueue,
					shadowRayQueue,
					rayIntersectionsStorage,
					shadowRayIntersectionsStorage,
					rayCount,
				} = this;

				// Step 2: reset the trace queues for this frame's population
				zeroDispatchKernel.target = rayQueue;
				zeroShadowDispatchKernel.target = shadowRayQueue;
				zeroDispatchKernel.kernel.dispatchSize = [ 1 ];
				zeroShadowDispatchKernel.kernel.dispatchSize = [ 1 ];

				// Step 3: evaluate materials — spawn camera rays for freed slots, sample the bsdf, and
				// enqueue this frame's bounce + shadow rays
				materialKernel.rayDataStorage = rayDataStorage;
				materialKernel.rayQueue = rayQueue;
				materialKernel.shadowRayQueue = shadowRayQueue;
				materialKernel.pixelQueue = this.pixelQueue;
				materialKernel.sampleCountTarget = this.sampleCountTarget;
				materialKernel.seed = this.seed;
				materialKernel.maxSamples = this.adaptiveSampling ? this.adaptiveStage : this.maxSamples;
				materialKernel.rayCount = rayCount;
				materialKernel.maxTransparentBounces = maxTransparentBounces;
				materialKernel.maxBounces = this.maxBounces;
				materialKernel.targetDimensions.copy( targetDimensions );
				materialKernel.kernel.dispatchSize = materialKernel.getDispatchSize( rayCount, 1, 1 );

				const materialQueueDispatchList = [];
                if (this.sortMaterialPaths) {
                    const classify = this.materialQueueKernel;
                    classify.rayDataStorage = rayDataStorage;
                    classify.workQueue = this.materialWorkQueue;
                    classify.rayCount = rayCount;
                    classify.kernel.dispatchSize = classify.getDispatchSize(rayCount, 1, 1);
                    this.clearMaterialQueueKernel.target = this.materialWorkQueue;
                    this.clearMaterialQueueKernel.kernel.dispatchSize = [MATERIAL_QUEUE_COUNT];
                    materialKernel.workQueue = this.materialWorkQueue;
                    materialQueueDispatchList.push(this.clearMaterialQueueKernel.kernel, classify.kernel);
                }
                // Step 4: convert the queue lengths into indirect dispatch sizes and trace
				rayDispatchConverter.queue = rayQueue;
				rayDispatchConverter.kernel.dispatchSize = [ 1, 1, 1 ];

				traceRayKernel.rayQueue = rayQueue;
				traceRayKernel.rayIntersectionsStorage = rayIntersectionsStorage;
				traceRayKernel.kernel.dispatchSize = rayDispatchConverter.outputDispatch;

				shadowDispatchConverter.queue = shadowRayQueue;
				shadowDispatchConverter.kernel.dispatchSize = [ 1, 1, 1 ];

				traceShadowRayKernel.shadowRayQueue = shadowRayQueue;
				traceShadowRayKernel.shadowRayIntersectionsStorage = shadowRayIntersectionsStorage;
				traceShadowRayKernel.kernel.dispatchSize = shadowDispatchConverter.outputDispatch;

				// Distinct clear nodes retain distinct bindings. Dispatch boundaries preserve
				// storage dependencies, including indirect argument production/consumption.
				const computeList = [
					zeroDispatchKernel.kernel, zeroShadowDispatchKernel.kernel, ...materialQueueDispatchList, materialKernel.kernel,
					rayDispatchConverter.kernel, traceRayKernel.kernel,
					shadowDispatchConverter.kernel, traceShadowRayKernel.kernel,
				];
				if ( batch ) {

					this._compute( [ logicKernel.kernel, ...computeList.slice( 2 ) ] );

				} else {

					for ( const node of computeList ) this._compute( node );

				}

			}

			yield;

		}

	}

	// Reduces the per pixel sample counts and reads them back. Runs a full resolution pass, so it
	// only happens when asked for.
	getSampleCountsAsync() {

		// share the in flight measurement rather than dispatching another
		if ( this._samplesPromise === null ) {

			const promise = this._measureSampleCounts().finally( () => {

				// a reset may have started a newer measurement in the meantime
				if ( this._samplesPromise === promise ) {

					this._samplesPromise = null;

				}

			} );

			this._samplesPromise = promise;

		}

		return this._samplesPromise;

	}

	async _measureSampleCounts() {
        const generation = this._adaptiveGeneration;

		const {
			renderer,
			sampleCountTarget,
			sampleCountersStorage,
			primeSampleCountersKernel,
			tallySampleCountsKernel,
		} = this;

		if ( ! renderer.initialized || this.lowResMode ) {

			return { min: 0, max: 0, avg: 0 };

		}

		primeSampleCountersKernel.counters = sampleCountersStorage;
		this._compute( primeSampleCountersKernel.kernel, [ 1, 1, 1 ] );

		tallySampleCountsKernel.counters = sampleCountersStorage;
		tallySampleCountsKernel.sampleCountTarget = sampleCountTarget;
        tallySampleCountsKernel.stageSamples = this.adaptiveStage;
		this._compute(
			tallySampleCountsKernel.kernel,
			tallySampleCountsKernel.getDispatchSize( sampleCountTarget.width, sampleCountTarget.height ),
		);

		const buffer = await renderer.getArrayBufferAsync( sampleCountersStorage );
		const counters = new Uint32Array( buffer );
		const pixelCount = counters[ SAMPLE_COUNTER_PIXEL_COUNT ];

		// no camera ray has been dispatched yet, so there is nothing to average over
		if ( pixelCount === 0 ) {

			return { min: 0, max: 0, avg: 0 };

		}

		// the total is accumulated as a split 64 bit value to survive high sample counts
		const total = counters[ SAMPLE_COUNTER_TOTAL_HI ] * U32_RANGE + counters[ SAMPLE_COUNTER_TOTAL_LO ];

        let convergedMin = counters[SAMPLE_COUNTER_CONVERGED_MIN];
        if (generation === this._adaptiveGeneration && this.adaptiveSampling && convergedMin >= this.maxSamples) this.activePixelCount = 0;
        if (generation === this._adaptiveGeneration && this.adaptiveSampling && convergedMin >= this.adaptiveStage && this.adaptiveStage < this.maxSamples) {
            await this._advanceAdaptiveStage(generation);
            if (this.activePixelCount === 0) convergedMin = this.maxSamples;
        }
		return {
            ...(this.adaptiveSampling ? {convergedMin, activePixels: this.activePixelCount, adaptiveStage: this.adaptiveStage} : {}),
			min: counters[ SAMPLE_COUNTER_MIN ],
			max: counters[ SAMPLE_COUNTER_MAX ],
			avg: total / pixelCount,
		};

	}

    async _advanceAdaptiveStage(generation) {
        const kernel = this.adaptiveSamplingKernel;
        kernel.moments = this.adaptiveMoments; kernel.risk = this.adaptiveRisk;
        kernel.targets = this.adaptiveTargets; kernel.pixelQueue = this.pixelQueue;
        kernel.sampleCountTarget = this.sampleCountTarget; kernel.finalSamples = this.maxSamples;
        // Zero BOTH header words; no pixel is in flight at this stage boundary.
        this.clearAdaptiveQueueKernel.target = this.pixelQueue;
        this._compute(this.clearAdaptiveQueueKernel.kernel, [2]);
        this._compute(kernel.kernel, kernel.getDispatchSize(this.outputTarget.width, this.outputTarget.height));
        // Read only the two header words, never download the full image or queue.
        const buffer = await this.renderer.getArrayBufferAsync(this.pixelQueue, null, 0, 8);
        if (generation !== this._adaptiveGeneration) return;
        this.activePixelCount = new Uint32Array(buffer)[1];
        this.adaptiveStage = Math.min(this.maxSamples, this.adaptiveStage * 2);
        this.rayCount = 0;
        if (this.activePixelCount > 0) this._applyBudget(this.outputTarget.width * this.outputTarget.height);
        await this.renderer.backend.device.queue.onSubmittedWorkDone();
    }

}
