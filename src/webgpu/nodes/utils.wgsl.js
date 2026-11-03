import { pipelineWgslFn, pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { wgslFn } from 'three/tsl';
import { scatterRecordStruct } from './structs.wgsl.js';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';

export const inverseMat3x3Func = pipelineWgslFn(215, ["\n\n\tfn inverse(m: mat3x3f) -> mat3x3f {\n\t\tvar adj: mat3x3f;\n\t\tadj[0][0] =   (m[1][1] * m[2][2] - m[2][1] * m[1][2]);\n\t\tadj[1][0] = - (m[1][0] * m[2][2] - m[2][0] * m[1][2]);\n\t\tadj[2][0] =   (m[1][0] * m[2][1] - m[2][0] * m[1][1]);\n\t\tadj[0][1] = - (m[0][1] * m[2][2] - m[2][1] * m[0][2]);\n\t\tadj[1][1] =   (m[0][0] * m[2][2] - m[2][0] * m[0][2]);\n\t\tadj[2][1] = - (m[0][0] * m[2][1] - m[2][0] * m[0][1]);\n\t\tadj[0][2] =   (m[0][1] * m[1][2] - m[1][1] * m[0][2]);\n\t\tadj[1][2] = - (m[0][0] * m[1][2] - m[1][0] * m[0][2]);\n\t\tadj[2][2] =   (m[0][0] * m[1][1] - m[1][0] * m[0][1]);\n\n\t\tlet det = ( m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])\n\t\t\t- m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])\n\t\t\t+ m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]));\n\n\t\treturn adj * ( 1.0 / det );\n\t}\n\n"], [], []);

export const getBasisFromNormalFunc = pipelineWgslFn(216, ["\n\n\tfn getBasisFromNormal( normal: vec3f ) -> mat3x3f {\n\n\t\tvar other: vec3f;\n\t\tif ( abs( normal.x ) > 0.5 ) {\n\n\t\t\tother = vec3f( 0.0, 1.0, 0.0 );\n\n\t\t} else {\n\n\t\t\tother = vec3f( 1.0, 0.0, 0.0 );\n\n\t\t}\n\n\t\tlet ortho = normalize( cross( normal, other ) );\n\t\tlet ortho2 = normalize( cross( normal, ortho ) );\n\t\treturn mat3x3f( ortho2, ortho, normal );\n\n\t}\n\n"], [], []);

export const iorToF0Func = pipelineWgslFn(217, ["\n\n\tfn iorToF0( ior: f32 ) -> f32 {\n\t\treturn pow( ( 1 - ior ) / ( 1 + ior ), 2 );\n\t}\n\n"], [], []);

export const iorToF0GeneralFunc = pipelineWgslFn(218, ["\n\n\tfn iorToF0General( transmittedIor: f32, incidentIor: f32 ) -> f32 {\n\n\t\treturn pow( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ), 2 );\n\n\t}\n\n"], [], []);

export const iorToF0GeneralVecFunc = pipelineWgslFn(219, ["\n\n\tfn iorToF0GeneralVec( transmittedIor: vec3f, incidentIor: vec3f ) -> vec3f {\n\n\t\tlet v = ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor );\n\t\treturn v * v;\n\n\t}\n\n"], [], []);

export const fresnel0ToIorFunc = pipelineWgslFn(220, ["\n\n\tfn fresnel0ToIor( f0: vec3f ) -> vec3f {\n\n\t\tlet sqrtF0 = sqrt( f0 );\n\t\treturn ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );\n\n\t}\n\n"], [], []);

export const schlickFresnelFunc = pipelineWgslFn(221, ["\n\n\tfn schlickFresnel( cosine: f32, f0: f32 ) -> f32 {\n\n\t\treturn f0 + ( 1.0 - f0 ) * pow( 1.0 - cosine, 5.0 );\n\n\t}\n\n"], [], []);

export const schlickFresnelVecFunc = pipelineWgslFn(222, ["\n\n\tfn schlickFresnelVec( cosine: f32, f0: vec3f, f90: vec3f ) -> vec3f {\n\n\t\treturn f0 + ( f90 - f0 ) * pow( 1.0 - cosine, 5.0 );\n\n\t}\n\n"], [], []);

export const totalInternalReflectionFunc = pipelineWgslFn(223, ["\n\n\tfn totalInternalReflection( cosTheta: f32, eta: f32 ) -> bool {\n\n\t\tlet sinTheta = sqrt( 1.0 - cosTheta * cosTheta );\n\t\treturn eta * sinTheta > 1.0;\n\n\t}\n\n"], [], []);

export const totalInternalReflectionVecFunc = pipelineWgslFn(224, ["\n\n\tfn totalInternalReflectionVec( cosTheta: f32, eta: vec3f ) -> vec3<bool> {\n\n\t\tlet sinTheta = sqrt( 1.0 - cosTheta * cosTheta );\n\t\treturn eta * sinTheta > vec3f( 1.0 );\n\n\t}\n\n"], [], []);

export const evaluateFresnelFunc = pipelineWgslFn(225, ["\n\n\tfn evaluateFresnel( cosine: f32, eta: f32, f0: vec3f, f90: vec3f ) -> vec3f {\n\n\t\tif ( totalInternalReflection( cosine, eta ) ) {\n\n\t\t\treturn f90;\n\n\t\t}\n\n\t\treturn f0 + ( f90 - f0 ) * pow( 1.0 - cosine, 5.0 );\n\t}\n\n"], [], [ totalInternalReflectionFunc ]);

export const dielectricFresnelFunc = pipelineWgslFn(226, ["\n\n\tfn dielectricFresnel( cosThetaI: f32, eta: f32 ) -> f32 {\n\n\t\t// https://schuttejoe.github.io/post/disneybsdf/\n\t\tlet ni = eta;\n\t\tlet nt = 1.0;\n\n\t\t// Check for total internal reflection\n\t\tlet sinThetaISq = 1.0 - cosThetaI * cosThetaI;\n\t\tlet sinThetaTSq = eta * eta * sinThetaISq;\n\t\tif ( sinThetaTSq >= 1.0 ) {\n\n\t\t\treturn 1.0;\n\n\t\t}\n\n\t\tlet sinThetaT = sqrt( sinThetaTSq );\n\t\tlet cosThetaT = sqrt( max( 0.0, 1.0 - sinThetaT * sinThetaT ) );\n\t\tlet rParallel = ( ( nt * cosThetaI ) - ( ni * cosThetaT ) ) / ( ( nt * cosThetaI ) + ( ni * cosThetaT ) );\n\t\tlet rPerpendicular = ( ( ni * cosThetaI ) - ( nt * cosThetaT ) ) / ( ( ni * cosThetaI ) + ( nt * cosThetaT ) );\n\t\treturn ( rParallel * rParallel + rPerpendicular * rPerpendicular ) / 2.0;\n\n\t}\n\n"], [], []);

export const disneyFresnelFunc = pipelineWgslFn(227, ["\n\n\tfn disneyFresnel( wo: vec3f, wi: vec3f, wh: vec3f, f0: f32, eta: f32, metalness: f32 ) -> f32 {\n\n\t\tlet dotHV = dot( wo, wh );\n\t\tif ( totalInternalReflection( dotHV, eta ) ) {\n\n\t\t\treturn 1.0;\n\n\t\t}\n\n\t\tlet dotHL = dot( wi, wh );\n\t\tlet dielectricF = dielectricFresnel( abs( dotHV ), eta );\n\t\tlet metallicF = schlickFresnel( dotHL, f0 );\n\n\t\treturn mix( dielectricF, metallicF, metalness );\n\n\t}\n\n"], [], [ totalInternalReflectionFunc, dielectricFresnelFunc, schlickFresnelFunc ]);

export const isTerminatingScatterFunc = pipelineWgslFn(228, ["\n\n\tfn isTerminatingScatter( scatterRec: ScatterRecord ) -> bool {\n\n\t\treturn scatterRec.pdf <= 0;\n\n\t}\n\n"], [], [ scatterRecordStruct ]);

// Clamp individual light-path contributions using Cycles' RGB sum while preserving chromaticity.
// Matches film_clamp_light in Cycles' kernel/film/light_passes.h, including the x3 limit
// scaling applied to the user-facing clamp values in scene/integrator.cpp.
export const clampPathContributionFunc = pipelineWgslFn(229, ["\n\n\tfn clampPathContribution( contribution: vec3f, pathDepth: u32, clampDirect: f32, clampIndirect: f32 ) -> vec3f {\n\n\t\tlet limit = select( clampIndirect, clampDirect, pathDepth <= 1u ) * 3.0;\n\t\tif ( limit <= 0.0 ) {\n\n\t\t\treturn contribution;\n\n\t\t}\n\n\t\tlet strength = dot( abs( contribution ), vec3f( 1.0 ) );\n\t\tif ( strength > limit ) {\n\n\t\t\treturn contribution * ( limit / strength );\n\n\t\t}\n\n\t\treturn contribution;\n\n\t}\n\n"], [], []);

export const applyWrapFunc = pipelineWgslFn(230, ["\n\n\tfn applyWrap( v: f32, wrapMode: i32 ) -> f32 {\n\n\t\tvar res = v;\n\t\tif ( wrapMode == 1 ) {\n\n\t\t\t// ClampToEdge\n\t\t\tres = clamp( res, 0.0, 1.0 );\n\n\t\t} else if ( wrapMode == 2 ) {\n\n\t\t\t// MirroredRepeat\n\t\t\tres = 1.0 - abs( 2.0 * fract( res * 0.5 ) - 1.0 );\n\n\t\t} else {\n\n\t\t\t// Repeat\n\t\t\tres = fract( res );\n\n\t\t}\n\n\t\treturn res;\n\n\t}\n\n"], [], []);

// Bit-level ray origin offset based on Section 6.2.2 of "A Fast and Robust
// Method for Avoiding Self-Intersection" in Ray Tracing Gems:
// https://github.com/Apress/ray-tracing-gems/blob/master/Ch_06_A_Fast_and_Robust_Method_for_Avoiding_Self-Intersection/offset_ray.cu
// The original implementation expects a normal oriented for the outgoing ray;
// this version orients the geometric normal internally.
export const offsetRayOriginFunc = pipelineWgslFn(231, ["\n\n\tfn offsetRayOrigin( point: vec3f, direction: vec3f, geometricNormal: vec3f ) -> vec3f {\n\n\t\tlet normal = normalize( select( -geometricNormal, geometricNormal, dot( direction, geometricNormal ) >= 0.0 ) );\n\t\tlet intScale = 256.0;\n\t\tlet integerOffset = vec3i( intScale * normal );\n\t\tlet pointBits = bitcast<vec3i>( point );\n\t\tlet signedOffset = select( integerOffset, -integerOffset, point < vec3f( 0.0 ) );\n\t\tlet offsetPoint = bitcast<vec3f>( pointBits + signedOffset );\n\t\tlet origin = 1.0 / 32.0;\n\t\tlet floatScale = 1.0 / 65536.0;\n\t\treturn select( offsetPoint, point + floatScale * normal, abs( point ) < vec3f( origin ) );\n\n\t}\n\n"], [], []);

// Wraps a bilinear tap that is off the edge of a tile back to the texel the wrap mode calls for
export const wrapTexelIndexFunc = pipelineWgslFn(232, ["\n\n\tfn wrapTexelIndex( i: i32, size: i32, wrapMode: i32 ) -> i32 {\n\n\t\tif ( wrapMode == 0 ) {\n\n\t\t\t// Repeat wraps to the opposite edge\n\t\t\treturn ( ( i % size ) + size ) % size;\n\n\t\t}\n\n\t\t// ClampToEdge and MirroredRepeat both fold back onto the edge texel\n\t\treturn clamp( i, 0, size - 1 );\n\n\t}\n\n"], [], []);

// Factory: builds sampleTexel bound to the given per-instance textureInfo uniform
// array node ( must be named "textureInfo" ). Called once per scene so a single
// sampleTexel / textureInfo binding is shared by every caller in a pipeline. A distinct
// name must be given when a second atlas binding is used in the same pipeline.
export const sampleTexelFunc = ( textureInfoUniform, atlas, name = 'sampleTexel' ) => pipelineWgslTagFn(233)`

	fn ${ name }( uv: vec2f, packed: i32, lod: f32 ) -> vec4f {

		let texIndex = packed & 0x7FFFFF;
		let wrapS    = ( packed >> 26 ) & 0x3;
		let wrapT    = ( packed >> 28 ) & 0x3;
		let nearest  = ( packed >> 30 ) & 0x1;

		// look up the texture's rect and page within the atlas. three wraps a
		// uniformArray in a struct ( textureInfoStruct ) with a "value" array member.
		let info   = ${ textureInfoUniform }[ u32( texIndex ) ];
		let offset = vec2f( vec2u( info.x & 0xFFFFu, info.x >> 16u ) );
		let size   = vec2f( vec2u( info.y & 0xFFFFu, info.y >> 16u ) );
		let page   = i32( info.z & 0xFFFFu );

		// wrap is applied on the logical uv first, then remapped into the tile
		let wrappedUv = vec2f(
			${ applyWrapFunc }( uv.x, wrapS ),
			${ applyWrapFunc }( uv.y, wrapT ),
		);

		if ( nearest == 1 ) {

			let tileTexel = clamp( vec2i( wrappedUv * size ), vec2i( 0 ), vec2i( size ) - vec2i( 1 ) );
			return textureLoad( ${ atlas }, vec2i( offset ) + tileTexel, page, 0 );

		} else {

			// The tile's neighbors in the atlas are unrelated textures, so hardware
			// filtering cannot supply the taps that fall outside it. Filter here instead
			// and resolve each tap through the wrap mode.
			let texelPos = wrappedUv * size - vec2f( 0.5 );
			let basePos = floor( texelPos );
			let base = vec2i( basePos );
			let f = texelPos - basePos;

			let intSize = vec2i( size );
			let x0 = ${ wrapTexelIndexFunc }( base.x,     intSize.x, wrapS );
			let x1 = ${ wrapTexelIndexFunc }( base.x + 1, intSize.x, wrapS );
			let y0 = ${ wrapTexelIndexFunc }( base.y,     intSize.y, wrapT );
			let y1 = ${ wrapTexelIndexFunc }( base.y + 1, intSize.y, wrapT );

			let tile = vec2i( offset );
			let c00 = textureLoad( ${ atlas }, tile + vec2i( x0, y0 ), page, 0 );
			let c10 = textureLoad( ${ atlas }, tile + vec2i( x1, y0 ), page, 0 );
			let c01 = textureLoad( ${ atlas }, tile + vec2i( x0, y1 ), page, 0 );
			let c11 = textureLoad( ${ atlas }, tile + vec2i( x1, y1 ), page, 0 );

			return mix(
				mix( c00, c10, f.x ),
				mix( c01, c11, f.x ),
				f.y,
			);

		}

	}
`;

