import { pipelineWgslFn, pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { wgslFn, mat3 } from 'three/tsl';
import {
	inverseMat3x3Func,
	getBasisFromNormalFunc,
	iorToF0Func,
	schlickFresnelFunc,
	schlickFresnelVecFunc,
	iorToF0GeneralFunc,
	fresnel0ToIorFunc,
	iorToF0GeneralVecFunc,
	dielectricFresnelFunc,
	totalInternalReflectionFunc,
	totalInternalReflectionVecFunc,
} from './utils.wgsl.js';
import {
	ggxSmithVisibilityFunc,
	ggxDistributionFunc,
	ggxDirectionFunc,
	ggxReflectionAdjustedPDFFunc,
	ggxRefractionAdjustedPDFFunc,
	ggxShadowMaskG1Func,
} from './ggx.wgsl.js';
import { constants, surfaceRecordStruct, transmissionFresnelStruct } from './structs.wgsl.js';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';

// Correct the shading normal to prevent scattering rays below the geometry surface by bending
// the normal towards the geometry normal such that the perfect reflection ray is just above the
// geometry surface. Ported verbatim from Blender Cycles "ensure_valid_specular_reflection"
// (src/kernel/closure/bsdf_util.h), which is derived from from the Iray paper (Keller et al. 2017, A.3).
//
// This only guarantees perfect reflection rays are above surface - other diffuse rays, for example,
// flipped when sampling.
const ensureValidReflectionNormal = pipelineWgslTagFn(149)`

	fn ensureValidReflectionNormal( n: vec3f, ng: vec3f, i: vec3f ) -> vec3f {

		// reflected ray
		let R = 2.0 * dot( n, i ) * n - i;

		let Iz = dot( i, ng );

		// reflection rays may always be at least as shallow as the incoming ray
		let threshold = min( 0.9 * Iz, 0.01 );
		if ( dot( ng, R ) >= threshold ) {

			return n;

		}

		// Form coordinate system with Ng as the Z axis and N inside the X-Z-plane.
   		// The X axis is found by normalizing the component of N that's orthogonal to Ng.
   		// The Y axis isn't actually needed.
		let orthoN = n - dot( n, ng ) * ng;
		let orthoLen = length( orthoN );
		let X = select( n, orthoN / orthoLen, orthoLen != 0.0 );

		// Calculate N.z and N.x in the local coordinate system.
		//
		// The goal of this computation is to find a N' that is rotated towards Ng just enough
   		// to lift R' above the threshold (here called t), therefore dot(R', Ng) = t.
		//
		// See the Blender implementation for a full description of the solution.
		let Ix = dot( i, X );

		let a = Ix * Ix + Iz * Iz;
		let b = 2.0 * ( a + Iz * threshold );
		let c = ( threshold + Iz ) * ( threshold + Iz );

		// only one root is valid; the sign of Ix selects it
		var Nz2: f32;
		if ( Ix < 0.0 ) {

			Nz2 = 0.25 * ( b + sqrt( max( b * b - 4.0 * a * c, 0.0 ) ) ) / a;

		} else {

			Nz2 = 0.25 * ( b - sqrt( max( b * b - 4.0 * a * c, 0.0 ) ) ) / a;

		}

		let Nx = sqrt( max( 1.0 - Nz2, 0.0 ) );
		let Nz = sqrt( max( Nz2, 0.0 ) );

		return Nx * X + Nz * ng;

	}

`;

// Adjusts the shading normal such that the provided view normal is guaranteed to be in
// positive hemisphere by rotating it towards the view direction just enough so view is
// just glancing the surface as a small epsilon if needed.
const ensureValidViewNormal = pipelineWgslTagFn(150)`

	fn ensureValidViewNormal( n: vec3f, ng: vec3f, view: vec3f ) -> vec3f {

		// ensure the view is at least slightly above the surface normal hemisphere
		const MIN_VIEW_COS = 1e-6;

		// if we're positive already then early out
		let c = dot( n, view );
		if ( c > MIN_VIEW_COS ) {

			return n;

		}

		// perpendicular vector to normal and view
		var perp = n - c * view;
		let perpLen = length( perp );

		// if we reach here that means that he view is nearly the opposite direction and
		// we can't derive plan to rotate on towards the view, so just use the geometry
		// normal. This could likely only really happen with a severe normal map application.
		if ( perpLen <= 1e-6 ) {

			return ng;

		}

		perp = perp / perpLen;

		// rotate the normal to be at larger than the above threshold - pythagorean
		// theorem for generating normal with threshold*view component
		return MIN_VIEW_COS * view + sqrt( 1.0 - MIN_VIEW_COS * MIN_VIEW_COS ) * perp;

	}

`;

// Builds getSurfaceRecord using the given per-instance sampleTexel and uv channel lookup
// Sentinel for a disabled glossy filter ( FLT_MAX, mirroring Cycles ): the blur term
// clamps to zero for any path pdf when the inverted filter value is this large.
export const FILTER_GLOSSY_DISABLED = 3.402823466e38;

export const getSurfaceRecordFunc = ( sampleTexel, getUvFromChannel, getColor ) => pipelineWgslFn(151, ["\n\n\tfn getSurfaceRecord(\n\t\tmaterial: Material,\n\t\tvertexData: bvh_GeometryStruct,\n\t\tside: f32,\n\t\tfaceNormal: vec3f,\n\t\tview: vec3f,\n\t\tblurRoughness: f32,\n\t) -> SurfaceRecord {\n\n\t\t// \"faceNormal\" is provided on the hit side so flip it back to the geometric side\n\t\tvar normal = faceNormal * side;\n\t\tif ( material.flatShading == 0 ) {\n\n\t\t\tnormal = normalize( vertexData.normal.xyz );\n\n\t\t}\n\n\t\tvar baseNormal = normal;\n\t\tif ( material.normalMap != -1 ) {\n\n\t\t\t// some provided tangents can be malformed (0, 0, 0) causing the normal to be degenerate\n\t\t\t// resulting in NaNs and slow path tracing.\n\t\t\tif ( length( vertexData.tangent.xyz ) > 0.0 && vertexData.tangent.w != 0.0 ) {\n\n\t\t\t\t// TODO: consider re-orthonormalizing against the normal here since attribute\n\t\t\t\t// interpolation could result in drift.\n\t\t\t\tlet tangent = normalize( vertexData.tangent.xyz );\n\t\t\t\tlet bitangent = normalize( cross( baseNormal, tangent ) * vertexData.tangent.w );\n\t\t\t\tlet vTBN = mat3x3f( tangent, bitangent, baseNormal );\n\n\t\t\t\tlet uvPrime = material.normalMapTransform * vec3( getUvFromChannel( vertexData, material.normalMap ), 1.0 );\n\t\t\t\tvar texNormal = sampleTexel( uvPrime.xy, material.normalMap, 0 ).xyz;\n\t\t\t\ttexNormal = texNormal * 2.0 - 1.0;\n\t\t\t\ttexNormal = texNormal * vec3f( material.normalScale, 1.0 );\n\t\t\t\tnormal = normalize( vTBN * texNormal );\n\n\t\t\t}\n\n\t\t}\n\n\t\tnormal *= side;\n\n\t\tnormal = ensureValidViewNormal( normal, faceNormal, view );\n\n\t\tnormal = ensureValidReflectionNormal( normal, faceNormal, view );\n\n\t\tvar albedo = vec4( material.color, material.opacity );\n\t\tif ( material.vertexColors == 1 ) {\n\n\t\t\talbedo *= getColor( vertexData );\n\n\t\t}\n\n\t\tif ( material.map != -1 ) {\n\n\t\t\tlet uvPrime = material.mapTransform * vec3f( getUvFromChannel( vertexData, material.map ), 1 );\n\t\t\talbedo *= sampleTexel( uvPrime.xy, material.map, 0 );\n\n\t\t}\n\n\t\tif ( material.alphaMap != -1 ) {\n\n\t\t\tlet uvPrime = material.alphaMapTransform * vec3f( getUvFromChannel( vertexData, material.alphaMap ), 1 );\n\t\t\talbedo.a *= sampleTexel( uvPrime.xy, material.alphaMap, 0 ).g;\n\n\t\t}\n\n\t\tvar roughness = material.roughness;\n\t\tif ( material.roughnessMap != -1 ) {\n\n\t\t\tlet uvPrime = material.roughnessMapTransform * vec3f( getUvFromChannel( vertexData, material.roughnessMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.roughnessMap, 0 );\n\t\t\troughness *= texColor.g;\n\n\t\t}\n\n\t\tvar metalness = material.metalness;\n\t\tif ( material.metalnessMap != -1 ) {\n\n\t\t\tlet uvPrime = material.metalnessMapTransform * vec3f( getUvFromChannel( vertexData, material.metalnessMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.metalnessMap, 0 );\n\t\t\tmetalness *= texColor.b;\n\n\t\t}\n\n\t\tvar emission = material.emissiveIntensity * material.emissive;\n\t\tif ( material.emissiveMap != -1 ) {\n\n\t\t\tlet uvPrime = material.emissiveMapTransform * vec3f( getUvFromChannel( vertexData, material.emissiveMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.emissiveMap, 0 );\n\t\t\temission *= texColor.rgb;\n\n\t\t}\n\n\t\tvar transmission = material.transmission;\n\t\tif ( material.transmissionMap != -1 ) {\n\n\t\t\tlet uvPrime = material.transmissionMapTransform * vec3f( getUvFromChannel( vertexData, material.transmissionMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.transmissionMap, 0 );\n\t\t\ttransmission *= texColor.r;\n\n\t\t}\n\n\t\tvar clearcoat = material.clearcoat;\n\t\tif ( material.clearcoatMap != -1 ) {\n\n\t\t\tlet uvPrime = material.clearcoatMapTransform * vec3f( getUvFromChannel( vertexData, material.clearcoatMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.clearcoatMap, 0 );\n\t\t\tclearcoat *= texColor.r;\n\n\t\t}\n\n\t\tvar clearcoatRoughness = material.clearcoatRoughness;\n\t\tif ( material.clearcoatRoughnessMap != -1 ) {\n\n\t\t\tlet uvPrime = material.clearcoatRoughnessMapTransform * vec3f( getUvFromChannel( vertexData, material.clearcoatRoughnessMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.clearcoatRoughnessMap, 0 );\n\t\t\tclearcoatRoughness *= texColor.g;\n\n\t\t}\n\n\t\tvar clearcoatNormal = baseNormal;\n\t\tif ( material.clearcoatNormalMap != -1 ) {\n\n\t\t\t// some provided tangents can be malformed (0, 0, 0) causing the normal to be degenerate\n\t\t\t// resulting in NaNs and slow path tracing.\n\t\t\tif ( length( vertexData.tangent.xyz ) > 0.0 && vertexData.tangent.w != 0.0 ) {\n\n\t\t\t\t// TODO: consider re-orthonormalizing against the normal here since attribute\n\t\t\t\t// interpolation could result in drift.\n\t\t\t\tlet tangent = normalize( vertexData.tangent.xyz );\n\t\t\t\tlet bitangent = normalize( cross( baseNormal, tangent ) * vertexData.tangent.w );\n\t\t\t\tlet vTBN = mat3x3f( tangent, bitangent, baseNormal );\n\n\t\t\t\tlet uvPrime = material.clearcoatNormalMapTransform * vec3( getUvFromChannel( vertexData, material.clearcoatNormalMap ), 1.0 );\n\t\t\t\tvar texNormal = sampleTexel( uvPrime.xy, material.clearcoatNormalMap, 0 ).xyz;\n\t\t\t\ttexNormal = texNormal * 2.0 - 1.0;\n\t\t\t\ttexNormal = texNormal * vec3f( material.clearcoatNormalScale, 1.0 );\n\t\t\t\tclearcoatNormal = normalize( vTBN * texNormal );\n\n\t\t\t}\n\n\t\t}\n\n\t\tclearcoatNormal *= side;\n\n\t\tclearcoatNormal = ensureValidViewNormal( clearcoatNormal, faceNormal, view );\n\n\t\tclearcoatNormal = ensureValidReflectionNormal( clearcoatNormal, faceNormal, view );\n\n\t\tvar sheenColor = material.sheenColor;\n\t\tif ( material.sheenColorMap != -1 ) {\n\n\t\t\tlet uvPrime = material.sheenColorMapTransform * vec3f( getUvFromChannel( vertexData, material.sheenColorMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.sheenColorMap, 0 );\n\t\t\tsheenColor *= texColor.rgb;\n\n\t\t}\n\n\t\tvar sheenRoughness = material.sheenRoughness;\n\t\tif ( material.sheenRoughnessMap != -1 ) {\n\n\t\t\tlet uvPrime = material.sheenRoughnessMapTransform * vec3f( getUvFromChannel( vertexData, material.sheenRoughnessMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.sheenRoughnessMap, 0 );\n\t\t\tsheenRoughness *= texColor.r;\n\n\t\t}\n\n\t\tvar iridescence = material.iridescence;\n\t\tif ( material.iridescenceMap != -1 ) {\n\n\t\t\tlet uvPrime = material.iridescenceMapTransform * vec3f( getUvFromChannel( vertexData, material.iridescenceMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.iridescenceMap, 0 );\n\t\t\tiridescence *= texColor.r;\n\n\t\t}\n\n\t\tvar iridescenceThickness = material.iridescenceThicknessMaximum;\n\t\tif ( material.iridescenceThicknessMap != -1 ) {\n\n\t\t\tlet uvPrime = material.iridescenceThicknessMapTransform * vec3f( getUvFromChannel( vertexData, material.iridescenceThicknessMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.iridescenceThicknessMap, 0 );\n\n\t\t\tiridescenceThickness = mix(\n\t\t\t\tmaterial.iridescenceThicknessMinimum,\n\t\t\t\tmaterial.iridescenceThicknessMaximum,\n\t\t\t\ttexColor.g,\n\t\t\t);\n\n\t\t}\n\n\t\tvar specularColor = material.specularColor;\n\t\tif ( material.specularColorMap != -1 ) {\n\n\t\t\tlet uvPrime = material.specularColorMapTransform * vec3f( getUvFromChannel( vertexData, material.specularColorMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.specularColorMap, 0 );\n\t\t\tspecularColor *= texColor.rgb;\n\n\t\t}\n\n\t\tvar specularIntensity = material.specularIntensity;\n\t\tif ( material.specularIntensityMap != -1 ) {\n\n\t\t\tlet uvPrime = material.specularIntensityMapTransform * vec3f( getUvFromChannel( vertexData, material.specularIntensityMap ), 1 );\n\t\t\tlet texColor = sampleTexel( uvPrime.xy, material.specularIntensityMap, 0 );\n\t\t\tspecularIntensity *= texColor.a;\n\n\t\t}\n\n\t\t// extract the anisotropy magnitude vector in tangent space\n\t\tvar anisotropyDirVec = material.anisotropy * vec2f( cos( material.anisotropyRotation ), sin( material.anisotropyRotation ) );\n\t\tif ( material.anisotropyMap != -1 ) {\n\n\t\t\tlet uvPrime = material.anisotropyMapTransform * vec3f( getUvFromChannel( vertexData, material.anisotropyMap ), 1.0 );\n\t\t\tlet aniTex = sampleTexel( uvPrime.xy, material.anisotropyMap, 0 );\n\n\t\t\t// map rg encode the direction ([-1,1]), b the strength; rotate + scale by the material anisotropy.\n\t\t\tlet mapDir = aniTex.rg * 2.0 - vec2f( 1.0 );\n\t\t\tlet mapStr = aniTex.b;\n\t\t\tlet mapLen = length( mapDir );\n\t\t\tif ( mapLen > EPSILON ) {\n\n\t\t\t\tanisotropyDirVec = mat2x2f(\n\t\t\t\t\tanisotropyDirVec.x, anisotropyDirVec.y,\n\t\t\t\t\t- anisotropyDirVec.y, anisotropyDirVec.x\n\t\t\t\t) * mapStr * mapDir / mapLen;\n\n\t\t\t}\n\n\t\t}\n\n\t\t// adjust the surface basis to be oriented along the anisotropic vector\n\t\tlet anisotropyStrength = length( anisotropyDirVec );\n\t\tvar surfaceBasis = getBasisFromNormal( normal );\n\t\tif ( anisotropyStrength > 0.0 && length( vertexData.tangent.xyz ) > 0.0 && vertexData.tangent.w != 0.0 ) {\n\n\t\t\tlet anisotropyDir = anisotropyDirVec / anisotropyStrength;\n\n\t\t\t// re-orthonormalize the tangent against the shading normal so the frame stays orthonormal -\n\t\t\t// normal map adjustments will cause the normal and tangent to become non-orthonormal.\n\t\t\tlet tangent = normalize( vertexData.tangent.xyz - normal * dot( normal, vertexData.tangent.xyz ) );\n\t\t\tlet bitangent = cross( normal, tangent ) * vertexData.tangent.w;\n\n\t\t\tsurfaceBasis = mat3x3f(\n\t\t\t\ttangent * anisotropyDir.x + bitangent * anisotropyDir.y,\n\t\t\t\tbitangent * anisotropyDir.x - tangent * anisotropyDir.y,\n\t\t\t\tnormal\n\t\t\t);\n\n\t\t}\n\n\t\tvar surf: SurfaceRecord;\n\n\t\tsurf.volumeParticle = false;\n\t\tsurf.faceNormal = faceNormal;\n\t\tsurf.normal = normal;\n\n\t\tsurf.metalness = metalness;\n\t\tsurf.diffuseRoughness = material.diffuseRoughness;\n\t\tsurf.color = albedo.rgb;\n\t\tsurf.emission = emission;\n\n\t\tsurf.opacity = select( 1.0, albedo.a, material.transparent != 0 );\n\n\t\tsurf.ior = material.ior;\n\t\tsurf.transmission = transmission;\n\t\tsurf.thinWall = material.thinWall == 1;\n\t\tsurf.attenuationColor = material.attenuationColor;\n\t\tsurf.attenuationDistance = material.attenuationDistance;\n\n\t\tsurf.clearcoatNormal = clearcoatNormal;\n\t\tsurf.clearcoat = clearcoat;\n\n\t\tsurf.iridescence = iridescence;\n\t\tsurf.iridescenceIor = material.iridescenceIor;\n\t\tsurf.iridescenceThickness = iridescenceThickness;\n\n\t\tsurf.specularColor = specularColor;\n\t\tsurf.specularIntensity = specularIntensity;\n\n\t\tsurf.sheen = material.sheen;\n\t\tsurf.sheenColor = sheenColor;\n\n\t\tlet minRoughness = max( MIN_ROUGHNESS, blurRoughness );\n\t\tsurf.roughness = clamp( roughness, minRoughness, 1.0 );\n\t\tsurf.clearcoatRoughness = clamp( clearcoatRoughness, minRoughness, 1.0 );\n\t\tsurf.sheenRoughness = clamp( sheenRoughness, minRoughness, 1.0 );\n\t\tsurf.anisotropy = anisotropyStrength;\n\n\t\t// frontFace is used to determine transmissive properties and PDF. If no transmission is used\n\t\t// then we can just always assume this is a front face.\n\t\tsurf.frontFace = side == 1.0 || transmission == 0.0;\n\t\tif ( material.thinWall == 1 || surf.frontFace ) {\n\n\t\t\tsurf.eta = 1.0 / material.ior;\n\n\t\t} else {\n\n\t\t\tsurf.eta = material.ior;\n\n\t\t}\n\t\tsurf.f0 = iorToF0( surf.eta );\n\n\t\t// get the normal frames\n\t\tsurf.normalBasis = surfaceBasis;\n\t\tsurf.normalInvBasis = inverse( surf.normalBasis );\n\n\t\tsurf.clearcoatBasis = getBasisFromNormal( surf.clearcoatNormal );\n\t\tsurf.clearcoatInvBasis = inverse( surf.clearcoatBasis );\n\n\t\treturn surf;\n\t}\n\n"], [], [
	inverseMat3x3Func,
	iorToF0Func,
	getBasisFromNormalFunc,
	sampleTexel,
	getUvFromChannel,
	getColor,
	surfaceRecordStruct,
	constants,
	ensureValidReflectionNormal,
	ensureValidViewNormal,
]);

/*
 *
 * N 			  : Macronormal of the surface
 * V ( wo ) : View direction
 * L ( wi ) : Light direction
 * H ( wh ) : Halfvector between V and L, micronormal of the surface in ggx
 * f0       : Amount of light reflected when looking at a surface head on - "fresnel 0"
 * f90      : Amount of light reflected at grazing angles
 *
 */

// Disney Diffuse BRDF without subsurface approximation
export const diffuseBrdfFunc = pipelineWgslFn(152, ["\n\n\tfn diffuseBrdf( NdotV: f32, NdotL: f32, VdotH: f32, surf: SurfaceRecord ) -> vec3f {\n\n\t\t// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf\n\t\t// See equation (4)\n\n\t\tlet fl = schlickFresnel( NdotL, 0.0 );\n\t\tlet fv = schlickFresnel( NdotV, 0.0 );\n\n\t\tlet alpha = surf.roughness * surf.roughness;\n\t\tlet bias = mix( 0.0, 0.5, alpha ) - 1;\n\t\tlet energyFactor = mix( 1.0, 1.0 / 1.51, alpha );\n\n\t\tlet rr = 2.0 * alpha * VdotH * VdotH;\n\t\tlet retro = rr * ( fl + fv + fl * fv * ( rr + 2.0 * bias ) );\n\t\tlet fresnel = ( 1.0 + bias * fl ) * ( 1.0f + bias * fv );\n\n\t\t// TODO: subsurface approx?\n\n\t\treturn energyFactor * ( surf.color / PI ) * ( retro + fresnel );\n\n\t}\n\n"], [], [ constants, schlickFresnelFunc, surfaceRecordStruct ]);

// Lambertian diffuse BRDF with cosine distribution
export const lambertBrdfFunc = pipelineWgslFn(153, ["\n\n\tfn lambertBrdf( NdotV: f32, NdotL: f32, VdotH: f32, surf: SurfaceRecord ) -> vec3f {\n\n\t\treturn surf.color / PI;\n\n\t}\n\n"], [], [ constants, surfaceRecordStruct ]);

export const specularBrdfFunc = pipelineWgslFn(154, ["\n\n\tfn specularBrdf( V: vec3f, L: vec3f, H: vec3f, alpha: vec2f ) -> vec3f {\n\n\t\tlet alphaT = alpha.x;\n\t\tlet alphaB = alpha.y;\n\n\t\tlet Vis = ggxSmithVisibility( V, L, alpha );\n\t\tlet D = ggxDistribution( H, alpha );\n\n\t\treturn vec3f( D * Vis );\n\n\t}\n\n"], [], [ ggxSmithVisibilityFunc, ggxDistributionFunc ]);

// PBRT-v4 treats matched media as perfect specular transmission because the rough dielectric
// half-vector and its Jacobian are undefined when eta is one:
// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/bxdfs.cpp
export const isMatchedIorFunc = pipelineWgslFn(155, ["\n\n\tfn isMatchedIor( eta: f32 ) -> bool {\n\n\t\treturn abs( eta - 1.0 ) < EPSILON;\n\n\t}\n\n"], [], [ constants ]);

export const specularBtdfFunc = pipelineWgslFn(156, ["\n\n\tfn specularBtdf( V: vec3f, L: vec3f, H: vec3f, alpha: vec2f, eta: f32 ) -> vec3f {\n\n\t\tlet NdotV = V.z;\n\t\tlet NdotL = L.z;\n\t\tlet HdotV = dot( V, H );\n\t\tlet HdotL = dot( L, H );\n\n\t\t// Heaviside function for G term\n\t\tif ( NdotV * HdotV < 0.0 || NdotL * HdotL < 0.0 || HdotV * HdotL > 0.0 ) {\n\n\t\t\treturn vec3f( 0.0 );\n\n\t\t}\n\n\t\tlet G1_i = ggxShadowMaskG1( V, alpha );\n\t\tlet G1_o = ggxShadowMaskG1( L, alpha );\n\n\t\t// separable G2 product, no height correlation\n\t\tlet denom = eta * HdotV + HdotL;\n\t\tlet Vis = G1_i * G1_o * abs( HdotV ) * abs( HdotL ) /\n\t\t\t( abs( NdotV ) * abs( NdotL ) * denom * denom );\n\n\t\tlet F = dielectricFresnel( abs( HdotV ), eta );\n\n\t\tlet D = ggxDistribution( H, alpha );\n\n\t\treturn vec3f( ( 1 - F ) * D * Vis );\n\n\t}\n\n"], [], [ ggxShadowMaskG1Func, ggxDistributionFunc, dielectricFresnelFunc ]);

// Effective roughness for thin walled transmission, from page 40 (note the slides' 3.7 is a typo for 3.4) -
// referenced from Blender Cycles:
// https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
export const thinWallTransmissionRoughnessFunc = pipelineWgslFn(157, ["\n\n\tfn thinWallTransmissionRoughness( alpha: f32, eta: f32 ) -> f32 {\n\n\t\tlet t = 3.4 * ( eta - 1.0 ) * ( eta - 0.5 ) * ( eta - 0.5 ) / ( eta * eta * eta );\n\t\treturn saturate( alpha * sqrt( max( t, 0.0 ) ) );\n\n\t}\n\n"], [], []);

// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_volume/README.md#attenuation
export const transmissionAttenuationFunc = pipelineWgslFn(158, ["\n\n\tfn transmissionAttenuation( dist: f32, attColor: vec3f, attDist: f32 ) -> vec3f {\n\n\t\treturn pow( attColor, vec3f( dist / attDist ) );\n\n\t}\n\n"], [], []);

// Continuous Cauchy IOR approximation derived from the Fraunhofer C, d, and F spectral lines:
// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_dispersion
// https://github.com/bsdorra/slang-pbr/blob/c12046ec3b9628bcfc384ed2c275c92d007e5d7e/src/models/enterprise/components/transmission.slang#L20-L27
export const DISPERSION_MIN_WAVELENGTH = 360;
export const DISPERSION_MAX_WAVELENGTH = 830;

export const dispersionIorFunc = pipelineWgslFn(159, ["\n\n\tfn dispersionIor( ior: f32, dispersion: f32, wavelength: f32 ) -> f32 {\n\n\t\tlet nd = max( ior, 1.0 );\n\t\tlet scale = ( nd - 1.0 ) * max( dispersion, 0.0 ) / 20.0;\n\t\tlet lambda = clamp( wavelength, ",".0, ",".0 );\n\t\treturn max( nd + scale * ( 523655.0 / ( lambda * lambda ) - 1.5168 ), 1.0 );\n\n\t}\n\n"], [DISPERSION_MIN_WAVELENGTH, DISPERSION_MAX_WAVELENGTH], []);

export const applyDispersionFunc = pipelineWgslTagFn(160)`

	fn applyDispersion( surf: ptr<function, ${ surfaceRecordStruct }>, dispersion: f32, wavelength: f32 ) -> void {

		surf.ior = ${ dispersionIorFunc }( surf.ior, dispersion, wavelength );
		surf.eta = select( surf.ior, 1.0 / surf.ior, surf.thinWall || surf.frontFace );
		surf.f0 = ${ iorToF0Func }( surf.eta );

	}

`;

// A path keeps one uniformly sampled hero wavelength after its first dispersive interaction.
// The CIE 1931 fit is converted to linear Rec. 709, clamped to the RGB gamut, and normalized so
// every channel has an expected weight of one over the uniformly sampled wavelength interval.
// This preserves the expected value of the renderer's existing RGB throughput while providing
// continuous hues. The renderer still uses RGB material and light data rather than spectral data.
// CIE fit: https://jcgt.org/published/0002/02/01/
// Hero wavelength sampling:
// https://pbr-book.org/4ed/Textures_and_Materials/Material_Interface_and_Implementations.html#sec:dielectric-material
// https://doi.org/10.1111/cgf.12419
export const dispersionColorWeightFunc = pipelineWgslFn(161, ["\n\n\tfn dispersionColorWeight( wavelength: f32 ) -> vec3f {\n\n\t\tlet lambda = clamp( wavelength, ",".0, ",".0 );\n\n\t\tlet x1 = ( lambda - 442.0 ) * select( 0.0374, 0.0624, lambda < 442.0 );\n\t\tlet x2 = ( lambda - 599.8 ) * select( 0.0323, 0.0264, lambda < 599.8 );\n\t\tlet x3 = ( lambda - 501.1 ) * select( 0.0382, 0.0490, lambda < 501.1 );\n\t\tlet x = 0.362 * exp( - 0.5 * x1 * x1 ) + 1.056 * exp( - 0.5 * x2 * x2 ) - 0.065 * exp( - 0.5 * x3 * x3 );\n\n\t\tlet y1 = ( lambda - 568.8 ) * select( 0.0247, 0.0213, lambda < 568.8 );\n\t\tlet y2 = ( lambda - 530.9 ) * select( 0.0322, 0.0613, lambda < 530.9 );\n\t\tlet y = 0.821 * exp( - 0.5 * y1 * y1 ) + 0.286 * exp( - 0.5 * y2 * y2 );\n\n\t\tlet z1 = ( lambda - 437.0 ) * select( 0.0278, 0.0845, lambda < 437.0 );\n\t\tlet z2 = ( lambda - 459.0 ) * select( 0.0725, 0.0385, lambda < 459.0 );\n\t\tlet z = 1.217 * exp( - 0.5 * z1 * z1 ) + 0.681 * exp( - 0.5 * z2 * z2 );\n\n\t\tlet linearRgb = max( vec3f(\n\t\t\t3.2404542 * x - 1.5371385 * y - 0.4985314 * z,\n\t\t\t- 0.9692660 * x + 1.8760108 * y + 0.0415560 * z,\n\t\t\t0.0556434 * x - 0.2040259 * y + 1.0572252 * z,\n\t\t), vec3f( 0.0 ) );\n\n\t\treturn linearRgb * vec3f( 2.6699448, 4.0708066, 4.2980494 );\n\n\t}\n\n"], [DISPERSION_MIN_WAVELENGTH, DISPERSION_MAX_WAVELENGTH], []);

// Dielectric layer fresnel operator that supports custom f0 color, specular weight.
// Based on the specular color specification:
// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_specular
export const fresnelMixFunc = pipelineWgslFn(162, ["\n\n\tfn fresnelMix( VdotH: f32, f0Color: vec3f, ior: f32, eta: f32, weight: f32, base: vec3f, layer: vec3f ) -> vec3f {\n\n\t\tvar f0 = iorToF0( ior ) * f0Color;\n\t\tf0 = min( f0, vec3f( 1.0 ) );\n\n\t\t// reflect all light on total internal reflection, matching the WebGL evaluateFresnel function\n\t\tvar fr = vec3f( 1.0 );\n\t\tif ( ! totalInternalReflection( abs( VdotH ), eta ) ) {\n\n\t\t\tfr = schlickFresnelVec( abs( VdotH ), f0, vec3f( 1.0 ) );\n\n\t\t}\n\n\t\tlet maxFr = max( max( fr.r, fr.g ), fr.b );\n\n\t\treturn ( 1.0 - weight * maxFr ) * base + weight * fr * layer;\n\n\t}\n\n"], [], [ schlickFresnelVecFunc, iorToF0Func, totalInternalReflectionFunc ]);

const XYZ_TO_REC709 = mat3(
	3.2404542, - 0.9692660, 0.0556434,
	- 1.5371385, 1.8760108, - 0.2040259,
	- 0.4985314, 0.0415560, 1.0572252,
);

const evalSensitivityFunc = pipelineWgslTagFn(163)`

	fn evalSensitivity( OPD: f32, shift: vec3f ) -> vec3f {

		let phase = 2.0 * ${ Math.PI } * OPD * 1.0e-9;
		const val = vec3(5.4856e-13, 4.4201e-13, 5.2481e-13);
		const pos = vec3(1.6810e+06, 1.7953e+06, 2.2084e+06);
		const _var = vec3(4.3278e+09, 9.3046e+09, 6.6121e+09);

		var xyz = val * sqrt(2.0 * ${ Math.PI } * _var) * cos(pos * phase + shift) * exp(-phase * phase * _var);
		xyz.x += 9.7470e-14 * sqrt(2.0 * ${ Math.PI } * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * phase * phase);
		xyz /= 1.0685e-7;

		let rgb = ${ XYZ_TO_REC709 } * xyz;
		return rgb;

	}

`;

// Reference: Belcour/Barla, 2017
// https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
// This is a simplified model that ignores light polarization and uses fresnel approximation
export const iridescentFresnelFunc = pipelineWgslFn(164, ["\n\n\tfn iridescentFresnel(\n\t\tcosTheta1: f32, baseF0: vec3f, iridescenceIor: f32,\n\t\toutsideIor: f32, iridescenceThickness: f32,\n\t) -> vec3f {\n\n\t\tlet sinTheta2Sq = pow( outsideIor / iridescenceIor, 2.0 ) * ( 1.0 - pow( cosTheta1, 2.0 ) );\n\t\tlet cosTheta2Sq = 1.0 - sinTheta2Sq;\n\n\t\t// Handle total internal reflection\n\t\tif ( cosTheta2Sq < 0.0 ) {\n\n\t\t\treturn vec3( 1.0 );\n\n\t\t}\n\n\t\tlet cosTheta2 = sqrt( cosTheta2Sq );\n\n\t\t// First interface: air -> iridescent thin film\n\t\tlet R0 = iorToF0General( iridescenceIor, outsideIor );\n\t\tlet R12 = schlickFresnel( cosTheta1, R0 );\n\t\tlet R21 = R12;\n\t\tlet T121 = 1.0 - R12;\n\t\tlet phi12 = select( 0.0, PI, iridescenceIor < outsideIor );\n\t\tlet phi21 = PI - phi12;\n\n\t\t// Second interface: iridescent thin film -> base material\n\t\tlet baseIor = fresnel0ToIor( clamp( baseF0, vec3( 0.0 ), vec3( 0.9999 ) ) ); // guard against 1.0\n\t\tlet R1 = iorToF0GeneralVec( baseIor, vec3( iridescenceIor ) );\n\t\tvar R23 = schlickFresnelVec( cosTheta2, R1, vec3( 1.0 ) );\n\n\t\t// Handle total internal reflection at the film -> base interface\n\t\t// NOTE: Added separately from the original implementation. Is this correct?\n\t\tR23 = select( R23, vec3( 1.0 ), totalInternalReflectionVec( cosTheta2, vec3( iridescenceIor ) / baseIor ) );\n\t\tlet phi23 = select( vec3( 0.0 ), vec3( PI ), baseIor < vec3( iridescenceIor ) );\n\n\t\t// Phase shift\n\t\tlet OPD = 2.0 * iridescenceIor * iridescenceThickness * cosTheta2;\n\t\tlet phi = vec3( phi21 ) + phi23;\n\n\t\t// Analytical integration\n\t\t// Compound terms\n\t\tlet R123 = clamp( R12 * R23, vec3( 1e-5 ), vec3( 0.9999 ) );\n\t\tlet r123 = sqrt( R123 );\n\t\tlet Rs = T121 * T121 * R23 / ( vec3( 1.0 ) - R123 );\n\n\t\t// Reflectance term for m = 0 (DC term amplitude)\n\t\tlet C0 = R12 + Rs;\n\t\tvar I = C0;\n\n\t\t// Reflectance term for m > 0 (pairs of diracs)\n\t\tvar Cm = Rs - T121;\n\t\tfor (var m = 1; m <= 2; m += 1) {\n\n\t\t\tCm *= r123;\n\t\t\tlet Sm = 2.0 * evalSensitivity( f32( m ) * OPD, f32( m ) * phi );\n\t\t\tI += Cm * Sm;\n\n\t\t}\n\n\t\treturn max( I, vec3(0.0) );\n\n\t}\n\n"], [], [ iorToF0GeneralFunc, iorToF0GeneralVecFunc, schlickFresnelFunc, fresnel0ToIorFunc, evalSensitivityFunc, totalInternalReflectionVecFunc ]);

// Reflection and transmission coefficients for the transmissive dielectric layer. The scalar
// branch probability follows Cycles: choose between the two colored coefficients by their mean
// energy while retaining the full RGB coefficient in the path throughput.
export const transmissionFresnelFunc = ( iridescentFresnel = iridescentFresnelFunc ) => pipelineWgslTagFn(165)`

	fn transmissionFresnel( cosTheta: f32, surf: ${ surfaceRecordStruct } ) -> ${ transmissionFresnelStruct } {

		var result: ${ transmissionFresnelStruct };
		let cosine = saturate( abs( cosTheta ) );
		let matchedIor = ${ isMatchedIorFunc }( surf.eta );
		let airIncident = surf.thinWall || surf.frontFace;
		let dielectricFresnel = ${ dielectricFresnelFunc }( cosine, surf.eta );

		// Keep matched media transparent unless a real thin-film layer is present. This avoids
		// introducing the non-physical Schlick F90 reflection produced by KHR_materials_specular
		// when the substrate itself has no IOR boundary.
		var reflectance = vec3f( 0.0 );
		if ( ! matchedIor ) {

			let specularF0 = min( surf.f0 * surf.specularColor, vec3f( 1.0 ) );
			let dielectricReflectance = select(
				vec3f( dielectricFresnel ),
				${ schlickFresnelVecFunc }( cosine, specularF0, vec3f( 1.0 ) ),
				airIncident,
			);
			reflectance = surf.specularIntensity * dielectricReflectance;

		}

		reflectance = clamp( reflectance, vec3f( 0.0 ), vec3f( 1.0 ) );
		var transmittance = vec3f( 1.0 ) - reflectance;

		if ( surf.iridescence > 0.0 ) {

			// The media on either side of the film are air outside and the volume interior as
			// the base. Swap them on interior hits so film-interface TIR remains possible.
			let outsideIor = select( surf.ior, 1.0, airIncident );
			let filmBaseIor = select( 1.0, surf.ior, airIncident );
			let filmReflectance = clamp( ${ iridescentFresnel }(
				cosine,
				vec3f( ${ iorToF0Func }( filmBaseIor ) ),
				surf.iridescenceIor,
				outsideIor,
				surf.iridescenceThickness,
			), vec3f( 0.0 ), vec3f( 1.0 ) );

			reflectance = mix( reflectance, filmReflectance, surf.iridescence );
			transmittance = mix( transmittance, vec3f( 1.0 ) - filmReflectance, surf.iridescence );

		}

		result.reflectance = reflectance;
		result.transmittance = transmittance;

		// Include the base-color tint in the transmission energy, as Cycles does when it
		// computes the glass reflection probability from its two tinted coefficients.
		let reflectEnergy = dot( reflectance, vec3f( 1.0 / 3.0 ) );
		let transmitEnergy = dot( transmittance * surf.color, vec3f( 1.0 / 3.0 ) );
		let totalEnergy = reflectEnergy + transmitEnergy;
		result.reflectProbability = select( 0.0, reflectEnergy / max( totalEnergy, EPSILON ), totalEnergy > EPSILON );

		return result;

	}

`;

export const conductorFresnelFunc = pipelineWgslFn(166, ["\n\n\tfn conductorFresnel( VdotH: f32, f0: vec3f, specular: vec3f ) -> vec3f {\n\n\t\treturn specular * schlickFresnelVec( abs( VdotH ), f0, vec3f( 1 ) );\n\n\t}\n\n"], [], [ schlickFresnelVecFunc ]);

export const fresnelCoatFunc = pipelineWgslFn(167, ["\n\n\tfn fresnelCoat( VdotNc: f32, ior: f32, base: vec3f, layer: vec3f, weight: f32 ) -> vec3f {\n\n\t\tlet f0 = iorToF0( ior );\n\t\tlet F = schlickFresnel( abs( VdotNc ), f0 );\n\n\t\treturn mix( base, layer, weight * F );\n\n\t}\n\n"], [], [ iorToF0Func, schlickFresnelFunc ]);

// GGX Multibounce compensation using Turquin's method. Bakes the directional albedo of the
// reflection lobe, optionally fresnel-weighted, and optionally including the refraction lobe
// for the full transmissive bsdf. "eta" is the incident over transmitted ior ratio.
export const turquinIntegralFn = outputTarget => pipelineWgslTagFn(168)`

	fn albedo(
		eta: f32,
		includeFresnel: bool,
		includeRefraction: bool,
		globalId: vec3u,
		layer: u32,
	) -> void {

		// The matched-IOR limit is a unit-energy delta transmission event, so it cannot
		// be integrated with the rough BTDF below.
		if ( includeRefraction && ${ isMatchedIorFunc }( eta ) ) {

			textureStore( ${ outputTarget }, vec3( globalId.xy, layer ), vec4( 1.0 ) );
			return;

		}

		// sample the brdf directions in a grid pattern
		const GRID_SIZE = 64u;

		// TODO: this sampling means that energy at 0.0 & 1.0 roughness (and 0 and 90deg cos) are never
		// written to the texture due to the half texel inset, resulting in small, though possibly noticeable,
		// error in common cases.
		let dimensions = textureDimensions( ${ outputTarget } ).xy;
		let uv = ( vec2f( globalId.xy ) + vec2f( 0.5 ) ) / vec2f( dimensions );

		let cosThetaO = uv.x;
		let roughness = uv.y;
		let alpha = roughness * roughness;

		let wo = vec3( sqrt( 1 - cosThetaO * cosThetaO ), 0 , cosThetaO );

		let f0 = ${ iorToF0Func }( eta );

		var result = 0.0;
		for ( var x = 0u; x < GRID_SIZE; x++ ) {

			for ( var y = 0u; y < GRID_SIZE; y++ ) {

				// calculate the half vector to sample
				let gridPoint = vec2f( vec2u( x, y ) ) + vec2f( 0.5 );
				let sampleUv = gridPoint / f32( GRID_SIZE );
				let wh = ${ ggxDirectionFunc }( wo, vec2( alpha ), sampleUv );

				var f = 1.0;
				if ( includeFresnel ) {

					// use the exact dielectric fresnel so the table agrees with the fresnel used
					// for lobe sampling
					f = ${ dielectricFresnelFunc }( dot( wo, wh ), eta );

				}

				// air-incident reflection is evaluated with schlick in bsdfEval - only the
				// volume-incident ( TIR ) glass reflection uses the exact fresnel
				var reflectionF = f;
				if ( eta < 1.0 ) {

					reflectionF = ${ schlickFresnelFunc }( dot( wo, wh ), f0 );

				}

				// reflection lobe - samples landing below the surface are lost
				let wi = - reflect( wo, wh );
				let reflectionPdf = ${ ggxReflectionAdjustedPDFFunc }( wo, wh, vec2f( alpha ) );
				if ( wi.z > 0.0 && reflectionPdf != 0.0 ) {

					let specular = ${ specularBrdfFunc }( wo, wi, wh, vec2f( alpha ) );
					result += reflectionF * specular.x * wi.z / reflectionPdf;

				}

				// refraction lobe - the btdf carries its own ( 1 - F ) fresnel weight, and
				// samples landing back above the surface are lost
				if ( includeRefraction ) {

					let wiRefract = refract( - wo, wh, eta );
					if ( wiRefract.z < 0.0 ) {

						let transmission = ${ specularBtdfFunc }( wo, wiRefract, wh, vec2f( alpha ), eta );
						let pdf = ${ ggxRefractionAdjustedPDFFunc }( wo, wiRefract, wh, vec2f( alpha ), eta );
						if ( pdf != 0.0 ) {

							result += transmission.x * abs( wiRefract.z ) / pdf;

						}

					}

				}

			}

		}

		result /= f32( GRID_SIZE * GRID_SIZE );

		textureStore( ${ outputTarget }, vec3( globalId.xy, layer ), vec4( result ) );

	}

`;
