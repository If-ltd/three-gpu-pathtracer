import { pipelineWgslFn } from 'three-mesh-bvh/webgpu';
import { wgslFn } from 'three/tsl';
import { constants, lobeWeightsStruct } from './structs.wgsl.js';
import { disneyFresnelFunc, iorToF0Func, schlickFresnelFunc } from './utils.wgsl.js';

/*
wi     : incident vector or light vector (pointing toward the light)
wo     : outgoing vector or view vector (pointing towards the camera)
wh     : computed half vector from wo and wi
Vectors above are assumed to be in tangent space. i.e. +z is along macronormal of the surface
eta    : Greek character used to denote the "ratio of ior"
*/

export const sampleSphereFunc = pipelineWgslFn(196, ["\n\n\tfn sampleSphere( uv: vec2f ) -> vec3f {\n\n\t\tlet u = ( uv.x - 0.5 ) * 2.0;\n\t\tlet t = uv.y * PI * 2.0;\n\t\tlet f = sqrt( 1.0 - u * u );\n\n\t\treturn vec3f( f * cos( t ), f * sin( t ), u );\n\n\t}\n\n"], [], [ constants ]);

// TODO: Investigate sampling directly in tagent space?
// See 16.6.1 in https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf
export const diffuseDirectionFunc = pipelineWgslFn(197, ["\n\n\tfn diffuseDirection( wo: vec3f, uv: vec2f ) -> vec3f {\n\n\t\tvar lightDirection = sampleSphere( uv );\n\t\tlightDirection.z += 1.0;\n\t\tlightDirection = normalize( lightDirection );\n\n\t\treturn lightDirection;\n\n\t}\n\n"], [], [ sampleSphereFunc ]);

export const getLobeWeightsFunc = pipelineWgslFn(198, ["\n\n\tfn getLobeWeights( wo: vec3f, wi: vec3f, woClearcoat: vec3f, wh: vec3f, clearcoatIor: f32, surf: SurfaceRecord ) -> LobeWeights {\n\n\t\tvar weights: LobeWeights;\n\t\tvar remaining = 1.0;\n\n\t\tlet clearcoatF0 = iorToF0( clearcoatIor );\n\t\tlet clearcoatFresnel = schlickFresnel( saturate( woClearcoat.z ), clearcoatF0 );\n\t\tweights.clearcoat = surf.clearcoat * clearcoatFresnel;\n\t\tremaining -= weights.clearcoat;\n\n\t\t// this specular lobe only handles the opaque portion of the specular\n\t\tlet fEstimate = disneyFresnel( wo, wi, wh, surf.f0, surf.eta, surf.metalness );\n\t\tweights.specular = remaining * ( surf.metalness + ( 1.0 - surf.metalness ) * ( 1.0 - surf.transmission ) * fEstimate );\n\t\tremaining -= weights.specular;\n\n\t\t// the transmission lobe covers both the reflected and refracted halves of the transmissive portion\n\t\tweights.transmission = remaining * surf.transmission;\n\t\tremaining -= weights.transmission;\n\n\t\tweights.diffuse = remaining;\n\n\t\treturn weights;\n\n\t}\n\n"], [], [ disneyFresnelFunc, schlickFresnelFunc, iorToF0Func, lobeWeightsStruct, constants ]);

export const equirectDirectionToUvFn = pipelineWgslFn(199, ["\n\n\tfn equirectDirectionToUv(direction: vec3f) -> vec2f {\n\n\t\t// from Spherical.setFromCartesianCoords\n\t\tvar uv = vec2f( atan2( direction.z, direction.x ), acos( direction.y ) );\n\t\tuv /= vec2f( 2.0 * PI, PI );\n\n\t\t// apply adjustments to get values in range [0, 1] and y right side up\n\t\tuv.x += 0.5;\n\t\tuv.y = 1.0 - uv.y;\n\t\treturn uv;\n\n\t}\n\n"], [], []);

export const sampleHemisphereFn = pipelineWgslFn(200, ["\n\n\tfn sampleHemisphere( n: vec3f, uv: vec2f ) -> vec3f {\n\n\t\t// https://www.rorydriscoll.com/2009/01/07/better-sampling/\n\t\t// https://graphics.pixar.com/library/OrthonormalB/paper.pdf\n\t\tlet sign = select( sign( n.z ), 1.0, n.z == 0.0 );\n\t\tlet a = - 1.0 / ( sign + n.z );\n\t\tlet b = n.x * n.y * a;\n\t\tlet b1 = vec3( 1.0 + sign * n.x * n.x * a, sign * b, - sign * n.x );\n\t\tlet b2 = vec3( b, sign + n.y * n.y * a, - n.y );\n\n\t\tlet r = sqrt( uv.x );\n\t\tlet theta = 2.0 * PI * uv.y;\n\t\tlet x = r * cos( theta );\n\t\tlet y = r * sin( theta );\n\t\treturn x * b1 + y * b2 + sqrt( 1.0 - uv.x ) * n;\n\n\t}\n\n"], [], [ constants ]);

// power heuristic for multiple importance sampling
export const misHeuristicFn = pipelineWgslFn(201, ["\n\n\tfn misHeuristic( a: f32, b: f32 ) -> f32 {\n\n\t\tlet aa = a * a;\n\t\tlet bb = b * b;\n\t\treturn aa / ( aa + bb );\n\n\t}\n\n"], [], []);

export const luminanceFn = pipelineWgslFn(202, ["\n\n\tfn luminance( color: vec3f ) -> f32 {\n\n\t\treturn dot( color, vec3f( 0.2126, 0.7152, 0.0722 ) );\n\n\t}\n\n"], [], []);

// inverse of equirectDirectionToUv: map an equirect uv back to a direction
export const equirectUvToDirectionFn = pipelineWgslFn(203, ["\n\n\tfn equirectUvToDirection( uvIn: vec2f ) -> vec3f {\n\n\t\t// undo the adjustments applied in equirectDirectionToUv\n\t\tvar uv = uvIn;\n\t\tuv.x -= 0.5;\n\t\tuv.y = 1.0 - uv.y;\n\n\t\tlet theta = uv.x * 2.0 * PI;\n\t\tlet phi = uv.y * PI;\n\t\tlet sinPhi = sin( phi );\n\n\t\treturn vec3f( sinPhi * cos( theta ), cos( phi ), sinPhi * sin( theta ) );\n\n\t}\n\n"], [], [ constants ]);

// solid-angle pdf factor for the equirect parameterization ( accounts for pole compression )
export const equirectDirectionPdfFn = pipelineWgslFn(204, ["\n\n\tfn equirectDirectionPdf( direction: vec3f ) -> f32 {\n\n\t\tlet uv = equirectDirectionToUv( direction );\n\t\tlet theta = uv.y * PI;\n\t\tlet sinTheta = sin( theta );\n\t\tif ( sinTheta == 0.0 ) {\n\n\t\t\treturn 0.0;\n\n\t\t}\n\n\t\treturn 1.0 / ( 2.0 * PI * PI * sinTheta );\n\n\t}\n\n"], [], [ constants, equirectDirectionToUvFn ]);

export const weightedAlphaBlendFn = pipelineWgslFn(205, ["\n\n\tfn weightedAlphaBlend( prevColor: vec4f, newColor: vec4f, weight: f32 ) -> vec4f {\n\n\t\tlet invWeight = 1.0 - weight;\n\t\tlet totalAlpha = prevColor.a * invWeight + newColor.a * weight;\n\t\tvar blendedColor = vec4f( 0 );\n\t\tif ( totalAlpha != 0.0 ) {\n\n\t\t\tlet prevContrib = prevColor.rgb * invWeight * prevColor.a / totalAlpha;\n\t\t\tlet resContrib = newColor.rgb * weight * newColor.a / totalAlpha;\n\t\t\tblendedColor = vec4f( prevContrib + resContrib, totalAlpha );\n\n\t\t}\n\n\t\treturn blendedColor;\n\n\t}\n\n"], [], []);
