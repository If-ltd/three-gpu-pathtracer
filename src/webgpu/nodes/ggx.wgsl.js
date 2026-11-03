import { pipelineWgslFn } from 'three-mesh-bvh/webgpu';
import { wgslFn } from 'three/tsl';
import { constants } from './structs.wgsl.js';

// See sampling.wgsl for vector shorthand explanations
// The GGX functions provide sampling and distribution information for normals as output so
// in order to get probability of scatter direction the half vector must be computed and
// provided. Anisotropic surfaces are represented with a 2-dimensional alpha storing the
// roughness along the tangent (x) and bitangent (y) in the TBN frame.

// [0] https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
// [1] https://hal.archives-ouvertes.fr/hal-01509746/document
// [2] http://jcgt.org/published/0007/04/01/
// [4] http://jcgt.org/published/0003/02/03/
// [5] https://seblagarde.wordpress.com/wp-content/uploads/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
// [6] https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_anisotropy/README.md
// [7] https://google.github.io/filament/Filament.md.html#materialsystem/anisotropicmodel/anisotropicspecularbrdf

// trowbridge-reitz === GGX === GTR
export const ggxDirectionFunc = pipelineWgslFn(134, ["\n\n\tfn ggxDirection( incidentDir: vec3f, alpha: vec2f, uv: vec2f ) -> vec3f {\n\n\t\t// Implementation from reference [1]\n\t\t// stretch view\n\t\tlet V = normalize( vec3f( alpha * incidentDir.xy, incidentDir.z ) );\n\n\t\t// orthonormal basis\n\t\tvar T1: vec3f;\n\t\tif ( V.z < 0.9999 ) {\n\n\t\t\tT1 = normalize( cross( V, vec3( 0.0, 0.0, 1.0 ) ) );\n\n\t\t} else {\n\n\t\t\tT1 = vec3( 1.0, 0.0, 0.0 );\n\n\t\t}\n\n\t\tlet T2 = cross( T1, V );\n\n\t\t// sample point with polar coordinates (r, phi)\n\t\tlet a = 1.0 / ( 1.0 + V.z );\n\t\tlet r = sqrt( uv.x );\n\t\tvar phi: f32;\n\t\tif ( uv.y < a ) {\n\n\t\t\tphi = uv.y / a * PI;\n\n\t\t} else {\n\n\t\t\tphi = PI + ( uv.y - a ) / ( 1.0 - a ) * PI;\n\n\t\t}\n\n\t\tlet P1 = r * cos( phi );\n\t\tvar P2 = r * sin( phi );\n\t\tif ( uv.y >= a ) {\n\n\t\t\tP2 *= V.z;\n\n\t\t}\n\n\t\t// compute normal\n\t\tvar N = P1 * T1 + P2 * T2 + V * sqrt( max( 0.0, 1.0 - P1 * P1 - P2 * P2 ) );\n\n\t\t// unstretch\n\t\tN = normalize( vec3( alpha * N.xy, max( 0.0, N.z ) ) );\n\n\t\treturn N;\n\n\t}\n\n"], [], [ constants ]);

// Smith masking lambda for a single direction, used to build the G1 term for the pdf
// See equation (34) from [0] and equation (43) from [7]
export const ggxLambdaFunc = pipelineWgslFn(135, ["\n\n\tfn ggxLambda( V: vec3f, alpha: vec2f ) -> f32 {\n\n\t\tlet alphaT = alpha.x;\n\t\tlet alphaB = alpha.y;\n\n\t\t// abs v.z to support transmitted rays\n\t\tlet NdotV = max( abs( V.z ), MIN_INCIDENT_COS );\n\t\tlet cos2 = NdotV * NdotV;\n\t\tlet t = ( alphaT * alphaT * V.x * V.x + alphaB * alphaB * V.y * V.y ) / cos2;\n\t\tlet numerator = - 1.0 + sqrt( 1.0 + t );\n\t\treturn numerator / 2.0;\n\n\t}\n\n"], [], [ constants ]);

// Based on equation (2) from reference [1]
export const ggxShadowMaskG1Func = pipelineWgslFn(136, ["\n\n\tfn ggxShadowMaskG1( V: vec3f, alpha: vec2f ) -> f32 {\n\n\t\t// TODO: this could be collapsed to a simpler form as an optimization\n\t\treturn 1.0 / ( 1.0 + ggxLambda( V, alpha ) );\n\n\t}\n\n"], [], [ ggxLambdaFunc ]);

// Smith height-correlated visibility term = G / ( 4 * NdotV * NdotL )
// See (listing 16) in [7] and from [6]
export const ggxSmithVisibilityFunc = pipelineWgslFn(137, ["\n\n\tfn ggxSmithVisibility( V: vec3f, L: vec3f, alpha: vec2f ) -> f32 {\n\n\t\tlet alphaT = alpha.x;\n\t\tlet alphaB = alpha.y;\n\n\t\tlet NdotV = max( V.z, MIN_INCIDENT_COS );\n\t\tlet NdotL = max( L.z, MIN_INCIDENT_COS );\n\n\t\tlet TdotV = V.x;\n\t\tlet TdotL = L.x;\n\n\t\tlet BdotV = V.y;\n\t\tlet BdotL = L.y;\n\n\t\tlet GGXV = NdotL * length( vec3f( alphaT * TdotV, alphaB * BdotV, NdotV ) );\n\t\tlet GGXL = NdotV * length( vec3f( alphaT * TdotL, alphaB * BdotL, NdotL ) );\n\n\t\treturn 0.5 / ( GGXV + GGXL );\n\n\t}\n\n"], [], [ constants ]);

// Trowbridge-Reitz ( GGX ) normal distribution
// See (listing 15) in [7]
export const ggxDistributionFunc = pipelineWgslFn(138, ["\n\n\tfn ggxDistribution( H: vec3f, alpha: vec2f ) -> f32 {\n\n\t\tlet alphaT = alpha.x;\n\t\tlet alphaB = alpha.y;\n\n\t\tlet NdotH = H.z;\n\t\tlet TdotH = H.x;\n\t\tlet BdotH = H.y;\n\n\t\tlet a2 = alphaT * alphaB;\n\t\tlet v = vec3f( alphaB * TdotH, alphaT * BdotH, a2 * NdotH );\n\t\tlet v2 = dot( v, v );\n\t\tlet w2 = a2 / v2;\n\n\t\treturn a2 * w2 * w2 / PI;\n\n\t}\n\n"], [], [ constants ]);

// ggxPDF, divided by the Jacobian of reflection operation
// PDF: See equation (17) from reference [2]
// Note: HdotV cancel out bc its guaranteed to be > 0
export const ggxReflectionAdjustedPDFFunc = pipelineWgslFn(139, ["\n\n\tfn ggxReflectionAdjustedPDF( V: vec3f, H: vec3f, alpha: vec2f ) -> f32 {\n\n\t\tlet NdotV = max( V.z, MIN_INCIDENT_COS );\n\t\tlet D = ggxDistribution( H, alpha );\n\t\tlet G1 = ggxShadowMaskG1( V, alpha );\n\n\t\treturn D * G1 / ( 4 * NdotV );\n\n\t}\n"], [], [ ggxDistributionFunc, ggxShadowMaskG1Func ]);

// ggxPDF, divided by the Jacobian of refraction operation
// See equation (3) from [2] for pdf and (17) from [0] for Jacobian
export const ggxRefractionAdjustedPDFFunc = pipelineWgslFn(140, ["\n\n\tfn ggxRefractionAdjustedPDF( V: vec3f, L: vec3f, H: vec3f, alpha: vec2f, eta: f32 ) -> f32 {\n\n\t\tlet NdotV = max( V.z, MIN_INCIDENT_COS );\n\t\tlet HdotV = dot( V, H );\n\t\tlet HdotL = dot( L, H );\n\t\tlet D = ggxDistribution( H, alpha );\n\t\tlet G1 = ggxShadowMaskG1( V, alpha );\n\n\t\tlet denom = eta * HdotV + HdotL;\n\t\treturn D * G1 * abs( HdotV ) * abs( HdotL ) / ( NdotV * denom * denom );\n\n\t}\n"], [], [ ggxDistributionFunc, ggxShadowMaskG1Func ]);
