import { pipelineWgslFn } from 'three-mesh-bvh/webgpu';
import { wgslFn } from 'three/tsl';
import { constants, lightStruct, lightRecordStruct } from './structs.wgsl.js';

// Light type tags matching LightsInfoUniformStruct's packing. The environment is treated as an
// additional light kind so env + analytic lights share one NEE path.
export const RECT_AREA_LIGHT_TYPE = 0;
export const CIRC_AREA_LIGHT_TYPE = 1;
export const SPOT_LIGHT_TYPE = 2;
export const DIR_LIGHT_TYPE = 3;
export const POINT_LIGHT_TYPE = 4;
export const ENVIRONMENT_LIGHT_TYPE = 5;
export const LIGHT_FAR_DISTANCE = 1e30;

// fraction of the light distance a shadow ray stops short of the light
export const LIGHT_EPSILON = 1e-5;

// light kinds that are also bsdf-sampled and so take MIS-weighted NEE - punctual lights take full weight
export const isMISWeightLightFn = pipelineWgslFn(141, ["\n\n\tfn isMISWeightLight( lightType: i32 ) -> bool {\n\n\t\treturn lightType == "," || lightType == "," || lightType == ",";\n\n\t}\n\n"], [ENVIRONMENT_LIGHT_TYPE, CIRC_AREA_LIGHT_TYPE, RECT_AREA_LIGHT_TYPE], []);

export const getSpotAttenuationFn = pipelineWgslFn(142, ["\n\n\tfn getSpotAttenuation( coneCosine: f32, penumbraCosine: f32, angleCosine: f32 ) -> f32 {\n\n\t\treturn smoothstep( coneCosine, penumbraCosine, angleCosine );\n\n\t}\n\n"], [], []);

export const getDistanceAttenuationFn = pipelineWgslFn(143, ["\n\n\tfn getDistanceAttenuation( lightDistance: f32, cutoffDistance: f32, decayExponent: f32 ) -> f32 {\n\n\t\t// based upon Frostbite 3 Moving to Physically-based Rendering\n\t\t// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf\n\t\tvar distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), EPSILON );\n\t\tif ( cutoffDistance > 0.0 ) {\n\n\t\t\tlet window = clamp( 1.0 - pow( lightDistance / cutoffDistance, 4.0 ), 0.0, 1.0 );\n\t\t\tdistanceFalloff *= window * window;\n\n\t\t}\n\n\t\treturn distanceFalloff;\n\n\t}\n\n"], [], [ constants ]);

// Ray/plane intersection constrained to a rectangle centered at "center" spanned by u, v.
// Returns the hit distance along the ray, or a negative value when there is no hit.
export const intersectsRectangleFn = pipelineWgslFn(144, ["\n\n\tfn intersectsRectangle( center: vec3f, normal: vec3f, u: vec3f, v: vec3f, rayOrigin: vec3f, rayDirection: vec3f ) -> f32 {\n\n\t\tlet t = dot( center - rayOrigin, normal ) / dot( rayDirection, normal );\n\t\tif ( t > EPSILON ) {\n\n\t\t\tlet p = rayOrigin + rayDirection * t;\n\t\t\tlet vi = p - center;\n\n\t\t\tlet a1 = dot( u, vi );\n\t\t\tif ( abs( a1 ) <= 0.5 ) {\n\n\t\t\t\tlet a2 = dot( v, vi );\n\t\t\t\tif ( abs( a2 ) <= 0.5 ) {\n\n\t\t\t\t\treturn t;\n\n\t\t\t\t}\n\n\t\t\t}\n\n\t\t}\n\n\t\treturn - 1.0;\n\n\t}\n\n"], [], [ constants ]);

// Ray/plane intersection constrained to a circle centered at "position" spanned by u, v.
// Returns the hit distance along the ray, or a negative value when there is no hit.
export const intersectsCircleFn = pipelineWgslFn(145, ["\n\n\tfn intersectsCircle( position: vec3f, normal: vec3f, u: vec3f, v: vec3f, rayOrigin: vec3f, rayDirection: vec3f ) -> f32 {\n\n\t\tlet t = dot( position - rayOrigin, normal ) / dot( rayDirection, normal );\n\t\tif ( t > EPSILON ) {\n\n\t\t\tlet hit = rayOrigin + rayDirection * t;\n\t\t\tlet vi = hit - position;\n\n\t\t\tlet a1 = dot( u, vi );\n\t\t\tlet a2 = dot( v, vi );\n\t\t\tif ( length( vec2f( a1, a2 ) ) <= 0.5 ) {\n\n\t\t\t\treturn t;\n\n\t\t\t}\n\n\t\t}\n\n\t\treturn - 1.0;\n\n\t}\n\n"], [], [ constants ]);

// Samples a random point on a rectangular or circular area light and forms the LightRecord.
export const randomAreaLightSampleFn = pipelineWgslFn(146, ["\n\n\tfn randomAreaLightSample( light: Light, rayOrigin: vec3f, ruv: vec2f ) -> LightRecord {\n\n\t\tvar randomPos = vec3f( 0.0 );\n\t\tif ( light.lightType == "," ) {\n\n\t\t\trandomPos = light.position + light.u * ( ruv.x - 0.5 ) + light.v * ( ruv.y - 0.5 );\n\n\t\t} else if ( light.lightType == "," ) {\n\n\t\t\tlet r = 0.5 * sqrt( ruv.x );\n\t\t\tlet theta = ruv.y * 2.0 * PI;\n\t\t\trandomPos = light.position + light.u * ( r * cos( theta ) ) + light.v * ( r * sin( theta ) );\n\n\t\t}\n\n\t\tlet toLight = randomPos - rayOrigin;\n\t\tlet lightDistSq = dot( toLight, toLight );\n\t\tlet dist = sqrt( lightDistSq );\n\t\tlet direction = toLight / dist;\n\t\tlet lightNormal = normalize( cross( light.u, light.v ) );\n\n\t\tvar lightRec: LightRecord;\n\t\tlightRec.lightType = light.lightType;\n\t\tlightRec.emission = light.color * light.intensity;\n\t\tlightRec.dist = dist;\n\t\tlightRec.direction = direction;\n\n\t\t// points behind or edge-on to the light receive no contribution\n\t\tlet cosTheta = dot( direction, lightNormal );\n\t\tif ( cosTheta <= 0.0 ) {\n\n\t\t\tlightRec.pdf = 0.0;\n\n\t\t} else {\n\n\t\t\tlightRec.pdf = lightDistSq / ( light.area * cosTheta );\n\n\t\t}\n\n\t\treturn lightRec;\n\n\t}\n\n"], [RECT_AREA_LIGHT_TYPE, CIRC_AREA_LIGHT_TYPE], [ lightStruct, lightRecordStruct, constants ]);

// Samples the disc of a spot light with distance falloff. Angular ( cone or IES ) attenuation is applied by the caller.
export const randomSpotLightSampleFn = pipelineWgslFn(147, ["\n\n\tfn randomSpotLightSample( light: Light, rayOrigin: vec3f, ruv: vec2f ) -> LightRecord {\n\n\t\tlet radius = light.radius * sqrt( ruv.x );\n\t\tlet theta = ruv.y * 2.0 * PI;\n\t\tlet x = radius * cos( theta );\n\t\tlet y = radius * sin( theta );\n\n\t\tlet u = light.u;\n\t\tlet v = light.v;\n\t\tlet normal = normalize( cross( u, v ) );\n\n\t\tlet angle = acos( light.coneCos );\n\t\tlet angleTan = tan( angle );\n\t\tlet startDistance = light.radius / max( angleTan, EPSILON );\n\n\t\tlet randomPos = light.position - normal * startDistance + u * x + v * y;\n\t\tlet toLight = randomPos - rayOrigin;\n\t\tlet lightDistSq = dot( toLight, toLight );\n\t\tlet dist = sqrt( lightDistSq );\n\n\t\tlet direction = toLight / max( dist, EPSILON );\n\t\tlet distanceAttenuation = getDistanceAttenuation( dist, light.distance, light.decay );\n\n\t\tvar lightRec: LightRecord;\n\t\tlightRec.lightType = light.lightType;\n\t\tlightRec.dist = dist;\n\t\tlightRec.direction = direction;\n\t\tlightRec.emission = light.color * light.intensity * distanceAttenuation;\n\t\tlightRec.pdf = 1.0;\n\n\t\treturn lightRec;\n\n\t}\n\n"], [], [ lightStruct, lightRecordStruct, constants, getDistanceAttenuationFn ]);

// The cosine to the spot axis and the distance from a position to the nearest point on the spot
// light's disc, where the disc lights the position the most
export const getSpotLightNearestFn = pipelineWgslFn(148, ["\n\n\tfn getSpotLightNearest( light: Light, position: vec3f ) -> vec2f {\n\n\t\tlet normal = normalize( cross( light.u, light.v ) );\n\t\tlet startDistance = light.radius / max( tan( acos( light.coneCos ) ), EPSILON );\n\t\tlet discCenter = light.position - normal * startDistance;\n\n\t\tlet toPosition = position - discCenter;\n\t\tlet along = dot( toPosition, - normal );\n\t\tlet across = max( length( toPosition + normal * along ) - light.radius, 0.0 );\n\t\tlet dist = length( vec2f( along, across ) );\n\t\treturn vec2f( along / max( dist, EPSILON ), dist );\n\n\t}\n\n"], [], [ lightStruct, constants ]);
