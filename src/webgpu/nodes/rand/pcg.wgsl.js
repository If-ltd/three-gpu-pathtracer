import { pipelineWgsl, pipelineWgslFn } from 'three-mesh-bvh/webgpu';
import { wgsl, wgslFn } from 'three/tsl';

const pcgStateStruct = pipelineWgsl(175, ["\n\n\tstruct PcgState {\n\t\ts0: vec4u,\n\t\ts1: vec4u,\n\t};\n\n\tvar<private> g_state: PcgState;\n"], [], []);

const pcgInitFunc = pipelineWgslFn(176, ["\n\tfn pcgInitialize( pixel: vec2u, pathIndex: u32, bounceIndex: u32 ) -> void {\n\n\t\t// white noise seed\n\t\tg_state.s0 = vec4u( pixel | vec2( bounceIndex << 16 ), pathIndex, pixel.x + pixel.y);\n\n\t\t// blue noise seed\n\t\tg_state.s1 = vec4u( pathIndex, pathIndex * 15843, pathIndex * 31 + 4566, pathIndex * 2345 + 58585 );\n\n\t}\n"], [], [ pcgStateStruct ]);

const pcg4d = pipelineWgslFn(177, ["\n\tfn pcg4d( v: ptr<private, vec4u> ) -> void {\n\n\t\t*v = *v * 1664525u + 1013904223u;\n\t\t*v = *v + v.yzxy * v.wxyz;\n\t\t*v = *v ^ ( *v >> vec4u( 16u ) );\n\t\t*v = *v + v.yzxy * v.wxyz;\n\t}\n"], [], []);

const pcgRandFunc = pipelineWgslFn(178, ["\n\tfn pcgRand( _id: u32 ) -> f32 {\n\n\t\tpcg4d( &g_state.s0 );\n\t\treturn abs( f32( g_state.s0.x ) / f32( 0xffffffffu ) );\n\n\t}\n"], [], [ pcg4d, pcgStateStruct ]);

const pcgRand2Func = pipelineWgslFn(179, ["\n\tfn pcgRand2( _id: u32 ) -> vec2f {\n\n\t\tpcg4d( &g_state.s0 );\n\t\treturn abs( vec2f( g_state.s0.xy ) / f32( 0xffffffffu ) );\n\n\t}\n"], [], [ pcg4d, pcgStateStruct ]);

const pcgRand3Func = pipelineWgslFn(180, ["\n\tfn pcgRand3( _id: u32 ) -> vec3f {\n\n\t\tpcg4d( &g_state.s0 );\n\t\treturn abs( vec3f( g_state.s0.xyz ) / f32( 0xffffffffu ) );\n\n\t}\n"], [], [ pcg4d, pcgStateStruct ]);

const pcgRand4Func = pipelineWgslFn(181, ["\n\tfn pcgRand4( _id: u32 ) -> vec4f {\n\n\t\tpcg4d( &g_state.s0 );\n\t\treturn abs( vec4f( g_state.s0 ) / f32( 0xffffffffu ) );\n\n\t}\n"], [], [ pcg4d, pcgStateStruct ]);

const pcgNextBounceFunc = pipelineWgslFn(182, ["fn noop() -> void {}"], [], []);

// PCG definitions
export {
	pcgInitFunc as rngInit,
	pcgNextBounceFunc as rngNextBounce,
	pcgRandFunc as rand1,
	pcgRand2Func as rand2,
	pcgRand3Func as rand3,
	pcgRand4Func as rand4,
};
