import { pipelineWgslFn, pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { MeshBasicNodeMaterial, NoToneMapping, StorageTexture } from 'three/webgpu';
import { uv, varying, texture, vec4, toneMapping, uniform, wgslFn } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';

// TODO: we could fall back to hardware-based filtering if available but it has to be specifically
// requested and available on the renderer which we don't have access to immediately. It's possible this
// can be detected during build time via a custom node to adjust the sample approach used?
const sampleTexelFn = pipelineWgslFn(100, ["\n\tfn sampleTexel( tex: texture_2d<f32>, coord: vec2f ) -> vec4f {\n\n\t\t// Manual bilinear filtering using textureLoad to support filterable float32 textures\n\t\t// on all devices\n\t\tlet size = vec2f( textureDimensions( tex, 0 ) );\n\n\t\t// the path tracer stores its color rows top down\n\t\tlet pxCoord = vec2f( coord.x, 1.0 - coord.y ) * size - 0.5;\n\t\tlet px = vec2i( floor( pxCoord ) );\n\t\tlet fr = fract( pxCoord );\n\n\t\t// get the four sibling samples\n\t\tlet s00 = textureLoad( tex, clamp( px,                 vec2i( 0 ), vec2i( size ) - 1 ), 0 );\n\t\tlet s10 = textureLoad( tex, clamp( px + vec2i( 1, 0 ), vec2i( 0 ), vec2i( size ) - 1 ), 0 );\n\t\tlet s01 = textureLoad( tex, clamp( px + vec2i( 0, 1 ), vec2i( 0 ), vec2i( size ) - 1 ), 0 );\n\t\tlet s11 = textureLoad( tex, clamp( px + vec2i( 1, 1 ), vec2i( 0 ), vec2i( size ) - 1 ), 0 );\n\n\t\t// interpolate on the x axis\n\t\tlet y0 = mix( s00, s10, fr.x );\n\t\tlet y1 = mix( s01, s11, fr.x );\n\n\t\t// then y\n\t\treturn mix( y0, y1, fr.y );\n\n\t}\n"], [], []);

// Material to apply tone mapping _before_ applying alpha blending
export class RenderToScreenNodeMaterial extends MeshBasicNodeMaterial {

	get texture() {

		return this._texNode.value;

	}

	set texture( v ) {

		this._texNode.value = v;

	}

	get fromTexture() {

		return this._fromTexNode.value;

	}

	set fromTexture( v ) {

		this._fromTexNode.value = v;

	}

	get toneMapping() {

		return this._toneMapping.toneMapping;

	}

	set toneMapping( v ) {

		this._toneMapping.setToneMapping( v );

	}

	get transition() {

		return this._transitionUniform.value;

	}

	set transition( v ) {

		this._transitionUniform.value = v;

	}

	get exposure() {

		return this._toneMapping.exposureNode.value;

	}

	set exposure( v ) {

		this._toneMapping.exposureNode.value = v;

	}

	constructor( params ) {

		super();

		const texNode = texture( new StorageTexture() );
		this._texNode = texNode;

		const fromTexNode = texture( new StorageTexture() );
		this._fromTexNode = fromTexNode;

		const texUV = varying( uv() );

		const transitionUniform = uniform( 1.0 );
		this._transitionUniform = transitionUniform;

		// NOTE: varyings cannot be referenced directly and must be passed as arguments
		const getFadedColorFn = pipelineWgslTagFn(101)`
			fn fade( uv: vec2f ) -> vec4f {

				if ( ${ transitionUniform } <= 0.0 ) {

					return ${ sampleTexelFn }( ${ fromTexNode }, uv );

				} else if ( ${ transitionUniform } >= 1.0 ) {

					return ${ sampleTexelFn }( ${ texNode }, uv );

				} else {

					let col0 = ${ sampleTexelFn }( ${ fromTexNode }, uv );
					let col1 = ${ sampleTexelFn }( ${ texNode }, uv );
					return mix( col0, col1, ${ transitionUniform } );

				}

			}
		`;

		const toneMappingNode = toneMapping( NoToneMapping, 1.0, getFadedColorFn( texUV ) );
		this._toneMapping = toneMappingNode;

		// apply alpha _after_ applying tone mapping
		// NOTE: alpha is being multiplied twice here to accommodate some odd blending in three.js
		// See mrdoob/three.js#33104. It's possible this should be removed or rethought once fixed.
		this.transparent = true;
		this.colorNode = vec4( toneMappingNode.rgb, toneMappingNode.a );

		this.setValues( params );

	}

}
