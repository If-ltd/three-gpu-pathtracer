import { pipelineWgslFn } from 'three-mesh-bvh/webgpu';
import { MeshBasicNodeMaterial, DataTexture, NoBlending } from 'three/webgpu';
import { uv, varying, texture, uniform, wgslFn } from 'three/tsl';

export class AtlasDebugMaterial extends MeshBasicNodeMaterial {

	set texture( v ) {

		this._texNode.value = v;

	}

	set layer( v ) {

		this._layerUniform.value = v;

	}

	constructor() {

		super();

		this.blending = NoBlending;

		this._texNode = texture( new DataTexture() );
		this._layerUniform = uniform( 0 );

		// Shows one layer of the atlas array texture.
		this.colorNode = pipelineWgslFn(102, ["\n\t\t\tfn atlasDebug( tex: texture_2d_array<f32>, coord: vec2f, layer: f32 ) -> vec4f {\n\n\t\t\t\t// flip y so texel origin shows at the top-left\n\t\t\t\tlet dims = vec2f( textureDimensions( tex, 0 ).xy );\n\t\t\t\tlet texel = vec2i( vec2f( coord.x, 1.0 - coord.y ) * dims );\n\t\t\t\treturn textureLoad( tex, texel, i32( layer ), 0 );\n\n\t\t\t}\n\t\t"], [], [])( this._texNode, varying( uv() ), this._layerUniform );

	}

}
