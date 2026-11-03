import { wgslTagFn } from 'three-mesh-bvh/webgpu';

// Rust supplies algorithm source; this host adapter only binds existing Three
// nodes. No eval, device creation, resource allocation or CPU readback.
export function bindPipelineShader(source, bindings) {
    if (typeof source !== 'string' || source.length === 0) throw new Error('Missing rendering pipeline shader');
    const tokens = [], args = [];
    let offset = 0;
    for (const match of source.matchAll(/\{\{([a-zA-Z][a-zA-Z0-9]*)\}\}/g)) {
        if (!Object.hasOwn(bindings, match[1])) throw new Error(`Unknown rendering shader binding: ${match[1]}`);
        tokens.push(source.slice(offset, match.index)); args.push(bindings[match[1]]);
        offset = match.index + match[0].length;
    }
    tokens.push(source.slice(offset));
    if (tokens.some(token => token.includes('{{') || token.includes('}}'))) throw new Error('Invalid rendering shader binding');
    return wgslTagFn(tokens, ...args);
}
