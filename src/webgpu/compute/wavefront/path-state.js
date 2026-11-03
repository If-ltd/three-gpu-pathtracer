import { pipelineWgslTagFn } from 'three-mesh-bvh/webgpu';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rayDataStruct } from './structs.js';

// Each scalar component occupies a contiguous plane of path slots. Integer and
// floating point bits are preserved exactly; no precision reduction is involved.
const typeMap = { uint: 'u32', int: 'i32', float: 'f32' };
let wordOffset = 0;
export const PATH_STATE_FIELDS = rayDataStruct.membersLayout.map(({ name, type }) => {
    const components = type.startsWith('vec') ? Number(type[3]) : 1;
    const field = { name, type: typeMap[type] || type, components, offset: wordOffset };
    wordOffset += components;
    return field;
});
export const PATH_STATE_WORDS = wordOffset;

export function createPathStateAccessors(buffer, soa) {
    const capacity = soa
        ? wgslTagFn([`fn pathCapacity() -> u32 { return arrayLength(&`, `) / ${PATH_STATE_WORDS}u; }`], buffer)
        : pipelineWgslTagFn(87)`fn pathCapacity() -> u32 { return arrayLength(&${buffer}); }`;
    const write = {};
    const read = {};
    for (const field of PATH_STATE_FIELDS) {
        const { name, type, components, offset } = field;
        if (!soa) {
            read[name] = wgslTagFn([`fn read_${name}(index: u32) -> ${type} { return `, `[index].${name}; }`], buffer);
            write[name] = wgslTagFn([`fn write_${name}(index: u32, value: ${type}) -> void { `, `[index].${name} = value; }`], buffer);
            continue;
        }
        const componentList = Array.from({length: components}, (_, i) => i);
        const scalarType = type.endsWith('u') || type === 'u32' ? 'u32' : type === 'i32' ? 'i32' : 'f32';
        const values = componentList.map(i => {
            const bits = `data[${offset + i}u * stride + index]`;
            return scalarType === 'u32' ? bits : `bitcast<${scalarType}>(${bits})`;
        });
        const readExpression = components === 1 ? values[0] : `${type}(${values.join(', ')})`;
        read[name] = wgslTagFn([`fn read_${name}(index: u32) -> ${type} { let data = &`, `;
            let stride = arrayLength(data) / ${PATH_STATE_WORDS}u; return ${readExpression}; }`], buffer);
        const assignments = componentList.map(i => {
            const value = components === 1 ? 'value' : `value[${i}]`;
            return `data[${offset + i}u * stride + index] = ${scalarType === 'u32' ? value : `bitcast<u32>(${value})`};`;
        }).join('\n');
        write[name] = wgslTagFn([`fn write_${name}(index: u32, value: ${type}) -> void { let data = &`, `;
            let stride = arrayLength(data) / ${PATH_STATE_WORDS}u; ${assignments} }`], buffer);
    }
    // Struct values are local only; compilers can discard fields unused by each
    // consumer while all persistent storage remains component-contiguous.
    const tokens = ['fn readPath(index: u32) -> ', ' { var result: ', ';\n'];
    const args = [rayDataStruct, rayDataStruct];
    for (const field of PATH_STATE_FIELDS) {
        tokens[tokens.length - 1] += `result.${field.name} = `;
        args.push(read[field.name]); tokens.push('(index);\n');
    }
    tokens[tokens.length - 1] += 'return result; }';
    return { capacity, read, write, load: wgslTagFn(tokens, ...args) };
}
