import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';

const upstream = JSON.parse(readFileSync('upstream.json', 'utf8'));
const tag = process.env.GITHUB_REF_NAME;
const sourceCommit = process.env.GITHUB_SHA;
const prefix = `v${upstream.version}-moda.`;
if (!tag?.startsWith(prefix) || !/^\d+$/.test(tag.slice(prefix.length))
    || !/^[0-9a-f]{40}$/.test(sourceCommit ?? '')) {
    throw new Error('Expected an internal release tag for this upstream version');
}
const revision = tag.slice(prefix.length);
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const version = `${upstream.version}+moda.${revision}`;
const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
if (pkg.name !== upstream.name || pkg.version !== version) throw new Error('Package metadata does not match the release tag');
const file = `${upstream.name}-${tag.slice(1)}.tgz`;
const files = JSON.parse(readFileSync('npm-payload.json', 'utf8'));
const bytes = gzipSync(execFileSync('git', ['archive', '--format=tar', '--prefix=package/', 'HEAD', '--', ...files], {
    maxBuffer: 32 * 1024 * 1024,
}), { level: 9 });
mkdirSync('release-assets', { recursive: true });
writeFileSync(`release-assets/${file}`, bytes);
writeFileSync('release-assets/release-manifest.json', JSON.stringify({
    repository: process.env.GITHUB_REPOSITORY, tag, sourceCommit,
    name: upstream.name, version, file, sha256: sha256(bytes),
    upstreamVersion: upstream.version, upstreamCommit: upstream.gitHead,
    upstreamIntegrity: upstream.dist?.integrity,
}, null, 2) + '\n');
