# Moda internal fork

The internal main branch includes the actual upstream Git revision recorded in
upstream.json. Upgrade from a stable upstream release, migrate the local
changes by feature and verify Studio before issuing an internal tag.

npm-payload.json lists the reviewed package payload. Generated upstream files
used by Studio are kept with the source so tagging does not regenerate them
through upstream lifecycle hooks. A tag of the form
`v<upstream-version>-moda.<revision>` publishes one npm archive and a manifest
containing source revision and SHA-256. Studio pins the tagged artifact.
