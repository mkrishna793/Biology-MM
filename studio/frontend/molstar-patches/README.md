# Mol* Core Patches for Biology-MM Studio

This directory contains the surgical patches applied to core Mol* v6 modules to enable robust multi-million atom biological assembly loading and visualization:

1. **Assembly Symmetry Robustness (`packages/extensions/assembly-symmetry/`)**:
   - `src/representation.ts` & `src/transforms.ts`: Adds optional chaining and null safety checks for `AssemblySymmetryProvider.get(data)?.value` and `rotation_axes`. When assemblies lack explicit axes, empty meshes are returned instead of uncaught null pointer exceptions (`Cannot read properties of undefined (reading 'rotation_axes')`).

2. **Default Assembly Download (`packages/plugin/core/src/state/actions/structure.ts` & `root-structure.ts`)**:
   - Configures default structure download type to `'assembly'` (falling back to `'model'` if no symmetry operations exist). In standard Mol*, viral capsids and large multimeric complexes (such as Faustovirus 5J7V) would default to `'model'` (downloading only 1 asymmetric unit trimer of ~14k atoms rather than the full 40-million atom capsid).

3. **PDBe Volume Streaming (`packages/plugin/core/src/config.ts`)**:
   - Updates `VolumeStreaming.DefaultServer` to `https://www.ebi.ac.uk/pdbe/densities`.

## Applying the Patch

To apply these modifications to any clean Mol* v6 clone:
```bash
cd molstar
git apply /path/to/studio/frontend/molstar-patches/molstar-core.patch
```
