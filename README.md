# tgel-stock

Headless Blender pipeline for procedural rolling-stock (train) model
generation: generation recipes in, FBX + baked textures + semantic manifest
out.

The toolchain builds mid-poly railway vehicles (road-switcher locomotive,
40 ft box wagon, shared knuckle coupler) with baked surface detail — albedo,
normal and weathering-mask atlases — entirely from procedural Python
definitions. No external models, no pre-exported assets, no legacy
conversions.

## Requirements

- **Blender 5.1.2** (pinned; the smoke test fails on any other version)
  - default expected path: `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`
  - override with the `-BlenderExe` parameter or the `TGEL_BLENDER_EXE`
    environment variable
- **PowerShell 7+** (`pwsh`) for the `Run-Blender.ps1` wrapper
- **Python >= 3.11** for the pure-Python test suite (the generator itself
  runs inside Blender's bundled interpreter)

## Install (optional)

The toolchain runs fine straight from a clone. To import the `tgel_stock`
package from elsewhere:

```bash
pip install git+https://github.com/smadrom/tgel-stock.git
```

Note: `tgel_stock` targets Blender's bundled Python and its `bpy` module.
Under a system Python only the bpy-free modules import cleanly (the package
root, recipe/manifest/canonical helpers) — that is expected.

## Usage

Build one model (production build, 4096 atlas, Cycles bakes). The output
directory must not exist yet; it is published atomically on success:

```bash
blender --background --factory-startup --python-exit-code 1 \
  --python build.py -- --recipe recipes/basic-box-wagon.rollingstock.json --out out/wagon
```

or via the PowerShell wrapper:

```powershell
pwsh -File Run-Blender.ps1 -PythonFile build.py `
  -ScriptArgs "--recipe", "recipes/basic-box-wagon.rollingstock.json", "--out", "out/wagon"
```

Final output set per build (exactly these five files):

- `<model_id>.fbx`
- `albedo.png`, `normal.png`, `mask.png` (4096x4096)
- `<model_id>.manifest.json` — semantic manifest (geometry/texture hashes,
  recipe and script digests, Blender version)

## Recipes

- `recipes/basic-diesel-locomotive.rollingstock.json` — road-switcher locomotive
- `recipes/basic-box-wagon.rollingstock.json` — 40 ft box wagon

Recipe dimensions, seeds and liveries are frozen and validated at build
time; editing them fails the build by design.

## Tests

Pure-Python suite (no Blender required), stdlib unittest:

```bash
py -3.12 -m unittest discover -s tests/pure -v
```

Blender smoke gate (pinned version + path guards), expects `SMOKE OK`:

```bash
pwsh -File Run-Blender.ps1 -PythonFile tests/bpy/smoke_test.py
```

The full Blender test suite lives in `tests/bpy/`; run any of its files the
same way via `Run-Blender.ps1`.

## Clean-source rule

The generator accepts only these inputs:

- generation recipes (JSON) and the procedural Python source under `tgel_stock/`
- Blender's bundled bfont library

No external models, no pre-exported assets, no legacy conversions. Every
model is generated fresh from procedural definitions, with reproducible
topology and material assignments.

## Building the package

Run the PyPA `build` frontend from OUTSIDE the repository root: this repo's
own `build.py` (the Blender orchestrator) shadows the `build` package when
`python -m build` runs with the repository root as the working directory.

```bash
py -3.12 -m venv .venv-build
.venv-build/Scripts/python -m pip install build
cd ..
tgel-stock/.venv-build/Scripts/python -m build tgel-stock
```

Produces `dist/tgel_stock-1.0.0-py3-none-any.whl` and
`dist/tgel_stock-1.0.0.tar.gz` (in `tgel-stock/dist/`). The wheel
intentionally contains only the `tgel_stock` Python package; recipes, tests
and `build.py` live in this repository and run from a clone.

## Repository layout note

`build_probe.py` and the bake/materials suites under `tests/bpy/` write their
outputs to a relative `../../artifacts/` path — outside the clone when this
repository is checked out on its own. Pass explicit `--out` paths (or run
`build.py`, which always takes an explicit output directory) if you want
artifacts inside the clone.

## License

MIT — see `LICENSE`.
