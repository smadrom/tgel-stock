"""Headless Blender orchestration for the generated TGEL V2 fleet.

Public CLI:

    blender --background --factory-startup --python-exit-code 1 \
      --python build.py -- --recipe <recipe.json> --out <new-directory>

The CLI is production-only at 4096.  ``run_build(..., atlas_px=...)`` is an
internal Python surface used by the Task 15 focused 512 orchestration test.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import bpy


TOOL_ROOT = Path(__file__).resolve().parent
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from tgel_stock import BLENDER_REQUIRED_VERSION  # noqa: E402
from tgel_stock import assert_clean_path  # noqa: E402
from tgel_stock import assemble  # noqa: E402
from tgel_stock import bake  # noqa: E402
from tgel_stock import canonical  # noqa: E402
from tgel_stock import export  # noqa: E402
from tgel_stock import highpoly  # noqa: E402
from tgel_stock import lod  # noqa: E402
from tgel_stock import manifest  # noqa: E402
from tgel_stock import materials  # noqa: E402
from tgel_stock import recipe as recipe_module  # noqa: E402
from tgel_stock import scene  # noqa: E402
from tgel_stock import uvmap  # noqa: E402
from tgel_stock import validate  # noqa: E402


PRODUCTION_ATLAS_PX = 4096
LOD_SUFFIX_RE = re.compile(r"__LOD[12]$")
FROZEN_NUMERIC_FIELDS = (
    ("lengthOverCouplers", "length_over_couplers"),
    ("bodyLength", "body_length"),
    ("width", "width"),
    ("height", "height"),
    ("wheelRadius", "wheel_radius"),
    ("bogieCentreOffset", "bogie_centre_offset"),
    ("bogieWheelbase", "bogie_wheelbase"),
    ("bogiePivotHeight", "bogie_pivot_height"),
)
FROZEN_RECIPE_SPECS = {
    "wagon": {
        "model_id": "rolling-stock.wagon.40ft-box-v2",
        "metrics": {
            "length_over_couplers": 13.5128,
            "body_length": 12.7508,
            "width": 3.2512,
            "height": 4.4196,
            "wheel_radius": 0.4191,
            "bogie_centre_offset": 4.699,
            "bogie_wheelbase": 1.6764,
            "bogie_pivot_height": 0.96,
        },
        "seed": 190702,
        "livery": {
            "body": (0.196, 0.070, 0.047),
            "frame": (0.055, 0.055, 0.058),
            "runningGear": (0.038, 0.036, 0.034),
            "roof": (0.072, 0.070, 0.066),
            "stencil": (0.75, 0.73, 0.66),
            "glass": (0.10, 0.12, 0.13),
        },
    },
    "locomotive": {
        "model_id": "rolling-stock.locomotive.road-switcher-v2",
        "metrics": {
            "length_over_couplers": 17.1196,
            "body_length": 16.0,
            "width": 3.1242,
            "height": 4.4196,
            "wheel_radius": 0.508,
            "bogie_centre_offset": 4.7244,
            "bogie_wheelbase": 2.7432,
            "bogie_pivot_height": 1.10,
        },
        "seed": 190701,
        "livery": {
            "body": (0.043, 0.093, 0.063),
            "frame": (0.055, 0.055, 0.058),
            "runningGear": (0.038, 0.036, 0.034),
            "roof": (0.072, 0.070, 0.066),
            "stencil": (0.75, 0.73, 0.66),
            "glass": (0.10, 0.12, 0.13),
        },
    },
}


def _clean_resolved_path(path):
    resolved = Path(path).expanduser().resolve()
    assert_clean_path(str(resolved))
    return resolved


def _validate_atlas_px(atlas_px):
    if isinstance(atlas_px, bool):
        raise ValueError("atlas_px must be a positive integer")
    try:
        numeric = int(atlas_px)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("atlas_px must be a positive integer") from exc
    if numeric <= 0 or numeric != atlas_px:
        raise ValueError("atlas_px must be a positive integer")
    return numeric


def _require_blender_version():
    if bpy.app.version_string != BLENDER_REQUIRED_VERSION:
        raise RuntimeError(
            f"Blender {bpy.app.version_string} != required "
            f"{BLENDER_REQUIRED_VERSION}")


def _recipe_digest(recipe_path):
    return canonical.file_sha256(str(recipe_path))


def _matches_frozen_number(value, expected):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value == expected
    except (OverflowError, TypeError, ValueError):
        return False


def _matches_frozen_rgb(value, expected):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(
            _matches_frozen_number(component, expected_component)
            for component, expected_component in zip(value, expected)
        )
    )


def _load_frozen_recipe(recipe_path):
    recipe = recipe_module.load(str(recipe_path))
    with open(recipe_path, encoding="utf-8") as handle:
        raw_document = json.load(handle)

    spec = FROZEN_RECIPE_SPECS.get(recipe.kind)
    if spec is None:
        raise ValueError(f"Recipe kind is not frozen: {recipe.kind!r}")
    if recipe.model_id != spec["model_id"]:
        raise ValueError(
            f"Recipe modelId {recipe.model_id!r} does not match frozen "
            f"{recipe.kind!r} modelId {spec['model_id']!r}")

    for json_name, attribute_name in FROZEN_NUMERIC_FIELDS:
        expected = spec["metrics"][attribute_name]
        actual = getattr(recipe, attribute_name)
        raw_value = raw_document.get(json_name)
        if (not _matches_frozen_number(actual, expected)
                or not _matches_frozen_number(raw_value, expected)):
            raise ValueError(
                f"Recipe {json_name} {raw_value!r} does not match frozen "
                f"{recipe.kind!r} value {expected!r}")

    raw_seed = raw_document.get("seed")
    if (type(raw_seed) is not int or raw_seed != spec["seed"]
            or type(recipe.seed) is not int or recipe.seed != spec["seed"]):
        raise ValueError(
            f"Recipe seed {raw_seed!r} does not match frozen "
            f"{recipe.kind!r} value {spec['seed']!r}")

    expected_livery = spec["livery"]
    raw_livery = raw_document.get("livery")
    if (not isinstance(raw_livery, dict)
            or set(raw_livery) != set(expected_livery)
            or set(recipe.livery) != set(expected_livery)):
        raise ValueError(
            f"Recipe livery keys do not match frozen {recipe.kind!r} keys")
    for name, expected_rgb in expected_livery.items():
        raw_rgb = raw_livery[name]
        actual_rgb = recipe.livery[name]
        if (not _matches_frozen_rgb(raw_rgb, expected_rgb)
                or not _matches_frozen_rgb(actual_rgb, expected_rgb)):
            raise ValueError(
                f"Recipe livery {name!r} {raw_rgb!r} does not match frozen "
                f"{recipe.kind!r} RGB {expected_rgb!r}")
    return recipe


def _script_paths():
    paths = [Path(__file__).resolve()]
    paths.extend(
        path.resolve()
        for path in (TOOL_ROOT / "tgel_stock").rglob("*.py")
        if path.is_file())
    return sorted(set(paths), key=lambda path: path.relative_to(TOOL_ROOT).as_posix())


def _script_digest():
    """Hashes stable relative names plus exact bytes of build.py + package."""
    hasher = hashlib.sha256()
    for path in _script_paths():
        relative = path.relative_to(TOOL_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        hasher.update(len(relative).to_bytes(8, "little"))
        hasher.update(relative)
        hasher.update(len(payload).to_bytes(8, "little"))
        hasher.update(payload)
    return hasher.hexdigest()


def _set_manifest_meta(document, recipe, atlas_px, recipe_digest, script_digest):
    document.set_meta(
        recipe.model_id,
        recipe_digest,
        script_digest,
        bpy.app.version_string,
        kind=recipe.kind,
        atlas_resolution=(atlas_px, atlas_px),
        length_over_couplers=recipe.length_over_couplers,
        body_length=recipe.body_length,
        width=recipe.width,
        height=recipe.height,
        track_gauge=recipe_module.GAUGE,
        wheel_back_to_back=recipe_module.BACK_TO_BACK,
        wheel_width=recipe_module.WHEEL_WIDTH,
        wheel_radius=recipe.wheel_radius,
        coupler_height=recipe_module.COUPLER_HEIGHT,
        coupler_pivot_to_face=recipe_module.COUPLER_PIVOT_TO_FACE,
        bogie_centre_offset=recipe.bogie_centre_offset,
        bogie_wheelbase=recipe.bogie_wheelbase,
        bogie_pivot_height=recipe.bogie_pivot_height,
    )


def _base_mesh_name(name):
    return LOD_SUFFIX_RE.sub("", name)


def _export_mesh_name_records(objects):
    """Preflights the exact object/manifest names and mesh ownership."""
    records = []
    mesh_owners = {}
    for name in sorted(objects):
        obj = objects[name]
        if obj is None or obj.type != 'MESH' or obj.data is None:
            raise ValueError(f"Export object {name!r} is not a mesh")
        if obj.name != name:
            raise ValueError(
                f"Export key {name!r} != Blender object name {obj.name!r}")
        mesh = obj.data
        if mesh.users != 1:
            raise ValueError(
                f"Export mesh {name!r} has {mesh.users} users; expected 1")
        pointer = mesh.as_pointer()
        previous = mesh_owners.get(pointer)
        if previous is not None:
            raise ValueError(
                f"Export objects {previous!r} and {name!r} share one mesh")
        mesh_owners[pointer] = name
        records.append((name, obj, mesh))

    scope = set(mesh_owners)
    for target_name, _obj, mesh in records:
        existing = bpy.data.meshes.get(target_name)
        if existing is not None and existing.as_pointer() not in scope:
            raise ValueError(
                f"Export mesh target {target_name!r} collides with "
                f"out-of-scope mesh {existing.name!r}")
        if existing is mesh:
            continue
    return records


def _unused_mesh_staging_name(label, index):
    attempt = 0
    while True:
        candidate = f".TGEL.ExportMeshName.{label}.{index:04d}.{attempt:02d}"
        if bpy.data.meshes.get(candidate) is None:
            return candidate
        attempt += 1


def _stage_export_mesh_names(records, label):
    for index, (_target_name, _obj, mesh) in enumerate(records):
        staging_name = _unused_mesh_staging_name(label, index)
        mesh.name = staging_name
        if mesh.name != staging_name:
            raise RuntimeError(
                f"Blender changed mesh staging name {staging_name!r} "
                f"to {mesh.name!r}")


def _restore_export_mesh_names(records, original_names):
    _stage_export_mesh_names(records, "rollback")
    for (_target_name, _obj, mesh), original_name in zip(
            records, original_names):
        mesh.name = original_name
        if mesh.name != original_name:
            raise RuntimeError(
                f"Could not restore mesh name {original_name!r}; "
                f"Blender produced {mesh.name!r}")


def assert_export_mesh_names(objects):
    """Asserts the FBX Geometry name equals each object/manifest name."""
    records = _export_mesh_name_records(objects)
    mismatches = [
        (target_name, mesh.name)
        for target_name, _obj, mesh in records
        if mesh.name != target_name
    ]
    if mismatches:
        raise ValueError(
            f"Export mesh datablock names differ for {len(mismatches)}/"
            f"{len(records)} objects: {mismatches[:8]}")


def prepare_export_mesh_names(objects):
    """Transactionally aligns every FBX Geometry ID with its manifest name.

    All objects must own distinct single-user meshes.  An out-of-scope mesh
    occupying any target name is rejected before the first write.  In-scope
    name permutations are safe because every mesh first moves to a unique
    staging name; an exceptional rename restores all original datablock names.
    """
    records = _export_mesh_name_records(objects)
    original_names = [mesh.name for _name, _obj, mesh in records]
    try:
        _stage_export_mesh_names(records, "apply")
        for target_name, _obj, mesh in records:
            mesh.name = target_name
            if mesh.name != target_name:
                raise RuntimeError(
                    f"Blender changed export mesh target {target_name!r} "
                    f"to {mesh.name!r}")
        assert_export_mesh_names(objects)
    except Exception as original_error:
        try:
            _restore_export_mesh_names(records, original_names)
        except Exception as rollback_error:
            raise RuntimeError(
                "Export mesh-name normalization failed and rollback could "
                f"not restore the original names: {rollback_error}") from original_error
        raise


def _fill_manifest_geometry(document, assembly, objects):
    for node in assembly.nodes:
        document.add_node(
            node.path,
            node.parent,
            node.local_position,
            node.local_rotation_quat)

    for name in sorted(objects):
        obj = objects[name]
        if obj is None or obj.type != 'MESH' or obj.data is None:
            raise ValueError(f"Manifest object {name!r} is not a mesh")
        if obj.name != name:
            raise ValueError(
                f"Manifest key {name!r} != Blender object name {obj.name!r}")
        base_name = _base_mesh_name(name)
        node_path = assembly.mesh_nodes.get(base_name)
        if node_path is None:
            raise ValueError(
                f"Manifest mesh {name!r} has no exact LOD0 node mapping "
                f"for base {base_name!r}")
        positions, normals, uvs, triangles, bounds_min, bounds_max = (
            scene.collect_mesh_data(obj))
        document.add_mesh(
            name,
            node_path,
            positions,
            normals,
            uvs,
            triangles,
            bounds_min,
            bounds_max)


def _geometry_manifest(
        recipe, assembly, objects, atlas_px, recipe_digest, script_digest):
    assert_export_mesh_names(objects)
    document = manifest.Manifest()
    _set_manifest_meta(
        document, recipe, atlas_px, recipe_digest, script_digest)
    _fill_manifest_geometry(document, assembly, objects)
    return document


def _build_geometry(recipe_path, atlas_px):
    scene.reset()
    _require_blender_version()
    recipe = _load_frozen_recipe(recipe_path)
    assembly = assemble.build_vehicle(recipe)
    uvmap.unwrap_and_pack(assembly.objects, atlas_px=atlas_px)
    validate.validate_assembly(assembly, recipe)
    objects = lod.build_lods(assembly.objects)
    prepare_export_mesh_names(objects)
    return recipe, assembly, objects


def build_geometry_snapshot(recipe_path, atlas_px=PRODUCTION_ATLAS_PX):
    """Returns an in-memory manifest snapshot without textures or FBX."""
    atlas_px = _validate_atlas_px(atlas_px)
    recipe_path = _clean_resolved_path(recipe_path)
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Recipe not found: {recipe_path}")
    recipe_digest = _recipe_digest(recipe_path)
    script_digest = _script_digest()
    recipe, assembly, objects = _build_geometry(recipe_path, atlas_px)
    return _geometry_manifest(
        recipe,
        assembly,
        objects,
        atlas_px,
        recipe_digest,
        script_digest).to_dict()


def atomic_write_directory(destination, writer):
    """Publishes one newly-created output directory with a same-volume rename.

    Existing destinations are rejected before ``writer`` runs.  Windows does
    not provide an honest atomic replacement of a non-empty directory, so
    replacement is intentionally outside this contract.
    """
    destination = _clean_resolved_path(destination)
    if destination.exists():
        raise FileExistsError(f"Output destination already exists: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.staging-",
        dir=str(parent)))
    published = False
    try:
        writer(staging)
        if destination.exists():
            raise FileExistsError(
                f"Output destination appeared during build: {destination}")
        os.replace(staging, destination)
        published = True
        return destination
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _texture_paths(staging, detail_paths, albedo_path, mask_path):
    paths = {
        "albedo": os.path.abspath(os.fspath(albedo_path)),
        "normal": os.path.abspath(os.fspath(detail_paths["normal"])),
        "mask": os.path.abspath(os.fspath(mask_path)),
    }
    expected_parent = staging.resolve()
    for name, path in paths.items():
        if Path(path).resolve().parent != expected_parent:
            raise ValueError(
                f"Texture {name} escaped staging directory: {path}")
    return paths


def _staging_output_path(staging, filename):
    expected_parent = Path(staging).resolve()
    path = (expected_parent / filename).resolve()
    if path.parent != expected_parent:
        raise ValueError(f"Output path escaped staging directory: {path}")
    return path


def _remove_intermediates(detail_paths):
    for key in ("ao", "curvature"):
        path = Path(detail_paths[key])
        if not path.is_file():
            raise ValueError(f"Expected bake intermediate missing: {path}")
        path.unlink()


def _assert_exact_outputs(staging, recipe):
    expected = {
        f"{recipe.model_id}.fbx",
        "albedo.png",
        "normal.png",
        "mask.png",
        f"{recipe.model_id}.manifest.json",
    }
    entries = list(staging.iterdir())
    non_files = sorted(path.name for path in entries if not path.is_file())
    actual = {path.name for path in entries if path.is_file()}
    if non_files or actual != expected:
        raise ValueError(
            f"Final output set differs; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}, nonFiles={non_files}")
    for path in entries:
        if path.stat().st_size <= 0:
            raise ValueError(f"Final output is empty: {path}")


def run_build(recipe_path, out_dir, atlas_px=PRODUCTION_ATLAS_PX):
    """Runs one complete build and atomically publishes a new directory.

    ``atlas_px`` is an internal focused-test seam.  ``main`` never exposes it
    and always calls this function with ``PRODUCTION_ATLAS_PX``.
    """
    atlas_px = _validate_atlas_px(atlas_px)
    recipe_path = _clean_resolved_path(recipe_path)
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Recipe not found: {recipe_path}")
    recipe = _load_frozen_recipe(recipe_path)
    recipe_digest = _recipe_digest(recipe_path)
    script_digest = _script_digest()

    def write(staging):
        # Build LODs only after baked materials exist; unlike the cheap
        # snapshot path, this path must first let LOD clones inherit them.
        scene.reset()
        _require_blender_version()
        assembly = assemble.build_vehicle(recipe)
        uvmap.unwrap_and_pack(assembly.objects, atlas_px=atlas_px)
        validate.validate_assembly(assembly, recipe)

        try:
            sources = highpoly.build_bake_sources(
                assembly.objects, seed=recipe.seed)
            detail_paths = bake.bake_detail(
                assembly.objects,
                sources,
                str(staging),
                atlas_px=atlas_px)
        finally:
            highpoly.cleanup_bake_sources()

        albedo_path = materials.bake_albedo(
            assembly.objects,
            recipe,
            str(staging),
            atlas_px=atlas_px)
        mask_path = materials.bake_weathering_masks(
            assembly.objects,
            recipe,
            detail_paths["ao"],
            detail_paths["curvature"],
            str(staging),
            atlas_px=atlas_px)

        objects = lod.build_lods(assembly.objects)
        prepare_export_mesh_names(objects)
        texture_paths = _texture_paths(
            staging, detail_paths, albedo_path, mask_path)
        validate.validate_textures(
            texture_paths, expected_resolution=atlas_px)

        fbx_path = _staging_output_path(
            staging, f"{recipe.model_id}.fbx")
        assert_export_mesh_names(objects)
        export.export_fbx(
            [objects[name] for name in sorted(objects)],
            str(fbx_path))
        if not fbx_path.is_file() or fbx_path.stat().st_size <= 0:
            raise ValueError(f"FBX export missing or empty: {fbx_path}")

        document = _geometry_manifest(
            recipe,
            assembly,
            objects,
            atlas_px,
            recipe_digest,
            script_digest)
        for name, color_space in (
                ("albedo", "sRGB"),
                ("normal", "Non-Color"),
                ("mask", "Non-Color")):
            document.add_texture(
                name,
                texture_paths[name],
                (atlas_px, atlas_px),
                color_space)
        manifest_path = _staging_output_path(
            staging, f"{recipe.model_id}.manifest.json")
        document.write(str(manifest_path))

        _remove_intermediates(detail_paths)
        _assert_exact_outputs(staging, recipe)

    return atomic_write_directory(out_dir, write)


def _public_argv(argv):
    if argv is not None:
        return list(argv)
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1:]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build one TGEL generated rolling-stock model at 4096")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(_public_argv(argv))
    output = run_build(
        arguments.recipe,
        arguments.out,
        atlas_px=PRODUCTION_ATLAS_PX)
    print(f"[build] OK {output}", flush=True)
    return 0


if __name__ == "__main__":
    main()
