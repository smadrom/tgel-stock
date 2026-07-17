"""Task 15 production validation for generated rolling-stock assemblies.

Geometry is authored node-local in Blender, while the node table is expressed
in vehicle/Unity space.  Validation therefore composes the node hierarchy in
vehicle space before checking the frozen physical envelope, wheel seating and
coupler pulling planes.  Texture validation is separate because an Assembly
has no filesystem-output ownership.
"""

import math
import os

import bpy
from mathutils import Quaternion, Vector
import numpy as np

from . import assert_clean_path
from . import recipe as recipe_module
from . import scene
from . import uvmap


POSITION_TOLERANCE = 0.001
# The wheel profile's 25 mm flange tip intentionally runs below the railhead
# origin.  Vehicle height is railhead-to-roof, not total mesh AABB height.
MAX_FLANGE_BELOW_RAILHEAD = 0.025
UNIT_TOLERANCE = 1e-5
UV_TOLERANCE = 1e-6
GEOMETRY_AREA_EPSILON = 1e-12
# Current small BrakeWheel triangles legitimately reach ~3.6e-13 of packed
# UV area.  Exact point collapse is zero; 1e-16 separates that defect without
# rejecting those authored nonzero islands.
UV_AREA_EPSILON = 1e-16
ROUGHNESS_RANGE_MIN = 1.0 / 255.0
PRODUCTION_ATLAS_PX = 4096
MIN_DENSITY_PX_PER_M = 45.0
MIN_COVERAGE = 0.05
MAX_OVERLAP = 0.01

EXPECTED_TEXTURE_FILES = {
    "albedo": "albedo.png",
    "normal": "normal.png",
    "mask": "mask.png",
}


def _contract_parents():
    parents = {
        "Visuals": "",
        "Visuals/Body": "Visuals",
        "Visuals/RunningGear": "Visuals",
    }
    for bogie_id in ("front", "rear"):
        bogie = f"Visuals/RunningGear/Bogie__{bogie_id}"
        parents[bogie] = "Visuals/RunningGear"
        parents[f"{bogie}/Frame"] = bogie
        for wheelset_id in ("leading", "trailing"):
            wheelset = f"{bogie}/Wheelset__{bogie_id}_{wheelset_id}"
            parents[wheelset] = bogie
            parents[f"{wheelset}/WheelLeft"] = wheelset
            parents[f"{wheelset}/WheelRight"] = wheelset

    parents["Couplers"] = ""
    for end in ("front", "rear"):
        pivot = f"Couplers/CouplerPivot__{end}"
        parents[pivot] = "Couplers"
        parents[f"{pivot}/CouplerFace"] = pivot

    parents.update({
        "Markers": "",
        "Markers/BogieTrack__front": "Markers",
        "Markers/BogieTrack__rear": "Markers",
        "Interaction": "",
    })
    return parents


CONTRACT_PARENTS = _contract_parents()


def _finite_vector(values, width, label):
    values = tuple(float(value) for value in values)
    if len(values) != width or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain {width} finite values")
    return values


def _node_map(assembly):
    nodes = {}
    for node in assembly.nodes:
        if node.path in nodes:
            raise ValueError(f"Duplicate contract node: {node.path}")
        nodes[node.path] = node

    expected = set(CONTRACT_PARENTS)
    actual = set(nodes)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Contract node table differs; missing={missing}, extra={extra}")

    for path, expected_parent in CONTRACT_PARENTS.items():
        node = nodes[path]
        if node.parent != expected_parent:
            raise ValueError(
                f"Node {path} parent {node.parent!r} != {expected_parent!r}")
        _finite_vector(node.local_position, 3, f"Node {path} localPosition")
        quaternion = _finite_vector(
            node.local_rotation_quat, 4, f"Node {path} localRotation")
        length = math.sqrt(sum(component * component for component in quaternion))
        if abs(length - 1.0) > UNIT_TOLERANCE:
            raise ValueError(f"Node {path} quaternion length {length} != 1")
    return nodes


def _world_transforms(nodes):
    transforms = {}
    visiting = set()

    def resolve(path):
        if path in transforms:
            return transforms[path]
        if path in visiting:
            raise ValueError(f"Node hierarchy cycle at {path}")
        visiting.add(path)
        node = nodes[path]
        x, y, z, w = node.local_rotation_quat
        local_rotation = Quaternion((w, x, y, z)).normalized()
        local_position = Vector(node.local_position)
        if node.parent:
            parent_position, parent_rotation = resolve(node.parent)
            world_position = parent_position + parent_rotation @ local_position
            world_rotation = parent_rotation @ local_rotation
        else:
            world_position = local_position
            world_rotation = local_rotation
        visiting.remove(path)
        transforms[path] = (world_position, world_rotation.normalized())
        return transforms[path]

    for path in sorted(nodes):
        resolve(path)
    return transforms


def _assert_near(actual, expected, label, tolerance=POSITION_TOLERANCE):
    actual = float(actual)
    expected = float(expected)
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(
            f"{label} must be finite; actual={actual}, expected={expected}")
    if abs(actual - expected) > tolerance:
        raise ValueError(
            f"{label} {actual:.9f} != {expected:.9f} "
            f"within {tolerance}")


def _assert_vector_near(actual, expected, label, tolerance=POSITION_TOLERANCE):
    if len(actual) != len(expected):
        raise ValueError(f"{label} vector width mismatch")
    for axis, (actual_value, expected_value) in enumerate(zip(actual, expected)):
        _assert_near(
            actual_value, expected_value, f"{label}[{axis}]", tolerance)


def _validate_frozen_constants():
    expected = {
        "GAUGE": 1.435,
        "BACK_TO_BACK": 1.348,
        "WHEEL_WIDTH": 0.135,
        "COUPLER_HEIGHT": 0.860,
        "COUPLER_PIVOT_TO_FACE": 0.600,
    }
    for name, value in expected.items():
        _assert_near(
            getattr(recipe_module, name), value, f"recipe.{name}", UNIT_TOLERANCE)


def _expected_local_contract(recipe):
    identity = (0.0, 0.0, 0.0, 1.0)
    yaw_180 = (0.0, 1.0, 0.0, 0.0)
    seat = (recipe_module.BACK_TO_BACK + recipe_module.WHEEL_WIDTH) * 0.5
    expected = {
        path: ((0.0, 0.0, 0.0), identity)
        for path in CONTRACT_PARENTS
    }
    for bogie_id in ("front", "rear"):
        bogie_sign = 1.0 if bogie_id == "front" else -1.0
        bogie_path = f"Visuals/RunningGear/Bogie__{bogie_id}"
        expected[bogie_path] = (
            (0.0, recipe.bogie_pivot_height,
             bogie_sign * recipe.bogie_centre_offset),
            identity)
        for wheelset_id in ("leading", "trailing"):
            wheel_sign = 1.0 if wheelset_id == "leading" else -1.0
            wheelset_path = f"{bogie_path}/Wheelset__{bogie_id}_{wheelset_id}"
            expected[wheelset_path] = (
                (0.0, recipe.wheel_radius - recipe.bogie_pivot_height,
                 wheel_sign * recipe.bogie_wheelbase * 0.5),
                identity)
            expected[f"{wheelset_path}/WheelLeft"] = (
                (-seat, 0.0, 0.0),
                identity)
            expected[f"{wheelset_path}/WheelRight"] = (
                (seat, 0.0, 0.0),
                yaw_180)

    expected["Markers/BogieTrack__front"] = (
        (0.0, 0.0, recipe.bogie_centre_offset),
        identity)
    expected["Markers/BogieTrack__rear"] = (
        (0.0, 0.0, -recipe.bogie_centre_offset),
        identity)

    half_length = recipe.length_over_couplers * 0.5
    face = recipe_module.COUPLER_PIVOT_TO_FACE
    expected["Couplers/CouplerPivot__front"] = (
        (0.0, recipe_module.COUPLER_HEIGHT, half_length - face),
        identity)
    expected["Couplers/CouplerPivot__rear"] = (
        (0.0, recipe_module.COUPLER_HEIGHT, -half_length + face),
        yaw_180)
    for end in ("front", "rear"):
        path = f"Couplers/CouplerPivot__{end}/CouplerFace"
        expected[path] = ((0.0, 0.0, face), identity)
    return expected


def _validate_authored_node_transforms(recipe, nodes):
    expected = _expected_local_contract(recipe)
    for path in sorted(expected):
        expected_position, expected_rotation = expected[path]
        _assert_vector_near(
            nodes[path].local_position, expected_position, f"{path} localPosition")
        actual_rotation = tuple(float(value) for value in nodes[path].local_rotation_quat)
        dot = sum(
            actual * target
            for actual, target in zip(actual_rotation, expected_rotation))
        # q and -q encode the same rotation.  Both inputs are unit length by
        # _node_map, so abs(dot) near one is the sign-equivalent comparison.
        if 1.0 - abs(dot) > UNIT_TOLERANCE:
            raise ValueError(
                f"{path} localRotation {actual_rotation} != "
                f"{expected_rotation} sign-equivalently")


def _validate_mesh_registry(assembly, nodes):
    if set(assembly.objects) != set(assembly.mesh_nodes):
        missing = sorted(set(assembly.objects) - set(assembly.mesh_nodes))
        extra = sorted(set(assembly.mesh_nodes) - set(assembly.objects))
        raise ValueError(
            f"Assembly mesh-node registry differs; missing={missing}, extra={extra}")
    for name, obj in assembly.objects.items():
        if obj is None or obj.type != 'MESH' or obj.data is None:
            raise ValueError(f"Assembly object {name!r} is not a mesh")
        if obj.name != name:
            raise ValueError(
                f"Assembly key {name!r} != Blender object name {obj.name!r}")
        if obj.parent is not None:
            raise ValueError(
                f"Assembly mesh {name!r} must be flat-exported without a parent")
        if not obj.users_collection:
            raise ValueError(
                f"Assembly mesh {name!r} is not linked to a collection")
        if bpy.context.scene.objects.get(obj.name) is not obj:
            raise ValueError(
                f"Assembly mesh {name!r} is not linked to the current scene")
        expected_node_path = name.rsplit(".", 1)[0].replace(".", "/")
        actual_node_path = assembly.mesh_nodes[name]
        if actual_node_path != expected_node_path:
            raise ValueError(
                f"Assembly mesh {name!r} node {actual_node_path!r} != "
                f"name-implied node {expected_node_path!r}")
        if expected_node_path not in nodes:
            raise ValueError(
                f"Assembly mesh {name!r} name-implied node "
                f"{expected_node_path!r} does not exist")
        if len(obj.data.vertices) == 0 or len(obj.data.polygons) == 0:
            raise ValueError(f"Assembly mesh {name!r} is empty")
        identity = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        for row_index, row in enumerate(obj.matrix_local):
            for column_index, value in enumerate(row):
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(
                        f"Assembly mesh {name!r} has non-finite object transform")
                if abs(value - identity[row_index][column_index]) > UNIT_TOLERANCE:
                    raise ValueError(
                        f"Assembly mesh {name!r} has non-identity object transform")
        for vertex in obj.data.vertices:
            coordinates = tuple(float(component) for component in vertex.co)
            normals = tuple(float(component) for component in vertex.normal)
            if not all(math.isfinite(component) for component in coordinates):
                raise ValueError(
                    f"Assembly mesh {name!r} has non-finite vertex coordinates")
            if not all(math.isfinite(component) for component in normals):
                raise ValueError(
                    f"Assembly mesh {name!r} has non-finite vertex normals")


def _validate_envelope(assembly, recipe, transforms):
    half_width = recipe.width * 0.5
    half_length = recipe.length_over_couplers * 0.5
    half_body_length = recipe.body_length * 0.5
    for name, obj in assembly.objects.items():
        minimum_y = -POSITION_TOLERANCE
        if name.endswith(".TreadAndFlange"):
            minimum_y = -MAX_FLANGE_BELOW_RAILHEAD - POSITION_TOLERANCE
        node_path = assembly.mesh_nodes[name]
        node_position, node_rotation = transforms[node_path]
        for vertex in obj.data.vertices:
            local = Vector(scene.to_unity(vertex.co))
            world = node_position + node_rotation @ local
            if abs(float(world.x)) > half_width + POSITION_TOLERANCE:
                raise ValueError(
                    f"Assembly envelope width exceeded by {name}: x={world.x}")
            if (float(world.y) < minimum_y
                    or float(world.y) > recipe.height + POSITION_TOLERANCE):
                raise ValueError(
                    f"Assembly envelope height exceeded by {name}: y={world.y}")
            if abs(float(world.z)) > half_length + POSITION_TOLERANCE:
                raise ValueError(
                    f"Assembly envelope length exceeded by {name}: z={world.z}")
            brake_wheel_exception = (
                recipe.kind == "wagon" and name == "Visuals.Body.BrakeWheel")
            if (name.startswith("Visuals.Body.")
                    and not brake_wheel_exception
                    and abs(float(world.z))
                    > half_body_length + POSITION_TOLERANCE):
                raise ValueError(
                    f"Assembly body-length envelope exceeded by {name}: "
                    f"z={world.z}")


def _validate_wheels(assembly, recipe, transforms):
    expected_half_width = recipe_module.WHEEL_WIDTH * 0.5
    for bogie_id in ("front", "rear"):
        for wheelset_id in ("leading", "trailing"):
            wheelset = (
                f"Visuals/RunningGear/Bogie__{bogie_id}/"
                f"Wheelset__{bogie_id}_{wheelset_id}")
            flange_positions = []
            for side in ("Left", "Right"):
                node_path = f"{wheelset}/Wheel{side}"
                mesh_name = (
                    f"{node_path.replace('/', '.')}.TreadAndFlange")
                obj = assembly.objects.get(mesh_name)
                if obj is None:
                    raise ValueError(f"Missing wheel tread mesh: {mesh_name}")
                positions, _normals, _uvs, _triangles, bounds_min, bounds_max = (
                    scene.collect_mesh_data(obj))
                _assert_near(
                    bounds_min[0], -expected_half_width,
                    f"{mesh_name} local min x")
                _assert_near(
                    bounds_max[0], expected_half_width,
                    f"{mesh_name} local max x")

                radial = [
                    math.hypot(float(position[1]), float(position[2]))
                    for position in positions
                ]
                max_radial = max(radial)
                _assert_near(
                    max_radial,
                    recipe.wheel_radius + 0.025,
                    f"{mesh_name} flange-tip radius")
                tread_band = [
                    radius for radius in radial
                    if abs(radius - recipe.wheel_radius) <= POSITION_TOLERANCE
                ]
                if not tread_band:
                    raise ValueError(
                        f"{mesh_name} has no authored tread-radius band near "
                        f"{recipe.wheel_radius}")
                _assert_near(
                    sum(tread_band) / len(tread_band),
                    recipe.wheel_radius,
                    f"{mesh_name} tread-band radius")
                flange_tip_x = [
                    float(position[0])
                    for position, radius in zip(positions, radial)
                    if max_radial - radius <= UNIT_TOLERANCE
                ]
                if not flange_tip_x:
                    raise ValueError(
                        f"{mesh_name} has no measurable max-radius flange band")
                for value in flange_tip_x:
                    _assert_near(
                        value,
                        expected_half_width,
                        f"{mesh_name} flange-tip local x")
                measured_flange_x = sum(flange_tip_x) / len(flange_tip_x)
                node_position, node_rotation = transforms[node_path]
                flange = node_position + node_rotation @ Vector(
                    (measured_flange_x, 0.0, 0.0))
                flange_positions.append(float(flange.x))
            measured = abs(flange_positions[1] - flange_positions[0])
            _assert_near(
                measured, recipe_module.BACK_TO_BACK,
                f"{wheelset} measured back-to-back")


def _validate_coupler_faces(recipe, transforms):
    half_length = recipe.length_over_couplers * 0.5
    expected = {
        "front": (0.0, recipe_module.COUPLER_HEIGHT, half_length),
        "rear": (0.0, recipe_module.COUPLER_HEIGHT, -half_length),
    }
    for end, target in expected.items():
        path = f"Couplers/CouplerPivot__{end}/CouplerFace"
        position, _rotation = transforms[path]
        _assert_vector_near(position, target, f"{path} world position")


def _validate_uvs(assembly):
    for name, obj in assembly.objects.items():
        mesh = obj.data
        uv_layer = mesh.uv_layers.active
        if uv_layer is None:
            raise ValueError(f"Assembly mesh {name!r} has no active UV layer")
        if len(uv_layer.data) != len(mesh.loops):
            raise ValueError(
                f"Assembly mesh {name!r} UV loop count {len(uv_layer.data)} "
                f"!= mesh loop count {len(mesh.loops)}")
        coordinates = [
            (float(loop_uv.uv[0]), float(loop_uv.uv[1]))
            for loop_uv in uv_layer.data
        ]
        if not coordinates:
            raise ValueError(f"Assembly mesh {name!r} has no UV coordinates")
        for coordinate in coordinates:
            if not all(math.isfinite(component) for component in coordinate):
                raise ValueError(f"Assembly mesh {name!r} has non-finite UVs")
            if not all(
                    -UV_TOLERANCE <= component <= 1.0 + UV_TOLERANCE
                    for component in coordinate):
                raise ValueError(
                    f"Assembly mesh {name!r} has UV outside [0,1]: {coordinate}")

        mesh.calc_loop_triangles()
        area = 0.0
        for triangle in mesh.loop_triangles:
            uv0, uv1, uv2 = (
                uv_layer.data[loop_index].uv
                for loop_index in triangle.loops)
            uv_area = abs(
                (float(uv1[0]) - float(uv0[0]))
                * (float(uv2[1]) - float(uv0[1]))
                - (float(uv2[0]) - float(uv0[0]))
                * (float(uv1[1]) - float(uv0[1]))) * 0.5
            if not math.isfinite(uv_area):
                raise ValueError(
                    f"Assembly mesh {name!r} triangle {triangle.index} "
                    f"has non-finite UV area")
            if (float(triangle.area) > GEOMETRY_AREA_EPSILON
                    and uv_area <= UV_AREA_EPSILON):
                raise ValueError(
                    f"Assembly mesh {name!r} nondegenerate triangle "
                    f"{triangle.index} has collapsed UV area {uv_area}")
            area += uv_area
        if area <= UV_AREA_EPSILON:
            raise ValueError(f"Assembly mesh {name!r} has zero UV triangle area")

    report = uvmap.report(
        assembly.objects, atlas_px=PRODUCTION_ATLAS_PX)
    if report["density_px_per_m"] < MIN_DENSITY_PX_PER_M:
        raise ValueError(
            f"UV density {report['density_px_per_m']} < {MIN_DENSITY_PX_PER_M}")
    if report["coverage"] < MIN_COVERAGE:
        raise ValueError(f"UV coverage {report['coverage']} < {MIN_COVERAGE}")
    if report["max_overlap"] > MAX_OVERLAP:
        raise ValueError(
            f"UV overlap {report['max_overlap']} > {MAX_OVERLAP}")


def validate_assembly(assembly, recipe):
    """Raises ``ValueError`` when the frozen generated-assembly contract fails."""
    if assembly is None or recipe is None:
        raise ValueError("validate_assembly requires assembly and recipe")
    if getattr(assembly, "recipe", None) is not recipe:
        raise ValueError("Assembly recipe identity does not match validation recipe")

    _validate_frozen_constants()
    nodes = _node_map(assembly)
    _validate_authored_node_transforms(recipe, nodes)
    transforms = _world_transforms(nodes)
    _validate_mesh_registry(assembly, nodes)
    _validate_envelope(assembly, recipe, transforms)
    _validate_wheels(assembly, recipe, transforms)
    _validate_coupler_faces(recipe, transforms)
    _validate_uvs(assembly)


def _load_png(path, expected_resolution, label):
    path = os.path.abspath(os.fspath(path))
    assert_clean_path(path)
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise ValueError(f"Texture {label} missing or empty: {path}")
    if os.path.splitext(path)[1].lower() != ".png":
        raise ValueError(f"Texture {label} is not PNG: {path}")
    try:
        image = bpy.data.images.load(path, check_existing=False)
    except Exception as exc:
        raise ValueError(f"Texture {label} is not a readable PNG: {path}") from exc
    try:
        if tuple(int(value) for value in image.size) != (
                expected_resolution, expected_resolution):
            raise ValueError(
                f"Texture {label} resolution {tuple(image.size)} != "
                f"{(expected_resolution, expected_resolution)}")
        if int(image.channels) != 4:
            raise ValueError(
                f"Texture {label} channels {image.channels} != RGBA")
        return image
    except Exception:
        bpy.data.images.remove(image)
        raise


def validate_textures(paths, expected_resolution=PRODUCTION_ATLAS_PX):
    """Validates the exact final albedo/normal/mask PNG set.

    The packed mask is read in Non-Color space before pixel access and its
    alpha channel must carry real roughness variation.
    """
    if isinstance(expected_resolution, bool):
        raise ValueError("expected_resolution must be a positive integer")
    try:
        numeric_resolution = float(expected_resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "expected_resolution must be a positive integer") from exc
    if (not math.isfinite(numeric_resolution)
            or numeric_resolution <= 0.0
            or not numeric_resolution.is_integer()):
        raise ValueError("expected_resolution must be a positive integer")
    expected_resolution = int(numeric_resolution)
    if set(paths) != set(EXPECTED_TEXTURE_FILES):
        missing = sorted(set(EXPECTED_TEXTURE_FILES) - set(paths))
        extra = sorted(set(paths) - set(EXPECTED_TEXTURE_FILES))
        raise ValueError(
            f"Texture set differs; missing={missing}, extra={extra}")

    loaded = {}
    actual_filenames = {}
    try:
        for name, expected_filename in EXPECTED_TEXTURE_FILES.items():
            path = os.path.abspath(os.fspath(paths[name]))
            actual_filenames[name] = os.path.basename(path)
            loaded[name] = _load_png(path, expected_resolution, name)

        mask = loaded["mask"]
        try:
            mask.colorspace_settings.name = 'Non-Color'
        except TypeError as exc:
            raise ValueError("Packed mask cannot be read as Non-Color") from exc
        pixels = np.asarray(mask.pixels[:], dtype=np.float32)
        if pixels.size != expected_resolution * expected_resolution * 4:
            raise ValueError(
                f"Packed mask pixel count {pixels.size} is not RGBA "
                f"{expected_resolution}x{expected_resolution}")
        roughness = pixels[3::4]
        roughness_range = float(np.max(roughness) - np.min(roughness))
        if not math.isfinite(roughness_range) or roughness_range <= ROUGHNESS_RANGE_MIN:
            raise ValueError(
                f"Packed mask roughness alpha is constant: range={roughness_range}")
        for name, expected_filename in EXPECTED_TEXTURE_FILES.items():
            if actual_filenames[name] != expected_filename:
                raise ValueError(
                    f"Texture {name} filename {actual_filenames[name]!r} "
                    f"!= {expected_filename!r}")
    finally:
        for image in loaded.values():
            if bpy.data.images.get(image.name) is image:
                bpy.data.images.remove(image)
