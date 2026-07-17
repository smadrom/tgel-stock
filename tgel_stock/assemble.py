"""Vehicle assembly: composes part builders into a full node table + mesh set.

Pure composition (Task 10): this module authors NO geometry and introduces
NO new dimension constants. Every position below is either a recipe field
or one of the frozen ``recipe`` module constants (BACK_TO_BACK, WHEEL_WIDTH,
COUPLER_HEIGHT, COUPLER_PIVOT_TO_FACE). Mesh objects returned by the part
builders are node-local by construction and are left untransformed at the
Blender origin -- node placement lives ONLY in the node table (``.nodes``);
nothing here parents objects or moves vertices.

Node table contract (adjudicated naming map):

    Visuals                                     identity
        Visuals/Body                            identity; all body meshes
        Visuals/RunningGear                     identity
            .../Bogie__<front|rear>              (0, pivot_height, +/-bogie_centre_offset)
                .../Frame                        identity; 13 bogie frame meshes
                .../Wheelset__<end>_<leading|trailing>
                    (0, wheel_radius - pivot_height, +/-wheelbase/2)
                    (AxleVisual mesh attaches to the wheelset node itself)
                    .../WheelLeft                (-seat_offset, 0, 0), identity
                    .../WheelRight               (+seat_offset, 0, 0), yaw 180
    Couplers                                    identity
        Couplers/CouplerPivot__front            (0, COUPLER_HEIGHT, +LOC/2 - PIVOT_TO_FACE)
            .../CouplerFace                     (0, 0, PIVOT_TO_FACE)
        Couplers/CouplerPivot__rear             (0, COUPLER_HEIGHT, -LOC/2 + PIVOT_TO_FACE), yaw 180
            .../CouplerFace                     (0, 0, PIVOT_TO_FACE)
    Markers                                     identity
        Markers/BogieTrack__front               (0, 0, +bogie_centre_offset)
        Markers/BogieTrack__rear                (0, 0, -bogie_centre_offset)
    Interaction                                 identity (BoxCollider added Unity-side)

Mesh naming: every mesh OBJECT is renamed to its node path with ``/``
replaced by ``.``, plus ``.`` plus the part's short name -- e.g. body part
"LongHood" attaching to node "Visuals/Body" becomes object
"Visuals.Body.LongHood". ``wheel.build_wheelset_meshes`` already names its
returned objects this way when given the wheelset's dotted node path as its
``prefix`` argument, so those objects need no further renaming.
"""

from dataclasses import dataclass

from . import recipe as recipe_module
from . import scene
from .parts import body_locomotive
from .parts import body_wagon
from .parts import bogie
from .parts import coupler
from .parts import wheel

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
YAW_180_QUAT = (0.0, 1.0, 0.0, 0.0)

BODY_BUILDERS = {
    "locomotive": body_locomotive.build_meshes,
    "wagon": body_wagon.build_meshes,
}

BOGIE_IDS = ("front", "rear")
WHEELSET_IDS = ("leading", "trailing")
COUPLER_ENDS = ("front", "rear")


@dataclass(frozen=True)
class NodeSpec:
    path: str
    parent: str
    local_position: tuple
    local_rotation_quat: tuple


class Assembly:
    """Composed vehicle: mesh objects plus the node table that places them.

    ``objects`` maps a mesh object's full dotted name to the bpy Object.
    ``nodes`` lists every node in the naming-map contract (paths use ``/``).
    ``mesh_nodes`` maps a mesh object's dotted name to the ``/``-separated
    node path it attaches to (used by ``fill_manifest`` and available for
    inspection/tests).
    """

    def __init__(self, recipe):
        self.recipe = recipe
        self.objects = {}
        self.nodes = []
        self.mesh_nodes = {}

    def add_node(self, path, parent, local_position, local_rotation_quat):
        self.nodes.append(NodeSpec(
            path, parent, tuple(local_position), tuple(local_rotation_quat)))

    def add_mesh(self, obj, node_path, short_name):
        """Renames ``obj`` to ``<node_path with '.' separators>.<short_name>``
        and registers it under that name."""
        full_name = f"{node_path.replace('/', '.')}.{short_name}"
        obj.name = full_name
        self.objects[full_name] = obj
        self.mesh_nodes[full_name] = node_path
        return full_name

    def register_mesh(self, obj, node_path):
        """Registers a mesh object whose name is already the full contract
        name (as produced by ``wheel.build_wheelset_meshes`` given a dotted
        node-path prefix)."""
        self.objects[obj.name] = obj
        self.mesh_nodes[obj.name] = node_path

    def fill_manifest(self, m):
        for node in self.nodes:
            m.add_node(node.path, node.parent, node.local_position, node.local_rotation_quat)
        for name, obj in self.objects.items():
            node_path = self.mesh_nodes[name]
            positions, normals, uvs, triangles, bounds_min, bounds_max = (
                scene.collect_mesh_data(obj))
            m.add_mesh(name, node_path, positions, normals, uvs, triangles,
                       bounds_min, bounds_max)


def _build_visuals(assembly, recipe):
    assembly.add_node("Visuals", "", (0.0, 0.0, 0.0), IDENTITY_QUAT)
    assembly.add_node("Visuals/Body", "Visuals", (0.0, 0.0, 0.0), IDENTITY_QUAT)
    assembly.add_node("Visuals/RunningGear", "Visuals", (0.0, 0.0, 0.0), IDENTITY_QUAT)

    body_builder = BODY_BUILDERS[recipe.kind]
    for short_name, obj in body_builder(recipe).items():
        assembly.add_mesh(obj, "Visuals/Body", short_name)


def _build_running_gear(assembly, recipe):
    seat_offset = (recipe_module.BACK_TO_BACK + recipe_module.WHEEL_WIDTH) / 2.0

    for bogie_id in BOGIE_IDS:
        z_sign = 1.0 if bogie_id == "front" else -1.0
        bogie_path = f"Visuals/RunningGear/Bogie__{bogie_id}"
        assembly.add_node(
            bogie_path, "Visuals/RunningGear",
            (0.0, recipe.bogie_pivot_height, z_sign * recipe.bogie_centre_offset),
            IDENTITY_QUAT)

        frame_path = f"{bogie_path}/Frame"
        assembly.add_node(frame_path, bogie_path, (0.0, 0.0, 0.0), IDENTITY_QUAT)
        frame_meshes = bogie.build_frame_meshes(
            recipe.bogie_wheelbase, recipe.wheel_radius, recipe.bogie_pivot_height)
        for short_name, obj in frame_meshes.items():
            assembly.add_mesh(obj, frame_path, short_name)

        for wheelset_id in WHEELSET_IDS:
            wz_sign = 1.0 if wheelset_id == "leading" else -1.0
            wheelset_path = f"{bogie_path}/Wheelset__{bogie_id}_{wheelset_id}"
            assembly.add_node(
                wheelset_path, bogie_path,
                (0.0, recipe.wheel_radius - recipe.bogie_pivot_height,
                 wz_sign * recipe.bogie_wheelbase / 2.0),
                IDENTITY_QUAT)

            left_path = f"{wheelset_path}/WheelLeft"
            right_path = f"{wheelset_path}/WheelRight"
            assembly.add_node(left_path, wheelset_path, (-seat_offset, 0.0, 0.0), IDENTITY_QUAT)
            assembly.add_node(right_path, wheelset_path, (seat_offset, 0.0, 0.0), YAW_180_QUAT)

            prefix = wheelset_path.replace("/", ".")
            wheelset_meshes = wheel.build_wheelset_meshes(prefix, recipe.wheel_radius)
            for key, obj in wheelset_meshes.items():
                if key == "AxleVisual":
                    node_path = wheelset_path
                elif key.startswith("WheelLeft."):
                    node_path = left_path
                else:
                    node_path = right_path
                assembly.register_mesh(obj, node_path)


def _build_couplers(assembly, recipe):
    assembly.add_node("Couplers", "", (0.0, 0.0, 0.0), IDENTITY_QUAT)
    half_loc = recipe.length_over_couplers / 2.0
    face_z = recipe_module.COUPLER_PIVOT_TO_FACE

    for end in COUPLER_ENDS:
        pivot_path = f"Couplers/CouplerPivot__{end}"
        if end == "front":
            pivot_z = half_loc - face_z
            pivot_quat = IDENTITY_QUAT
        else:
            pivot_z = -half_loc + face_z
            pivot_quat = YAW_180_QUAT

        assembly.add_node(
            pivot_path, "Couplers", (0.0, recipe_module.COUPLER_HEIGHT, pivot_z), pivot_quat)
        assembly.add_node(
            f"{pivot_path}/CouplerFace", pivot_path, (0.0, 0.0, face_z), IDENTITY_QUAT)

        for short_name, obj in coupler.build_meshes().items():
            assembly.add_mesh(obj, pivot_path, short_name)


def _build_markers(assembly, recipe):
    assembly.add_node("Markers", "", (0.0, 0.0, 0.0), IDENTITY_QUAT)
    assembly.add_node(
        "Markers/BogieTrack__front", "Markers",
        (0.0, 0.0, recipe.bogie_centre_offset), IDENTITY_QUAT)
    assembly.add_node(
        "Markers/BogieTrack__rear", "Markers",
        (0.0, 0.0, -recipe.bogie_centre_offset), IDENTITY_QUAT)


def build_vehicle(recipe):
    """Composes all part builders into one Assembly for ``recipe``.

    Calls the body builder matching ``recipe.kind``, two bogie frames, four
    wheelsets, and two couplers, renaming every returned mesh object to its
    contract full-path name and recording the full node table.
    """
    assembly = Assembly(recipe)
    _build_visuals(assembly, recipe)
    _build_running_gear(assembly, recipe)
    _build_couplers(assembly, recipe)
    _build_markers(assembly, recipe)
    assembly.add_node("Interaction", "", (0.0, 0.0, 0.0), IDENTITY_QUAT)
    return assembly
