import json
import math

from . import canonical


class Manifest:
    def __init__(self):
        self._meta = {}
        self._nodes = []
        self._meshes = []
        self._textures = []

    def set_meta(
            self, model_id, recipe_digest, script_digest, blender_version, *,
            kind, atlas_resolution, length_over_couplers, body_length, width,
            height, track_gauge, wheel_back_to_back, wheel_width, wheel_radius,
            coupler_height, coupler_pivot_to_face, bogie_centre_offset,
            bogie_wheelbase, bogie_pivot_height):
        """Set the complete frozen production contract for this model.

        Measurements are required keyword-only arguments so a production build
        cannot silently emit an incomplete manifest.  Rounding here, before
        JSON serialization, makes the metadata use the same six-decimal
        precision as node transforms and bounds.
        """
        def metric(value, name):
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Manifest meta {name} must be finite")
            rounded = round(value, 6)
            return 0.0 if rounded == 0.0 else rounded

        try:
            raw_resolution = list(atlas_resolution)
        except TypeError as exc:
            raise TypeError(
                "Manifest meta atlasResolution must be an iterable") from exc
        if len(raw_resolution) != 2:
            raise ValueError(
                "Manifest meta atlasResolution must contain two positive pixels")
        resolution = []
        for value in raw_resolution:
            if isinstance(value, bool):
                raise ValueError(
                    "Manifest meta atlasResolution pixels must be integers")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Manifest meta atlasResolution pixels must be integers") from exc
            if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
                raise ValueError(
                    "Manifest meta atlasResolution pixels must be positive integers")
            resolution.append(int(numeric))

        self._meta = {
            "modelId": model_id,
            "kind": kind,
            "atlasResolution": resolution,
            "lengthOverCouplers": metric(
                length_over_couplers, "lengthOverCouplers"),
            "bodyLength": metric(body_length, "bodyLength"),
            "width": metric(width, "width"),
            "height": metric(height, "height"),
            "trackGauge": metric(track_gauge, "trackGauge"),
            "wheelBackToBack": metric(
                wheel_back_to_back, "wheelBackToBack"),
            "wheelWidth": metric(wheel_width, "wheelWidth"),
            "wheelRadius": metric(wheel_radius, "wheelRadius"),
            "couplerHeight": metric(coupler_height, "couplerHeight"),
            "couplerPivotToFace": metric(
                coupler_pivot_to_face, "couplerPivotToFace"),
            "bogieCentreOffset": metric(
                bogie_centre_offset, "bogieCentreOffset"),
            "bogieWheelbase": metric(bogie_wheelbase, "bogieWheelbase"),
            "bogiePivotHeight": metric(
                bogie_pivot_height, "bogiePivotHeight"),
            "recipeDigest": recipe_digest,
            "scriptDigest": script_digest,
            "blenderVersion": blender_version,
            "schema": "tgel.rollingstock.manifest.v2",
        }

    def add_node(self, path, parent, local_position, local_rotation_quat):
        self._nodes.append({
            "path": path,
            "parent": parent,
            "localPosition": [round(float(c), 6) for c in local_position],
            "localRotation": [round(float(c), 6) for c in local_rotation_quat],  # x,y,z,w
        })

    def add_mesh(self, name, node_path, positions, normals, uvs, triangles,
                 bounds_min, bounds_max):
        self._meshes.append({
            "name": name,
            "node": node_path,
            "vertexCount": len(positions),
            "triangleCount": len(triangles),
            "boundsMin": [round(float(c), 6) for c in bounds_min],
            "boundsMax": [round(float(c), 6) for c in bounds_max],
            "semanticHash": canonical.geometry_hash(positions, normals, uvs, triangles),
        })

    def add_texture(self, name, file_path, resolution, color_space):
        self._textures.append({
            "name": name,
            "file": file_path.replace("\\", "/").split("/")[-1],
            "resolution": list(resolution),
            "colorSpace": color_space,
            "sha256": canonical.file_sha256(file_path),
        })

    def to_dict(self):
        return {
            "meta": self._meta,
            "nodes": sorted(self._nodes, key=lambda n: n["path"]),
            "meshes": sorted(self._meshes, key=lambda m: m["name"]),
            "textures": sorted(self._textures, key=lambda t: t["name"]),
        }

    def write(self, path):
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
