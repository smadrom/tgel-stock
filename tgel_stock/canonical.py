import hashlib


def _q(value: float) -> int:
    quantized = round(value * 100000.0)
    return 0 if quantized == 0 else int(quantized)  # normalizes -0.0


def geometry_hash(positions, normals, uvs, triangles) -> str:
    hasher = hashlib.sha256()
    for label, rows, width in (
        (b"P", positions, 3), (b"N", normals, 3), (b"U", uvs, 2)):
        hasher.update(label)
        hasher.update(len(rows).to_bytes(8, "little"))
        for row in rows:
            for i in range(width):
                hasher.update(_q(row[i]).to_bytes(8, "little", signed=True))
    hasher.update(b"T")
    hasher.update(len(triangles).to_bytes(8, "little"))
    for tri in triangles:
        for index in tri:
            hasher.update(int(index).to_bytes(8, "little", signed=True))
    return hasher.hexdigest()


def file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
