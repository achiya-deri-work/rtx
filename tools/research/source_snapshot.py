"""Reproducible source-only snapshots; verify without extracting or executing."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

SOURCE_DIRS=('rtx','tools','tests')
METADATA=('setup.py','pyproject.toml','requirements.txt','requirements-dev.txt')
MANIFEST='snapshot_manifest.json'
MAX_FILE=16*1024*1024
MAX_TOTAL=128*1024*1024


def inventory(root):
    root=Path(root).resolve()
    files=[]
    for directory in SOURCE_DIRS:
        parent=root/directory
        if parent.is_symlink():raise ValueError('source directories cannot be symlinks')
        if parent.exists():files.extend(parent.rglob('*.py'))
    files.extend(root/name for name in METADATA if (root/name).exists())
    result={}
    for path in sorted(files):
        relative=path.relative_to(root)
        if any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)):
            raise ValueError('source snapshot refuses symlink paths')
        if not path.is_file():continue
        if path.stat().st_size>MAX_FILE:raise ValueError('source file too large')
        result[relative.as_posix()]=path.read_bytes()
    if sum(map(len,result.values()))>MAX_TOTAL:raise ValueError('source snapshot too large')
    if not result:raise ValueError('empty source inventory')
    return result


def source_manifest(files):
    return dict(schema=1,files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                               for name,data in sorted(files.items())},
                scope=dict(source_dirs=SOURCE_DIRS,metadata=METADATA),
                contains_datasets=False,contains_git_history=False)


def write_snapshot(root, output):
    files=inventory(root)
    manifest=source_manifest(files)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents overwriting historical evidence. No
    # timestamps or absolute workspace paths enter the archive content.
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in [*sorted(files.items()),(MANIFEST,json.dumps(manifest,sort_keys=True).encode())]:
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            archive.writestr(info,data)
    return verify_snapshot(output,root=root)


def verify_snapshot(path, *, root=None):
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        members=archive.infolist()
        names=[info.filename for info in members]
        if len(names)!=len(set(names)) or MANIFEST not in names:
            raise ValueError('duplicate members or missing snapshot manifest')
        if sum(info.file_size for info in members)>MAX_TOTAL+MAX_FILE:
            raise ValueError('snapshot exceeds size limit')
        for info in members:
            name=PurePosixPath(info.filename)
            if (name.is_absolute() or '..' in name.parts or '\\' in info.filename or
                    info.file_size>MAX_FILE or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('unsafe snapshot member')
        manifest=json.loads(archive.read(MANIFEST))
        if manifest.get('schema')!=1:raise ValueError('unsupported snapshot schema')
        if set(names)!={MANIFEST,*manifest['files']}:
            raise ValueError('snapshot inventory mismatch')
        files={name:archive.read(name) for name in manifest['files']}
        for name in files:
            parts=PurePosixPath(name).parts
            if name not in METADATA and not (parts[0] in SOURCE_DIRS and name.endswith('.py')):
                raise ValueError('member outside source allowlist')
        if source_manifest(files)!=manifest:
            # JSON round-trips tuples to lists; normalize that representation.
            if json.loads(json.dumps(source_manifest(files)))!=manifest:
                raise ValueError('snapshot content digest mismatch')
    if root is not None and inventory(root)!=files:
        raise ValueError('workspace sources differ from snapshot')
    return dict(schema=1,path=path.name,archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                files=len(files),source_bytes=sum(map(len,files.values())),verified=True)
