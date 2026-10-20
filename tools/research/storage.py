"""Lossless, deduplicated JSONL collection. No filtering or training admission.

Exact record bytes are deduplicated, not measurements sharing a config or ID.
Every occurrence keeps its source/member/line. Build never deletes originals;
explicit archival retirement removes only independently verified copies.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zipfile


def retire_archive(source, archive_path):
    """Retire an inactive directory using an already verified archive."""
    source = Path(source).absolute()
    archive_path = Path(archive_path).absolute()
    repo = Path(__file__).resolve().parents[2]
    if source in (Path('/'),Path.home(),repo) or not source.is_dir():
        raise ValueError('expected a specific dataset directory')
    if source.is_symlink() or any(p.is_symlink() for p in source.parents):
        raise ValueError('symlink source')
    if source in archive_path.parents:
        raise ValueError('archive must be outside source')
    if archive_path.with_suffix(archive_path.suffix+'.retired.json').exists():
        raise ValueError('archive already has a retirement receipt')
    receipt = json.loads(archive_path.with_suffix(archive_path.suffix+'.receipt.json').read_text())
    if receipt['source'] != str(source) or receipt['archive'] != str(archive_path) or not receipt['verified']:
        raise ValueError('archive receipt identity mismatch')
    with archive_path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest() != receipt['archive_sha256']:
            raise ValueError('archive hash mismatch')
    paths = sorted(source.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('symlink source member')
    files = [p for p in paths if p.is_file()]
    expected = receipt['file_hashes']
    if {str(p.relative_to(source.parent)) for p in files} != set(expected):
        raise ValueError('source membership changed')
    fingerprints = {}
    with zipfile.ZipFile(archive_path) as archive:
        for path in files:
            name = str(path.relative_to(source.parent))
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream,'sha256').hexdigest()
            with archive.open(name) as stream:
                archived = hashlib.file_digest(stream,'sha256').hexdigest()
            if digest != expected[name] or archived != digest:
                raise ValueError('source/archive content mismatch')
            stat = path.stat()
            fingerprints[path] = (stat.st_ino,stat.st_size,stat.st_mtime_ns)
    if sorted(source.rglob('*')) != paths:
        raise ValueError('source changed during verification')
    for path, expected_stat in fingerprints.items():
        stat=path.stat()
        if path.is_symlink() or (stat.st_ino,stat.st_size,stat.st_mtime_ns) != expected_stat:
            raise ValueError('source changed during verification')
    for path in files:
        path.unlink()
    for path in sorted((p for p in paths if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        path.rmdir()
    source.rmdir()
    with archive_path.with_suffix(archive_path.suffix+'.retired.json').open('x') as stream:
        json.dump(dict(source=str(source),removed_files=len(files),archive=str(archive_path)),stream)
    print(f'RETIRED verified_files={len(files)} recover_from={archive_path}',flush=True)


def archive_tree(source, archive_path, *, retire=False):
    """Pack an inactive tree; optionally remove only byte-verified originals.

    No symlinks, overwrite, workspace-root targets or nested archive output.
    Archive is independently re-read, then the full source is checked again
    before any unlink. On failure originals remain; archive is diagnostic.
    """
    source = Path(source).absolute()
    archive_path = Path(archive_path).absolute()
    repo = Path(__file__).resolve().parents[2]
    if source in (Path('/'),Path.home(),repo) or source.is_symlink() or not source.is_dir():
        raise ValueError('expected a specific non-symlink dataset directory')
    if any(p.is_symlink() for p in source.parents):
        raise ValueError('symlink source ancestor')
    if source == archive_path or source in archive_path.parents:
        raise ValueError('archive must be outside source')
    if any(archive_path.with_suffix(archive_path.suffix+suffix).exists()
           for suffix in ('.receipt.json', '.retired.json')):
        raise ValueError('archive receipts already exist')
    paths = sorted(source.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('archive refuses symlinks')
    files = [p for p in paths if p.is_file()]
    if not files:
        raise ValueError('refusing to retire an empty tree')
    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream,'sha256').hexdigest()
    hashes = {}
    archive_path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for i,path in enumerate(files):
            name = str(path.relative_to(source.parent))
            hashes[name] = digest(path)
            archive.write(path,name)
            if i % 5000 == 0:
                print(f'ARCHIVE files={i+1}/{len(files)}',flush=True)
        for path in paths:
            if path.is_dir():
                archive.write(path,str(path.relative_to(source.parent))+'/')
    with archive_path.open('rb') as stream:
        os.fsync(stream.fileno())
    with zipfile.ZipFile(archive_path) as archive:
        for name, expected in hashes.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest() != expected:
                    raise ValueError(f'archive verification failed: {name}')
    current = sorted(source.rglob('*'))
    if current != paths or any(p.is_symlink() for p in current):
        raise ValueError('source tree changed; refusing retirement')
    for path in files:
        if digest(path) != hashes[str(path.relative_to(source.parent))]:
            raise ValueError('source contents changed; refusing retirement')
    receipt = dict(source=str(source), archive=str(archive_path), file_count=len(files),
                   file_hashes=hashes, archive_sha256=digest(archive_path), verified=True,
                   retired=False, restore='Extract this ZIP into the original source parent to restore the exact paths.')
    receipt_path = archive_path.with_suffix(archive_path.suffix+'.receipt.json')
    with receipt_path.open('x') as stream:
        json.dump(receipt,stream,indent=2)
        stream.flush();os.fsync(stream.fileno())
    if retire:
        # Targets are explicit files under the validated tree, never a wildcard
        # or recursive shell deletion. Unexpected added files prevent rmdir.
        for path in files:
            path.unlink()
        for path in sorted((p for p in paths if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
            path.rmdir()
        source.rmdir()
        with archive_path.with_suffix(archive_path.suffix+'.retired.json').open('x') as stream:
            json.dump(dict(source=str(source),removed_files=len(files),archive=str(archive_path)),stream)
        receipt['retired']=True
    print(f'ARCHIVE VERIFIED files={len(files)} retired={retire} path={archive_path}',flush=True)
    return receipt


def journals(sources):
    seen = set()
    for value in sources:
        source = Path(value).resolve()
        if source.is_file() and zipfile.is_zipfile(source):
            before = source.stat()
            with zipfile.ZipFile(source) as archive:
                for member in sorted(archive.infolist(), key=lambda m: m.filename):
                    if member.is_dir() or not member.filename.endswith('.jsonl'):
                        continue
                    key = (str(source), member.filename, member.header_offset)
                    if key in seen:
                        continue
                    seen.add(key)
                    with archive.open(member) as stream:
                        yield key, stream
            after = source.stat()
            if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                raise ValueError(f'source changed during ingestion: {source}')
        elif source.is_dir():
            for path in sorted(source.rglob('*.jsonl')):
                if path.is_symlink() or any(p.is_symlink() for p in path.parents):
                    raise ValueError(f'refusing symlink journal: {path}')
                key = (str(path.resolve()), '', 0)
                if key in seen:
                    continue
                seen.add(key)
                before = path.stat()
                with path.open('rb') as stream:
                    yield key, stream
                after = path.stat()
                if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                    raise ValueError(f'source changed during ingestion: {path}')
        else:
            raise ValueError(f'expected directory or ZIP: {source}')


def fields(raw):
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return ('non_object', '', '', '', '', '')
        outcome = obj.get('outcome')
        outcome = outcome if isinstance(outcome, dict) else {}
        metadata = outcome.get('metadata')
        metadata = metadata if isinstance(metadata, dict) else {}
        return ('json_object', str(obj.get('family', '')), str(obj.get('kernel_revision', '')),
                str(obj.get('context_id', '')), str(outcome.get('status', '')),
                str(metadata.get('measurement_mode', 'unknown')))
    except (ValueError, UnicodeError, TypeError):
        return ('malformed', '', '', '', '', '')


def export_table(db, output, table, names, types, shard_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    folder = output / table
    folder.mkdir()
    schema = pa.schema(list(zip(names, types)))
    cursor = db.execute(f'SELECT {",".join(names)} FROM {table} ORDER BY rowid')
    parts = []
    while rows := cursor.fetchmany(shard_rows):
        path = folder / f'part-{len(parts):05d}.parquet'
        data = pa.Table.from_arrays([pa.array(column, type=kind)
             for column, kind in zip(zip(*rows), types)], schema=schema)
        pq.write_table(data, path, compression='zstd')
        parts.append(dict(path=str(path.relative_to(output)), rows=len(rows),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return parts


def verify(output):
    import pyarrow.parquet as pq
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    totals = Counter()
    record_keys = set()
    source_rows = Counter()
    sources = {source['source_id']: source for source in manifest['sources']}
    for table in ('records', 'occurrences'):
        parts = manifest['tables'][table]
        for part in parts:
            path = output/part['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != part['sha256']:
                raise ValueError(f'corrupt shard: {path}')
            data = pq.read_table(path)
            if data.num_rows != part['rows']:
                raise ValueError(f'row count mismatch: {path}')
            if table == 'records':
                for key, raw in zip(data['record_hash'].to_pylist(), data['payload'].to_pylist()):
                    if hashlib.sha256(raw).hexdigest() != key:
                        raise ValueError('record roundtrip mismatch')
                    if key in record_keys:
                        raise ValueError('duplicate record key')
                    record_keys.add(key)
            else:
                for key, source_id, line in zip(data['record_hash'].to_pylist(),
                                               data['source_id'].to_pylist(),
                                               data['line'].to_pylist()):
                    if key not in record_keys or source_id not in sources or line < 1:
                        raise ValueError('invalid occurrence reference')
                    source_rows[source_id] += 1
            totals[table] += data.num_rows
    if totals['records'] != manifest['unique_records'] or totals['occurrences'] != manifest['occurrences']:
        raise ValueError('collection row accounting mismatch')
    if any(source_rows[key] != source['rows'] for key, source in sources.items()):
        raise ValueError('source occurrence accounting mismatch')
    return dict(verified=True, **totals)


def build(sources, output, shard_rows=50000):
    import pyarrow as pa
    if shard_rows < 1:
        raise ValueError('shard_rows must be positive')
    output = Path(output).resolve()
    for value in sources:
        source = Path(value).resolve()
        if source == output or source in output.parents:
            raise ValueError('output must not be inside an input tree')
    output.mkdir(parents=True, exist_ok=False)
    database = output/'building.sqlite'
    db = sqlite3.connect(database)
    db.execute('CREATE TABLE records(record_hash TEXT PRIMARY KEY,payload BLOB,parse_status TEXT,family TEXT,revision TEXT,context_id TEXT,outcome_status TEXT,measurement_mode TEXT)')
    db.execute('CREATE TABLE occurrences(record_hash TEXT,source_id INTEGER,line INTEGER)')
    counts = Counter()
    inventory = []
    try:
        for key, stream in journals(sources):
            source_id = len(inventory)
            digest = hashlib.sha256()
            rows = blanks = 0
            for line, raw in enumerate(stream, 1):
                digest.update(raw)
                if not raw.strip():
                    blanks += 1
                    continue
                record_hash = hashlib.sha256(raw).hexdigest()
                parsed = fields(raw)
                db.execute('INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?,?,?)',
                           (record_hash, raw, *parsed))
                db.execute('INSERT INTO occurrences VALUES (?,?,?)', (record_hash, source_id, line))
                rows += 1
                counts[parsed[0]] += 1
                if rows % 10000 == 0:
                    db.commit()
            inventory.append(dict(source_id=source_id, source=key[0], member=key[1],
                zip_offset=key[2], sha256=digest.hexdigest(), rows=rows, blank_lines=blanks))
            db.commit()
            if source_id % 100 == 0:
                print(f'INGEST journals={source_id+1} occurrences={sum(counts.values())}', flush=True)
        unique = db.execute('SELECT count(*) FROM records').fetchone()[0]
        occurrences = db.execute('SELECT count(*) FROM occurrences').fetchone()[0]
        tables = {
            'records': export_table(db, output, 'records',
                ['record_hash','payload','parse_status','family','revision','context_id','outcome_status','measurement_mode'],
                [pa.string(),pa.binary(),*[pa.string()]*6], shard_rows),
            'occurrences': export_table(db, output, 'occurrences',
                ['record_hash','source_id','line'], [pa.string(),pa.int64(),pa.int64()], shard_rows),
        }
        manifest = dict(schema=1, unique_records=unique, occurrences=occurrences,
            duplicate_occurrences=occurrences-unique, parse_counts=dict(counts), sources=inventory,
            tables=tables, training_admission=False, runtime_promotion=False,
            semantics='Exact-byte deduplication only; all occurrences, failures, raw metadata, conflicts and malformed records retained. No averaging or inferred split assignment.',
            scope='JSONL journals only. Original ZIPs, configs, model artifacts and dataset sidecars remain authoritative and untouched.')
        with (output/'manifest.json').open('x') as stream:
            json.dump(manifest, stream, indent=2)
        result = verify(output)
        with (output/'verification.json').open('x') as stream:
            json.dump(result, stream, indent=2)
    finally:
        db.close()
    # Only the temporary database produced by this successful build is removed.
    database.unlink()
    print(json.dumps(result), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build','verify','archive','retire'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--sources', nargs='+')
    parser.add_argument('--shard-rows', type=int, default=50000)
    parser.add_argument('--retire', action='store_true',help='Archive action: remove originals only after full byte verification; restore from ZIP')
    args = parser.parse_args()
    if args.retire and args.action != 'archive':
        parser.error('--retire applies only to archive')
    if args.action in ('archive','retire'):
        if not args.sources or len(args.sources)!=1:
            parser.error('archive requires exactly one source directory')
        if args.action == 'retire':
            retire_archive(args.sources[0],args.output)
        else:
            archive_tree(args.sources[0],args.output,retire=args.retire)
    elif args.action == 'build':
        if not args.sources:
            parser.error('build requires --sources')
        build(args.sources,args.output,args.shard_rows)
    else:
        print(json.dumps(verify(args.output)))


if __name__ == '__main__':
    main()
