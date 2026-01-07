# Archive Management

## Managing Archives

The archive module provides CLI utilities:

```powershell
# List all archived runs
python scripts/archive.py list

# Show details of a specific run
python scripts/archive.py show <run_id_or_uuid>

# Show the latest run
python scripts/archive.py latest

# Cleanup old runs (keep last 10)
python scripts/archive.py cleanup --keep 10

# Export a run as ZIP archive
python scripts/archive.py export <run_id_or_uuid>

# Export a run as TAR archive with gzip compression
python scripts/archive.py export <run_id_or_uuid> --format tar

# Export latest run as ZIP
python scripts/archive.py export-latest

# Export with custom output path
python scripts/archive.py export <run_id_or_uuid> --output /path/to/archive.zip

# Export as TAR with different compression
python scripts/archive.py export <run_id_or_uuid> --format tar --compression bz2
```

## Export Options

### ZIP Export
Export runs as ZIP archives for easy sharing and backup:

```powershell
# Export specific run as ZIP (default format)
python scripts/archive.py export 2025-12-16_12-12-01_e76b4d00

# Export with custom output path
python scripts/archive.py export 2025-12-16_12-12-01_e76b4d00 --output /path/to/my-archive.zip

# Export latest run
python scripts/archive.py export-latest
```

### TAR Export
Export runs as TAR archives with optional compression:

```powershell
# Export as TAR with gzip compression (default)
python scripts/archive.py export 2025-12-16_12-12-01_e76b4d00 --format tar

# Export as TAR with bzip2 compression
python scripts/archive.py export 2025-12-16_12-12-01_e76b4d00 --format tar --compression bz2

# Export as TAR with xz compression
python scripts/archive.py export 2025-12-16_12-12-01_e76b4d00 --format tar --compression xz

# Export as uncompressed TAR
python scripts/archive.py export 2025-12-16_12-12-01_e76b4d00 --format tar --compression ""
```

### Compression Options

- **ZIP**: Uses Python's `zipfile.ZIP_DEFLATED` by default (good balance of speed and compression)
- **TAR.GZ**: Gzip compression (fast, good compression)
- **TAR.BZ2**: Bzip2 compression (slower, better compression)
- **TAR.XZ**: LZMA compression (slowest, best compression)
- **TAR**: No compression (fastest, largest file size)

### Export Location

By default, exported archives are saved in the `exports/` directory with the run ID as the filename:
- ZIP: `exports/{run_id}.zip`
- TAR: `exports/{run_id}.tar.gz` (or `.tar.bz2`, `.tar.xz`, `.tar`)

You can specify a custom output path using the `--output` option.

## Archiving Configuration

Archive settings can be configured in `scripts/config.py`:
- `ARCHIVE_ENABLED`: Enable/disable archiving (default: `True`)
- `ARCHIVE_DIR`: Base directory for archived runs (default: `"../exports/runs"`)

To disable archiving and use legacy output structure, set `ARCHIVE_ENABLED = False` in `config.py`.