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
```

## Archiving Configuration

Archive settings can be configured in `scripts/config.py`:
- `ARCHIVE_ENABLED`: Enable/disable archiving (default: `True`)
- `ARCHIVE_DIR`: Base directory for archived runs (default: `"../exports/runs"`)

To disable archiving and use legacy output structure, set `ARCHIVE_ENABLED = False` in `config.py`.