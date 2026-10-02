"""Check that the wheel contains standalone report and schema resources."""
from pathlib import Path
import sys
from zipfile import ZipFile

wheels = list(Path(sys.argv[1]).glob('*.whl'))
if len(wheels) != 1:
    raise SystemExit('Expected exactly one wheel')
required = {
    'admet_benchmark/report_assets/report.html',
    'admet_benchmark/report_assets/report.css',
    'admet_benchmark/report_assets/report.js',
    'admet_benchmark/report_assets/workspace.js',
    'admet_benchmark/report_assets/scientific.js',
    'admet_benchmark/report_assets/delivery.js',
    'admet_benchmark/schemas/study.schema.json',
    'admet_benchmark/schemas/evaluation-config.schema.json',
}
with ZipFile(wheels[0]) as archive:
    missing = required.difference(archive.namelist())
if missing:
    raise SystemExit(f'Missing package resources: {sorted(missing)}')
print('Wheel contains all report assets and schemas.')
