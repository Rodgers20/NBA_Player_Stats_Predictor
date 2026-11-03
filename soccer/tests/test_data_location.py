import os
import subprocess
import sys
from pathlib import Path

import pytest

from futball import config


def test_missing_drive_does_not_create_local_replacement(tmp_path, monkeypatch):
    missing = tmp_path / 'disconnected-drive' / 'FUTBALL-data'
    monkeypatch.setattr(config, 'DATA_ROOT', missing)
    with pytest.raises(config.VolumeNotMounted):
        config.ensure_dirs()
    assert not missing.exists()


def test_dotenv_loads_from_project_and_respects_environment(tmp_path):
    project = tmp_path / 'project'
    package = project / 'futball'
    package.mkdir(parents=True)
    (package / 'config.py').write_text(Path(config.__file__).read_text())
    (project / '.env').write_text('FUTBALL_DATA_ROOT=/example/external-data\n')
    env = dict(os.environ)
    env.pop('FUTBALL_DATA_ROOT', None)
    command = [sys.executable, '-c',
               f'import runpy; print(runpy.run_path({str(package / "config.py")!r})["DATA_ROOT"])']
    result = subprocess.check_output(command, cwd=tmp_path, env=env, text=True)
    assert result.strip() == '/example/external-data'
    env['FUTBALL_DATA_ROOT'] = '/explicit/override'
    result = subprocess.check_output(command, cwd=tmp_path, env=env, text=True)
    assert result.strip() == '/explicit/override'
