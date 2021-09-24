import subprocess
import pytest
import pathlib
import os
def run_test_script(name):
    p=subprocess.Popen(["python","{}".format(name)])
    (stdout, stderr) = p.communicate()
    assert p.returncode ==0, stderr

TARGET_SCRIPTS_FOLDER=pathlib.Path("./tests/scripts/")
assert TARGET_SCRIPTS_FOLDER.exists()
p = TARGET_SCRIPTS_FOLDER.glob('**/*')
files = [x for x in p if x.is_file()]

TARGET_SCRIPTS_FOLDER=pathlib.Path("./tests/other_scripts/")
assert TARGET_SCRIPTS_FOLDER.exists()
p = TARGET_SCRIPTS_FOLDER.glob('**/*')
files = files+[x for x in p if x.is_file()]

@pytest.mark.parametrize("name", files)
def test_script(name):
    print(name)
    run_test_script(name)