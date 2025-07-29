import os
import pytest
from thema.thema import Thema


def test_init_with_valid_yaml(valid_yaml_file):
    t = Thema(valid_yaml_file)
    assert t.YAML_PATH == valid_yaml_file
    assert hasattr(t, "params")
    assert t.params.outDir is not None
    assert t.params.runName is not None


def test_genesis_workflow_success(valid_yaml_file, monkeypatch):
    t = Thema(valid_yaml_file)
    # Patch all sub-methods to just record calls
    called = []
    monkeypatch.setattr(
        t,
        "spaghettify_innerSystem",
        lambda: called.append("spaghettify_innerSystem"),
    )
    monkeypatch.setattr(
        t, "innerSystem_genesis", lambda: called.append("innerSystem_genesis")
    )
    monkeypatch.setattr(
        t,
        "spaghettify_outerSystem",
        lambda: called.append("spaghettify_outerSystem"),
    )
    monkeypatch.setattr(
        t, "outerSystem_genesis", lambda: called.append("outerSystem_genesis")
    )
    monkeypatch.setattr(
        t, "spaghettify_galaxy", lambda: called.append("spaghettify_galaxy")
    )
    monkeypatch.setattr(
        t, "galaxy_genesis", lambda: called.append("galaxy_genesis")
    )
    t.genesis()
    assert called == [
        "spaghettify_innerSystem",
        "innerSystem_genesis",
        "spaghettify_outerSystem",
        "outerSystem_genesis",
        "spaghettify_galaxy",
        "galaxy_genesis",
    ]


def test_spaghettify_cleans_directories(valid_yaml_file):
    t = Thema(valid_yaml_file)
    # Setup directory structure
    out_dir = os.path.join(t.params.outDir, t.params.runName)
    clean_dir = os.path.join(out_dir, "clean")
    proj_dir = os.path.join(out_dir, "projections")
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(clean_dir)
    os.makedirs(proj_dir)
    os.makedirs(model_dir)
    # Place dummy files
    for d in [clean_dir, proj_dir, model_dir]:
        with open(os.path.join(d, "dummy.txt"), "w") as f:
            f.write("test")
    t.spaghettify()
    # All directories should be removed
    assert not os.path.exists(clean_dir)
    assert not os.path.exists(proj_dir)
    assert not os.path.exists(model_dir)
    assert not os.path.exists(out_dir)


def test_init_with_invalid_yaml_path(tmp_path):
    # Non-existent file
    invalid_path = str(tmp_path / "doesnotexist.yaml")
    with pytest.raises(ValueError):
        Thema(invalid_path)
    # Not a yaml file
    not_yaml = tmp_path / "params.txt"
    not_yaml.write_text("outDir: foo\nrunName: bar")
    with pytest.raises(ValueError):
        Thema(str(not_yaml))


def test_innerSystem_genesis_nonempty_directory(valid_yaml_file):
    t = Thema(valid_yaml_file)
    clean_outdir = os.path.join(t.params.outDir, t.params.runName + "/clean")
    os.makedirs(clean_outdir)
    # Add a dummy file to make directory non-empty
    with open(os.path.join(clean_outdir, "dummy.txt"), "w") as f:
        f.write("test")
    with pytest.raises(AssertionError):
        t.innerSystem_genesis()


def test_spaghettify_innerSystem_removal_failure(valid_yaml_file, monkeypatch):
    t = Thema(valid_yaml_file)
    clean_outdir = os.path.join(t.params.outDir, t.params.runName + "/clean")
    os.makedirs(clean_outdir)
    file_path = os.path.join(clean_outdir, "dummy.txt")
    with open(file_path, "w") as f:
        f.write("test")

    # Patch os.remove to simulate a failure
    def mock_remove(path):
        if path == file_path:
            raise PermissionError(f"Permission denied: '{path}'")
        else:
            os.remove(path)

    monkeypatch.setattr("os.remove", mock_remove)

    # Patch print to capture output
    output = []
    monkeypatch.setattr("builtins.print", lambda msg: output.append(msg))
    t.spaghettify_innerSystem()

    # Should have printed an error message
    assert any("Error while deleting" in msg for msg in output)
