import importlib
import os
import pathlib

from infiniop.ninetoothed.build import BUILD_DIRECTORY_PATH

CURRENT_FILE_PATH = pathlib.Path(__file__)

SRC_DIR_PATH = CURRENT_FILE_PATH.parent.parent / "src"


def _find_and_build_ops():
    ops_path = SRC_DIR_PATH / "infiniop" / "ops"
    requested = os.environ.get("INFINIOP_NINETOOTHED_OPS", "").strip()
    selected = {
        name.strip() for name in requested.split(",") if name.strip()
    }

    if selected:
        available = {path.name for path in ops_path.iterdir() if path.is_dir()}
        unknown = selected - available
        if unknown:
            raise ValueError(
                "Unknown NineToothed operators: " + ", ".join(sorted(unknown))
            )

    # Each operator already parallelizes its specialization grid.  Building
    # operators sequentially avoids nested process pools exhausting CPU and
    # memory on clean build hosts.
    for op_dir in sorted(ops_path.iterdir()):
        if selected and op_dir.name not in selected:
            continue
        ninetoothed_path = op_dir / "ninetoothed"
        if not ninetoothed_path.is_dir():
            continue
        build_file = ninetoothed_path / "build.py"
        if build_file.exists():
            _build(ninetoothed_path)


def _build(ninetoothed_path):
    module_path = ninetoothed_path / "build"
    relative_path = module_path.relative_to(SRC_DIR_PATH)
    import_name = ".".join(relative_path.parts)
    module = importlib.import_module(import_name)

    module.build()


if __name__ == "__main__":
    BUILD_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)

    _find_and_build_ops()
