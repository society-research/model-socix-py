import os


def Settings(**kwargs):
    venv = os.getenv("VIRTUAL_ENV")
    if not venv:
        raise RuntimeError("should source some python virtual environment!")
    return {"interpreter_path": os.path.join(venv, "bin/python")}
