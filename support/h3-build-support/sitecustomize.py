import os, shutil, errno
_original_replace = os.replace
def _replace(src, dst, *args, **kwargs):
    try:
        return _original_replace(src, dst, *args, **kwargs)
    except OSError as exc:
        if exc.errno != errno.EXDEV or args or kwargs:
            raise
        shutil.copyfile(src, dst)
        os.unlink(src)
os.replace = _replace
