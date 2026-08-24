"""Low-level publication handles with Windows share-deny semantics."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def _is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & 0x00000400
    )


def _windows_descriptor(
    path: Path,
    *,
    access: int,
    creation_disposition: int,
    descriptor_flags: int,
    open_reparse_point: bool,
) -> int:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    flags_and_attributes = 0x80  # FILE_ATTRIBUTE_NORMAL
    if open_reparse_point:
        flags_and_attributes |= 0x00200000  # FILE_FLAG_OPEN_REPARSE_POINT
    handle = create_file(
        str(path),
        access,
        0x00000001,  # FILE_SHARE_READ; deny WRITE and DELETE until close
        None,
        creation_disposition,
        flags_and_attributes,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        error = ctypes.get_last_error()
        if error in (80, 183):  # ERROR_FILE_EXISTS / ERROR_ALREADY_EXISTS
            raise FileExistsError(17, f"path already exists: {path}", str(path))
        raise ctypes.WinError(error)
    try:
        return msvcrt.open_osfhandle(
            int(handle),
            descriptor_flags | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0),
        )
    except BaseException:
        close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        close_handle(handle)
        raise


def open_exclusive_publication(path: Path) -> int:
    """Create a new binary file while denying concurrent Windows writers."""
    if os.name == "nt":
        return _windows_descriptor(
            path,
            access=0x80000000 | 0x40000000,  # GENERIC_READ | GENERIC_WRITE
            creation_disposition=1,  # CREATE_NEW
            descriptor_flags=os.O_RDWR,
            open_reparse_point=False,
        )
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    return os.open(path, flags, 0o444)


def open_verified_publication(path: Path) -> int:
    """Open a published file while denying Windows writers and deletion."""
    if os.name == "nt":
        return _windows_descriptor(
            path,
            access=0x80000000 | 0x00000100,  # GENERIC_READ | FILE_WRITE_ATTRIBUTES
            creation_disposition=3,  # OPEN_EXISTING
            descriptor_flags=os.O_RDONLY,
            open_reparse_point=True,
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def close_publication_descriptor(descriptor: int, *, suppress_errors: bool) -> None:
    """Close an FD without replacing an exception already being propagated."""
    try:
        os.close(descriptor)
    except OSError:
        if not suppress_errors:
            raise


def set_publication_descriptor_read_only(descriptor: int) -> None:
    """Remove write access through an existing descriptor without data fsync."""
    if hasattr(os, "fchmod"):
        current_mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
        os.fchmod(descriptor, current_mode & ~0o222)
        return
    if os.name == "nt":  # pragma: win32 cover - executed in Windows CI
        import ctypes
        import msvcrt

        class _FileBasicInfo(ctypes.Structure):
            _fields_ = (
                ("creation_time", ctypes.c_longlong),
                ("last_access_time", ctypes.c_longlong),
                ("last_write_time", ctypes.c_longlong),
                ("change_time", ctypes.c_longlong),
                ("file_attributes", ctypes.c_uint32),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = ctypes.c_void_p(msvcrt.get_osfhandle(descriptor))
        basic = _FileBasicInfo()
        if not kernel32.GetFileInformationByHandleEx(
            handle, 0, ctypes.byref(basic), ctypes.sizeof(basic)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        basic.file_attributes |= 0x1  # FILE_ATTRIBUTE_READONLY
        if not kernel32.SetFileInformationByHandle(
            handle, 0, ctypes.byref(basic), ctypes.sizeof(basic)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return
    raise OSError("descriptor-safe read-only sealing is unavailable")


def ensure_real_directory(path: Path, *, error_type: type[Exception], role: str) -> None:
    """Create a directory tree without following a pre-existing linked ancestor."""
    absolute = Path(os.path.abspath(path))
    missing: list[Path] = []
    current = absolute
    while not os.path.lexists(current):
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise error_type(f"cannot find an existing ancestor for {role}: {path}")
        current = parent
    try:
        resolved = current.resolve(strict=True)
        info = os.lstat(current)
    except (OSError, RuntimeError) as exc:
        raise error_type(f"cannot inspect {role} ancestor {current}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode) or _is_link_or_reparse(info):
        raise error_type(f"{role} ancestor must be a real directory: {current}")
    if os.path.normcase(str(resolved)) != os.path.normcase(str(current.absolute())):
        raise error_type(f"{role} ancestor must not traverse a link: {current}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except OSError as exc:
            raise error_type(f"cannot create {role} {directory}: {exc}") from exc
        try:
            created = os.lstat(directory)
            resolved = directory.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise error_type(f"cannot verify created {role} {directory}: {exc}") from exc
        if not stat.S_ISDIR(created.st_mode) or _is_link_or_reparse(created):
            raise error_type(f"created {role} is not a real directory: {directory}")
        if os.path.normcase(str(resolved)) != os.path.normcase(str(directory.absolute())):
            raise error_type(f"created {role} traverses a link: {directory}")
