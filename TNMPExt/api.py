import os, sys, subprocess, platform, psutil, time, inspect, traceback
from threading import *
from multiprocessing import *
import threading as _threading
import multiprocessing as _multiprocessing
from typing import Union, TypeAlias, Callable, Optional, List, Dict

WINDOWS = 0x01
DARWIN = 0x02
LINUX = 0x03
UNDEFINED = 0x00
PROCESS_ALL_ACCESS = 0x1F0FFF
HIGH_PRIORITY_CLASS = 0x00000080
PROCESS_PID = os.getpid()
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
THREAD_SUSPEND_RESUME = 0x0002
THREAD_GET_CONTEXT = 0x0008
CONTEXT_FULL = 0x00010007
CONTEXT_ALL = 0x1003F
THREAD_ALL_ACCESS = 0x1F03FF
CREATE_SUSPENDED = 0x00000004
PAGE_READWRITE = 0x04
PAGE_EXECUTE_READWRITE = 0x40
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000

ForcedExit = os._exit

def IsCPython():
    return platform.python_implementation() == "CPython"

def Is64Bit():
    from struct import calcsize
    return calcsize("P") * 8 == 64

THREAD_OR_ID: TypeAlias = Union[_threading.Thread, int]

def failed(CODE=-1):
    '''not recommended to use alone'''
    frame = sys._getframe()
    with open("tnmpext_err.log", "w") as f:
        f.write(f"Exception at line: {frame.f_lineno}, Function: {frame.f_code.co_name}")
    ForcedExit(CODE)

def GetCurrentPlatform():
    '''The result is store in _platform'''
    platform_name = sys.platform
    if platform_name.startswith("win"):
        return WINDOWS
    elif platform_name == "darwin":
        return DARWIN
    elif platform_name.startswith("linux"):
        return LINUX
    else:
        return UNDEFINED

_platform = GetCurrentPlatform()

CPYTHON_SUPPORTED = IsCPython()

if _platform == WINDOWS and CPYTHON_SUPPORTED:
    import ctypes, nt
    from ctypes import wintypes
    ntdll = ctypes.windll.ntdll
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    SuspendThread = kernel32.SuspendThread
    ResumeThread = kernel32.ResumeThread
    GetThreadContext = kernel32.GetThreadContext
    SetThreadContext = kernel32.SetThreadContext
    Wow64GetThreadContext = kernel32.Wow64GetThreadContext
    Wow64SetThreadContext = kernel32.Wow64SetThreadContext
    if not Is64Bit():
        GetThreadContext = Wow64GetThreadContext
        SetThreadContext = Wow64SetThreadContext
    LPVOID = wintypes.LPVOID
    DWORD = wintypes.DWORD
    HANDLE = wintypes.HANDLE
    SIZE_T = ctypes.c_size_t
    class M128A(ctypes.Structure):
        _fields_ = [
        ("Low", ctypes.c_ulonglong),
        ("High", ctypes.c_ulonglong),
    ]
    class XMM_SAVE_AREA32(ctypes.Structure):
        _fields_ = [
        ("ControlWord", ctypes.c_ushort),
        ("StatusWord", ctypes.c_ushort),
        ("TagWord", ctypes.c_ubyte),
        ("Reserved1", ctypes.c_ubyte),
        ("ErrorOpcode", ctypes.c_ushort),
        ("ErrorOffset", ctypes.c_ulong),
        ("ErrorSelector", ctypes.c_ushort),
        ("Reserved2", ctypes.c_ushort),
        ("DataOffset", ctypes.c_ulong),
        ("DataSelector", ctypes.c_ushort),
        ("Reserved3", ctypes.c_ushort),
        ("MxCsr", ctypes.c_ulong),
        ("MxCsr_Mask", ctypes.c_ulong),
        ("FloatRegisters", M128A * 8),
        ("XmmRegisters", M128A * 16),
        ("Reserved4", ctypes.c_ubyte * 96),
    ]
    class DUMMYSTRUCTNAME(ctypes.Structure):
        _fields_ = [
        ("Header", M128A * 2),
        ("Legacy", M128A * 8),
        ("Xmm0", M128A),
        ("Xmm1", M128A),
        ("Xmm2", M128A),
        ("Xmm3", M128A),
        ("Xmm4", M128A),
        ("Xmm5", M128A),
        ("Xmm6", M128A),
        ("Xmm7", M128A),
        ("Xmm8", M128A),
        ("Xmm9", M128A),
        ("Xmm10", M128A),
        ("Xmm11", M128A),
        ("Xmm12", M128A),
        ("Xmm13", M128A),
        ("Xmm14", M128A),
        ("Xmm15", M128A),
    ]
    class DUMMYUNIONNAME(ctypes.Union):
        _fields_ = [
        ("FltSave", XMM_SAVE_AREA32),
        ("Q", M128A * 16),
        ("D", ctypes.c_ulonglong * 32),
        ("S", ctypes.c_uint * 32),
        ("DUMMYSTRUCTNAME", DUMMYSTRUCTNAME),
    ]
    class FLOATING_SAVE_AREA(ctypes.Structure):
        _fields_ = [
        ("ControlWord", wintypes.DWORD),
        ("StatusWord", wintypes.DWORD),
        ("TagWord", wintypes.DWORD),
        ("ErrorOffset", wintypes.DWORD),
        ("ErrorSelector", wintypes.DWORD),
        ("DataOffset", wintypes.DWORD),
        ("DataSelector", wintypes.DWORD),
        ("RegisterArea", ctypes.c_byte * 80),
        ("Cr0NpxState", wintypes.DWORD),
        ]
    class _CONTEXT32(ctypes.Structure):
        _fields_ = [
        ("ContextFlags", wintypes.DWORD),
        ("Dr0", wintypes.DWORD),
        ("Dr1", wintypes.DWORD),
        ("Dr2", wintypes.DWORD),
        ("Dr3", wintypes.DWORD),
        ("Dr6", wintypes.DWORD),
        ("Dr7", wintypes.DWORD),
        ("FloatSave", FLOATING_SAVE_AREA),
        ("SegGs", wintypes.DWORD),
        ("SegFs", wintypes.DWORD),
        ("SegEs", wintypes.DWORD),
        ("SegDs", wintypes.DWORD),
        ("Edi", wintypes.DWORD),
        ("Esi", wintypes.DWORD),
        ("Ebx", wintypes.DWORD),
        ("Edx", wintypes.DWORD),
        ("Ecx", wintypes.DWORD),
        ("Eax", wintypes.DWORD),
        ("Ebp", wintypes.DWORD),
        ("Eip", wintypes.DWORD),
        ("SegCs", wintypes.DWORD),
        ("EFlags", wintypes.DWORD),
        ("Esp", wintypes.DWORD),
        ("SegSs", wintypes.DWORD),
        ("ExtendedRegisters", ctypes.c_byte * 512),
        ]
    class CONTEXT(ctypes.Structure):
        _fields_ = [
        ("P1Home", ctypes.c_ulonglong),
        ("P2Home", ctypes.c_ulonglong),
        ("P3Home", ctypes.c_ulonglong),
        ("P4Home", ctypes.c_ulonglong),
        ("P5Home", ctypes.c_ulonglong),
        ("P6Home", ctypes.c_ulonglong),
        ("ContextFlags", ctypes.c_uint),
        ("MxCsr", ctypes.c_uint),
        ("SegCs", ctypes.c_ushort),
        ("SegDs", ctypes.c_ushort),
        ("SegEs", ctypes.c_ushort),
        ("SegFs", ctypes.c_ushort),
        ("SegGs", ctypes.c_ushort),
        ("SegSs", ctypes.c_ushort),
        ("EFlags", ctypes.c_uint),
        ("Dr0", ctypes.c_ulonglong),
        ("Dr1", ctypes.c_ulonglong),
        ("Dr2", ctypes.c_ulonglong),
        ("Dr3", ctypes.c_ulonglong),
        ("Dr6", ctypes.c_ulonglong),
        ("Dr7", ctypes.c_ulonglong),
        ("Rax", ctypes.c_ulonglong),
        ("Rcx", ctypes.c_ulonglong),
        ("Rdx", ctypes.c_ulonglong),
        ("Rbx", ctypes.c_ulonglong),
        ("Rsp", ctypes.c_ulonglong),
        ("Rbp", ctypes.c_ulonglong),
        ("Rsi", ctypes.c_ulonglong),
        ("Rdi", ctypes.c_ulonglong),
        ("R8", ctypes.c_ulonglong),
        ("R9", ctypes.c_ulonglong),
        ("R10", ctypes.c_ulonglong),
        ("R11", ctypes.c_ulonglong),
        ("R12", ctypes.c_ulonglong),
        ("R13", ctypes.c_ulonglong),
        ("R14", ctypes.c_ulonglong),
        ("R15", ctypes.c_ulonglong),
        ("Rip", ctypes.c_ulonglong),
        ("DUMMYUNIONNAME", DUMMYUNIONNAME),
        ("VectorRegister", M128A * 26),
        ("VectorControl", ctypes.c_ulonglong),
        ("DebugControl", ctypes.c_ulonglong),
        ("LastBranchToRip", ctypes.c_ulonglong),
        ("LastBranchFromRip", ctypes.c_ulonglong),
        ("LastExceptionToRip", ctypes.c_ulonglong),
        ("LastExceptionFromRip", ctypes.c_ulonglong),
            ]
        if not Is64Bit():
            CONTEXT = _CONTEXT32
else:
    pass

def SetHighPriority():
    try:
        if _platform == WINDOWS:
            handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, PROCESS_PID)
            kernel32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
        else:
            os.nice(-20)
    except Exception as e:
        failed()

def PreventSleep():
    try:
        if _platform == WINDOWS:
            kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
        elif _platform == DARWIN:
            subprocess.Popen(["caffeinate"])
        elif _platform == LINUX:
            subprocess.Popen(["systemd-inhibit", "--why=Prevent sleep", "sleep", "infinity"])
        else:
            failed()
    except:
        failed()

def _async_raise(tid, exctype):
    '''For ShutDownThread(). Use it instead'''
    if not CPYTHON_SUPPORTED:
        failed()
    if not isinstance(exctype, type):
        failed()
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        failed()
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        failed()
    try:
        frame = inspect.currentframe()
        while frame:
            if frame.f_globals.get("__name__") == "threading":
                frame.f_trace = None
            frame = frame.f_back
    except Exception:
        pass

def ShutDownThread(thread: _threading.Thread):
    '''Shuts down a thread by raising an exception'''
    if CPYTHON_SUPPORTED:
        if not thread.is_alive():
            return
        _async_raise(thread.ident, SystemExit)
    else:
        failed()

def KillThread(thread: _threading.Thread):
    '''Forcefully terminate a thread'''
    if not thread.is_alive():
        return
    tid = ctypes.c_long(thread.ident)
    handle = kernel32.OpenThread(1, False, tid)
    if handle:
        kernel32.TerminateThread(handle, 0)
        kernel32.CloseHandle(handle)
    else:
        failed()

def RestartThread(thread_func, args=(), kwargs={}, delay=0):
    '''Restart a thread by forcefully terminate it before starting it again (not recommended)'''
    def wrapper():
        while True:
            t = Thread(target=thread_func, args=args, kwargs=kwargs)
            t.start()
            time.sleep(delay)
            if _platform == WINDOWS:
                KillThread(t)
            t.join()
            time.sleep(delay)
    controller = Thread(target=wrapper)
    controller.start()
    return controller

def GetProcessMemoryUsage():
    return psutil.Process(os.getpid()).memory_info().rss

def GetProcessCPUUsage():
    return psutil.Process(os.getpid()).cpu_percent(interval=1.0)

def SuppressException():
    '''Suppress exception text and returns the original standard error output stream.'''
    from io import StringIO
    sys.stderr = StringIO
    return sys.__stderr__

def ProcessInitFix():
    '''Only call once in a script.'''
    set_start_method("spawn", force=True)

from logging import getLogger
from logging import DEBUG as _debug

DEBUG = _debug

def ThreadLog(thread: _threading.Thread):
    name = f"Thread-{thread.name or thread.ident}"
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    return logger

def ProcessLog(process: _multiprocessing.Process):
    name = f"Process-{process.name or process.pid}"
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    return logger

def GetCurrentHandle():
    '''Get current thread or process handle. Jump to main thread if found none.'''
    return current_thread() or current_process() or main_thread()

from traceback import format_stack

def ThreadStackTrace(thread):
    if CPYTHON_SUPPORTED:
        for tid, frame in sys._current_frames().items():
            if tid == thread.ident:
                return format_stack(frame)
    else:
        failed()

def ListActiveThreadsAndProcesses():
    return enumerate(), active_children()

NOT_ALLOWED = 0x127

from random import randrange

ThreadLock = _threading.Lock
ProcessLock = _multiprocessing.Lock

class CriticalLock:
    def __init__(self, lock, timeout=None, name=f"Unnamed_{randrange(10000,1000000)}"):
        self.lock = lock
        self.name = name
        self.timeout = timeout
        self.owner = None
        self.lock_count = 0
        self.total_hold_time = 0.0
    def __enter__(self):
        thread_name = current_thread().name
        if self.owner == thread_name:
            failed(NOT_ALLOWED)
        start = time.time()
        acquired = self.lock.acquire(timeout=self.timeout) if self.timeout else self.lock.acquire()
        if not acquired:
            failed()
        self.owner = thread_name
        self.lock_count += 1
        self.start_time = start
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        thread_name = current_thread().name
        if thread_name != self.owner:
            failed(NOT_ALLOWED)
        duration = time.time() - self.start_time
        self.total_hold_time += duration
        self.lock.release()
        self.owner = None

_Thread = Thread
_Process = Process

def _threadnoerror(target, *args, **kwargs):
    """For SuppressThreadError(). Use it instead"""
    def wrapped():
        try:
            target(*args, **kwargs)
        except Exception:
            pass
    return _threading.Thread(target=wrapped)

def _processnoerror(target, *args, **kwargs):
    """"For SuppressProcessError(). Use it instead"""
    def wrapped():
        try:
            target(*args, **kwargs)
        except Exception:
            pass
    return _multiprocessing.Process(target=wrapped)

def SuppressThreadError():
    """Suppresses all the exceptions from threading.Thread and returns the original (Experimental)"""
    global Thread
    Thread = _threadnoerror
    return _Thread

def SuppressProcessError():
    """Suppresses all the exceptions from multiprocessing.Process and returns the original (Experimental)"""
    global Process
    Process = _processnoerror
    return _Process

def ExtractThreadContext(thread: THREAD_OR_ID):
    """Extract the execution content from a running thread (Experimental)"""
    if _platform != WINDOWS:
        failed()
    if not isinstance(thread, int):
        handle = kernel32.OpenThread(THREAD_ALL_ACCESS, False, thread.ident)
    else:
        handle = kernel32.OpenThread(THREAD_ALL_ACCESS, False, thread)
    SuspendThread(handle)
    ctx = CONTEXT()
    ctx.ContextFlags = CONTEXT_FULL
    res = GetThreadContext(handle, ctypes.byref(ctx))
    if not res:
        failed()
    ResumeThread(handle)
    return ctx

def ImplantThreadContext(thread: THREAD_OR_ID, _ctx):
    """Implanting a new execution context on a running thread and returning the original context (Experimental)"""
    if _platform != WINDOWS:
        failed()
    if not isinstance(thread, int):
        handle = kernel32.OpenThread(THREAD_ALL_ACCESS, False, thread.ident)
    else:
        handle = kernel32.OpenThread(THREAD_ALL_ACCESS, False, thread)
    SuspendThread(handle)
    ctx = CONTEXT()
    ctx.ContextFlags = CONTEXT_FULL
    res = GetThreadContext(handle, ctypes.byref(ctx))
    if not res:
        failed()
    __ctx__ = ctx
    ctx = _ctx
    if not SetThreadContext(handle, ctypes.byref(ctx)):
        failed()
    ResumeThread(handle)
    return __ctx__

def ThreadMorph(thread1: _threading.Thread, thread2: _threading.Thread):
    """Morph a running thread into another running thread by implanting remote execution context on it (Experimental, Not Recommended)"""
    ctx = ExtractThreadContext(thread1)
    ImplantThreadContext(thread2, ctx)

def ReadProcessAsThreads(process: _multiprocessing.Process) -> list[int]:
    """Returns all thread IDs in the target running process."""
    proc = psutil.Process(process.pid)
    return [t.id for t in proc.threads()]

__local__ = _threading.local
__event__ = _threading.Event
__p_event__ = _multiprocessing.Event

class _CriticalTransaction:
    def __init__(self, store):
        """For CriticalLocal. Use it instead"""
        self.store = store
        self._backup = {}
    def __enter__(self):
        self._backup = dict(getattr(self.store._local, "data", {}))
        return self.store
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.store._local.data = self._backup

class CriticalLocal:
    def __init__(self):
        self._local = local()
        self._registry = {}
    def _thread_id(self):
        return get_ident()
    def set(self, key, value, ttl=None):
        tid = self._thread_id()
        if not hasattr(self._local, "data"):
            self._local.data = {}
        self._local.data[key] = value
        self._registry[(tid, key)] = {
            "timestamp": time.time(),
            "ttl": ttl,
            "origin": self._capture_context()
        }
    def get(self, key, default=None):
        tid = self._thread_id()
        if not hasattr(self._local, "data"):
            return default
        meta = self._registry.get((tid, key))
        if meta and meta["ttl"] is not None:
            if time.time() - meta["timestamp"] > meta["ttl"]:
                del self._local.data[key]
                return default
        return self._local.data.get(key, default)
    def delete(self, key):
        tid = self._thread_id()
        if hasattr(self._local, "data") and key in self._local.data:
            del self._local.data[key]
            self._registry.pop((tid, key), None)
    def has(self, key):
        return hasattr(self._local, "data") and key in self._local.data
    def keys(self):
        return list(getattr(self._local, "data", {}).keys())
    def snapshot(self):
        return self._capture_context()
    def _capture_context(self):
        frame = inspect.currentframe().f_back.f_back
        return {
            "function": frame.f_code.co_name,
            "file": frame.f_code.co_filename,
            "line": frame.f_lineno,
            "stack": inspect.stack()[2:6]
        }
    def inherit(self, keys):
        parent_tid = current_thread()._parent_ident if hasattr(current_thread(), "_parent_ident") else None
        if parent_tid:
            for key in keys:
                parent_data = self._registry.get((parent_tid, key))
                if parent_data:
                    self.set(key, parent_data["value"], ttl=parent_data["ttl"])
    def transaction(self):
        return _CriticalTransaction(self)

__author__ = "HOW2NOTARCHIVE" #you shouldn't have seen this

class CriticalEvent:
    _registry = []
    def __init__(self, name: Optional[str] = None, process=False):
        self._event = _threading.Event() if process == False else _multiprocessing.Event()
        self._name = name or f"CriticalEvent-{id(self)}"
        self._metadata = None
        self._expiry = None
        self._history = []
        self._on_set_hooks = []
        self._on_clear_hooks = []
        CriticalEvent._registry.append(self)
    def set(self, reason: Optional[str] = None, ttl: Optional[float] = None):
        self._event.set()
        self._expiry = time.time() + ttl if ttl else None
        self._metadata = {
            "reason": reason,
            "timestamp": time.time(),
            "caller": self._get_caller(),
        }
        self._history.append({"action": "set", **self._metadata})
        for hook in self._on_set_hooks:
            hook()
    def clear(self):
        self._event.clear()
        self._expiry = None
        self._history.append({
            "action": "clear",
            "timestamp": time.time(),
            "caller": self._get_caller(),
        })
        for hook in self._on_clear_hooks:
            hook()
    def is_set(self) -> bool:
        if self._expiry and time.time() > self._expiry:
            self.clear()
        return self._event.is_set()
    def wait(self, timeout: Optional[float] = None, on_timeout: Optional[Callable] = None) -> bool:
        result = self._event.wait(timeout)
        if not result and on_timeout:
            on_timeout()
        return result
    def last_set_info(self) -> Optional[Dict]:
        return self._metadata
    def history(self) -> List[Dict]:
        return list(self._history)
    def on_set(self, callback: Callable):
        self._on_set_hooks.append(callback)
    def on_clear(self, callback: Callable):
        self._on_clear_hooks.append(callback)
    def bind_to(self, context):
        context.set(f"event:{self._name}", self)
    @staticmethod
    def wait_any(events: List["CriticalEvent"], timeout: Optional[float] = None) -> Optional["CriticalEvent"]:
        start = time.time()
        while True:
            for event in events:
                if event.is_set():
                    return event
            if timeout and time.time() - start > timeout:
                return None
    @staticmethod
    def wait_all(events: List["CriticalEvent"], timeout: Optional[float] = None) -> bool:
        start = time.time()
        while True:
            if all(event.is_set() for event in events):
                return True
            if timeout and time.time() - start > timeout:
                return False
    def _get_caller(self) -> str:
        stack = traceback.extract_stack(limit=5)
        caller = stack[-3]
        return f"{caller.filename}:{caller.name}:{caller.lineno}"

__doc__ = """A not so light-weight extension for threading and multiprocessing with more features."""

if _platform == WINDOWS and CPYTHON_SUPPORTED:
    __pythonapi__ = ctypes.pythonapi
    py_object = ctypes.py_object
    Py_ssize_t = ctypes.c_ssize_t
    start_new_thread = __pythonapi__.PyThread_start_new_thread
    exit_thread = __pythonapi__.PyThread_exit_thread
    get_thread_ident = __pythonapi__.PyThread_get_thread_ident
    allocate_lock = __pythonapi__.PyThread_allocate_lock
    acquire_lock = __pythonapi__.PyThread_acquire_lock
    release_lock = __pythonapi__.PyThread_release_lock
    free_lock = __pythonapi__.PyThread_free_lock
    acquire_gil = __pythonapi__.PyGILState_Ensure
    release_gil = __pythonapi__.PyGILState_Release
    init_threads = __pythonapi__.PyEval_InitThreads
    save_thread = __pythonapi__.PyEval_SaveThread
    restore_thread = __pythonapi__.PyEval_RestoreThread
    PyFrame_FastToLocalsWithError = __pythonapi__.PyFrame_FastToLocalsWithError
    PyFrame_FastToLocalsWithError.argtypes = [py_object]
    PyFrame_FastToLocalsWithError.restype = ctypes.c_int
    PyFrame_LocalsToFast = __pythonapi__.PyFrame_LocalsToFast
    PyFrame_LocalsToFast.argtypes = [py_object, ctypes.c_int]
    PyFrame_LocalsToFast.restype = None
    
    class PyObject(ctypes.Structure):
        _fields_ = [
        ("ob_refcnt", Py_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ]

    class PyVarObject(ctypes.Structure):
        _fields_ = [
        ("ob_base", PyObject),
        ("ob_size", Py_ssize_t),
        ]

    class PyFrameObject(ctypes.Structure):
        _fields_ = [
        ("ob_base", PyVarObject),
        ("f_back", ctypes.c_void_p),
        ("f_code", py_object),
        ("f_builtins", py_object),
        ("f_globals", py_object),
        ("f_locals", py_object),
        ("f_valuestack", ctypes.POINTER(py_object)),
        ("f_stacktop", ctypes.POINTER(py_object)),
        ("f_trace", py_object),
        ("f_trace_lines", ctypes.c_int),
        ("f_trace_opcodes", ctypes.c_int),
        ("f_gen", py_object),
        ("f_lasti", ctypes.c_int),
        ("f_lineno", ctypes.c_int),
        ("f_iblock", ctypes.c_int),
        ("f_state", ctypes.c_char_p),
        ("f_executing", ctypes.c_char_p),
        ("f_localsplus", ctypes.POINTER(py_object))
        ]

import pickle

__pickle__ = pickle

def SyncFastToLocals(frame):
    PyFrame_FastToLocalsWithError(frame)

def SyncLocalsToFast(frame):
    PyFrame_LocalsToFast(frame, 0)

def GetCFrame(frame):
    if _platform != WINDOWS or not CPYTHON_SUPPORTED:
        failed()
    return ctypes.cast(id(frame), ctypes.POINTER(PyFrameObject))

def CaptureProcess():
    frame = sys._getframe(1)
    SyncFastToLocals(frame)
    code = frame.f_code
    locals_ = {k: v for k, v in frame.f_locals.items() if IsPickleable(v)}
    cp = {
        "filename": code.co_filename,
        "name": code.co_name,
        "lineno": frame.f_lineno,
        "lasti": frame.f_lasti,
        "locals": locals_,
    }
    return pickle.dumps(cp)

def IsPickleable(x):
    try:
        pickle.dumps(x)
        return True
    except Exception:
        return False

def _resume_process_entry(serialized_cp, func, args, kwargs):
    ResumeProcessMidFunction(func, serialized_cp, *args, **kwargs)

def RestoreProcess(serialized_cp, func, *args, **kwargs):
    proc = Process(
        target=_resume_process_entry,
        args=(serialized_cp, func, args, kwargs)
    )
    proc.start()
    proc.join()

def ResumeProcessMidFunction(func, checkpoint_bytes, *args, **kwargs):
    if _platform != WINDOWS or not CPYTHON_SUPPORTED:
        failed()
    cp = pickle.loads(checkpoint_bytes)
    def tracer(frame, event, arg):
        if event == "call":
            if frame.f_code.co_name == cp["name"] and frame.f_code.co_filename == cp["filename"]:
                SyncFastToLocals(frame)
                frame.f_locals.update(cp["locals"])
                SyncLocalsToFast(frame)
                try:
                    frame.f_lineno = cp["lineno"]
                except Exception as e:
                    pass
                try:
                    GetCFrame(frame).contents.f_lasti = cp["lasti"]
                except:
                    pass
                sys.settrace(None)
                return None
        return tracer
    sys.settrace(tracer)
    try:
        func(*args, **kwargs)
    finally:
        sys.settrace(None)

def _cleanup():
    """For CleanupAtExit(). Use it instead"""
    threads, processes = ListActiveThreadsAndProcesses()
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()
    for t in threads:
        if t is _threading.main_thread():
            continue
        if t.is_alive():
            KillThread(t)
            t.join()
    ForcedExit(0)

def CleanupAtExit():
    """Kill all threads and processes at exit"""
    import atexit
    atexit.register(_cleanup)
    import signal
    signal.signal(signal.SIGINT, lambda sig, frame: _cleanup())

SharedValue = _multiprocessing.Value

class LinkedThread(_threading.Thread):
    def __init__(self, base_thread: _threading.Thread, target=None):
        """(Experimental, would recommend on non-stopping threads)"""
        super().__init__(target=self._entry_point)
        self._base_thread = base_thread
        self._ctx = None
        self._custom_target = target
        self._ctx_ready = CriticalEvent(name=f"Thread{base_thread.ident}_ready")
    def _entry_point(self):
        self._ctx_ready.wait()
        if self._custom_target:
            self._custom_target()
        else:
            ImplantThreadContext(_threading.current_thread(), self._ctx)
    def run(self):
        self._ctx = ExtractThreadContext(self._base_thread)
        self._ctx_ready.set(reason="Context extracted")
        super().run()

class LinkedProcess(_multiprocessing.Process):
    def __init__(self, base_process: _multiprocessing.Process, target=None):
        """(Experimental, would recommend on non-stopping processes)"""
        super().__init__(target=self._entry_point)
        self._base_process = base_process
        self._ctx_list = []
        self._custom_target = target
        self._ctx_ready = CriticalEvent(name=f"Process{base_process.pid}_ready", process=True)
    def _entry_point(self):
        self._ctx_ready.wait()
        if self._custom_target:
            self._custom_target()
        else:
            current_threads = ReadProcessAsThreads(_multiprocessing.current_process())
            for i in range(min(len(current_threads), len(self._ctx_list))):
                ImplantThreadContext(current_threads[i], self._ctx_list[i])
    def run(self):
        base_threads = ReadProcessAsThreads(self._base_process)
        self._ctx_list = [ExtractThreadContext(base_threads[i]) for i in range(len(base_threads))]
        self._ctx_ready.set(reason="Context list extracted")
        super().run()

if _platform == LINUX:
    fork = os.fork

class FrozenSupport:
    def __init__(self):
        """(Experimental)"""
        self.is_frozen = getattr(sys, 'frozen', False)
        self.start_method = _multiprocessing.get_start_method(allow_none=True)
        self.is_spawn_child = any(arg.startswith('--multiprocessing-fork') for arg in sys.argv)
    def initialize(self):
        if not self.is_frozen:
            return
        if self.start_method == 'spawn' and self.is_spawn_child:
            self._bootstrap_spawn_child()
    def _bootstrap_spawn_child(self):
        """Runs the spawn-mode bootstrap logic for frozen child processes"""
        try:
            from multiprocessing.spawn import freeze_support as spawn_freeze_support  # pyright: ignore[reportMissingImports]
            spawn_freeze_support()
        except ImportError:
            return
    def Optimize(self):
        """Apply optimizations for frozen apps (Experimental)"""
        _threading.stack_size(2 * 1024 * 1024)
        __spawn_context__ = _multiprocessing.get_context(self.start_method or 'spawn')

from datetime import datetime
__now__ = datetime.now()

if not getattr(sys, "frozen", False) or not "HOW2NOTARCHIVE_SUPPRESS_MESSAGE" in os.environ:
    from sys import platform
    print(f"{__now__} | {platform}")
    print("MODULE NAME: THREADING AND MULTIPROCESSING EXTENSION")
    help(sys.modules[__name__])
    print(f"Small note from {__author__}:")
    print("Thank you for trusting and using my threading and multiprocessing extension!")
    print('To disable this instruction, add "HOW2NOTARCHIVE_SUPPRESS_MESSAGE" to os.environ. Thank you for your patience.')
    print("Have a nice day.")
#860th line lol
