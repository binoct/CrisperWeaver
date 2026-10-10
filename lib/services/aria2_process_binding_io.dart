import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

/// Windows-only process-lifetime binding for aria2c, mirroring what the
/// audio.cpp server does in C++: the child is placed inside a Job Object
/// configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. The job handle is
/// owned by this process; when CrisperWeaver exits — **normally, via a
/// crash, or killed from Task Manager** — the OS closes every handle and
/// the job is torn down, terminating every process inside it. No shutdown
/// code path needs to cooperate.
///
/// Returns the job handle, or `0` when binding is unavailable (non-Windows
/// platform or FFI failure). The caller must **not** close the returned
/// handle: closing it early would kill the running download. One handle per
/// download lives until process exit; that is the mechanism, not a leak.
int bindProcessToAppLifetimeJob(int pid) {
  if (!Platform.isWindows) return 0;
  try {
    final kernel32 = DynamicLibrary.open('kernel32.dll');
    final createJobObjectW = kernel32.lookupFunction<
        IntPtr Function(Pointer<Void>, Pointer<Uint16>),
        int Function(
            Pointer<Void>, Pointer<Uint16>)>('CreateJobObjectW');
    final setInformationJobObject = kernel32.lookupFunction<
        Int32 Function(IntPtr, Int32, Pointer<Uint8>, Uint32),
        int Function(int, int, Pointer<Uint8>, int)>(
        'SetInformationJobObject');
    final openProcess = kernel32.lookupFunction<
        IntPtr Function(Uint32, Int32, Uint32),
        int Function(int, int, int)>('OpenProcess');
    final assignProcessToJobObject = kernel32.lookupFunction<
        Int32 Function(IntPtr, IntPtr),
        int Function(int, int)>('AssignProcessToJobObject');
    final closeHandle =
        kernel32.lookupFunction<Int32 Function(IntPtr), int Function(int)>(
            'CloseHandle');
    // dart:ffi has no malloc (that lives in package:ffi) — allocate the
    // struct buffer through kernel32's own LocalAlloc/LocalFree instead, so
    // the file stays dependency-free.
    final localAlloc = kernel32.lookupFunction<
        IntPtr Function(Uint32, UintPtr),
        int Function(int, int)>('LocalAlloc');
    final localFree = kernel32.lookupFunction<IntPtr Function(IntPtr),
        int Function(int)>('LocalFree');
    const lmemFixed = 0x0000;

    const jobObjectExtendedLimitInformation = 9;
    const killOnJobClose = 0x2000;
    const processSetQuota = 0x0100;
    const processTerminate = 0x0001;

    final job = createJobObjectW(nullptr, nullptr);
    if (job == 0) return 0;

    // JOBOBJECT_EXTENDED_LIMIT_INFORMATION: `LimitFlags` sits at offset 16
    // (after the two LARGE_INTEGER time limits); every other field stays
    // zero, so one zero-filled 144-byte block with the flag set is enough.
    final infoAddr = localAlloc(lmemFixed, 144);
    if (infoAddr == 0) {
      closeHandle(job);
      return 0;
    }
    final infoBytes = Pointer<Uint8>.fromAddress(infoAddr).asTypedList(144);
    infoBytes.fillRange(0, 144, 0);
    ByteData.view(infoBytes.buffer).setUint64(16, killOnJobClose, Endian.little);
    final okJob = setInformationJobObject(
        job, jobObjectExtendedLimitInformation, Pointer<Uint8>.fromAddress(infoAddr), 144);
    localFree(infoAddr);
    if (okJob == 0) {
      closeHandle(job);
      return 0;
    }

    final process = openProcess(processSetQuota | processTerminate, 0, pid);
    if (process == 0) {
      closeHandle(job);
      return 0;
    }
    final okAssign = assignProcessToJobObject(job, process);
    closeHandle(process);
    if (okAssign == 0) {
      closeHandle(job);
      return 0;
    }
    // Success: keep `job` open on purpose (see doc comment above).
    return job;
  } catch (_) {
    // Best effort only — the explicit cancel/stall paths still kill the
    // child directly, so a failed binding must never break a download.
    return 0;
  }
}
