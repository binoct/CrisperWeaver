import 'dart:async';
import 'dart:io';

import 'log_service.dart';

/// Which inference "task" currently owns the resident GPU model slot.
///
/// CrisperWeaver keeps its model services alive for the whole app session
/// (they are app-scoped Riverpod providers, not auto-disposed), so a model
/// loaded for one task stays resident on the GPU even after the user
/// navigates to a different task. On GPUs with little VRAM that stacks up
/// (e.g. the kokoro / qwen3-tts model plus a translation GGUF, or the
/// whisper transcription model), causing allocation failures.
///
/// [ModelResidency] keeps at most the models that fit in VRAM resident:
/// before a task loads its model it calls [reconcile], which frees other
/// tasks' models only when there isn't enough free VRAM for the new one.
/// This implements "unload the previous model before loading the next one
/// to save VRAM" — but smartly: a pipeline that genuinely needs two models
/// resident (and has the VRAM) keeps both instead of thrashing.
enum ModelSlot { transcribe, tts, translate, chat }

class _Slot {
  _Slot({required this.release, required this.footprint});
  final Future<void> Function() release;
  final int? Function() footprint;
}

/// Singleton coordinator that releases other tasks' resident models when one
/// task is about to load. Services register a release callback (and an
/// optional footprint reporter) for their slot in their provider; each load
/// path calls [reconcile] first.
class ModelResidency {
  ModelResidency._();
  static final instance = ModelResidency._();

  final Map<ModelSlot, _Slot> _slots = {};
  ModelSlot? _active;

  /// The task that was active immediately before the current one. After a
  /// switch this is the model most likely still holding VRAM, so it is the
  /// first candidate to shed when the new task needs room.
  ModelSlot? _previous;

  /// Master switch. When false (e.g. the local HTTP server is driving
  /// several models and owns their lifetimes itself), [reconcile] only
  /// records the active slot and never unloads another task's model.
  bool enabled = true;

  /// Optional live probe returning FREE GPU memory in bytes. When set, the
  /// coordinator uses the real figure for its decision. When null (the
  /// default — there is no portable free-VRAM read from Dart), it falls back
  /// to [gpuBudgetBytes] minus the sum of resident model footprints. A
  /// native DXGI / Vulkan query can be wired in here later.
  int? Function()? vramFreeProbe;

  /// VRAM held by models this coordinator does NOT own — typically models
  /// loaded inside worker isolates. Live captions spawns a recogniser and a
  /// translator isolate that open the native CrispASR session directly; they
  /// never call [reconcile] and cannot be released from here.
  ///
  /// That makes them invisible to [residentSum] (which only sums registered
  /// slots), yet they still occupy VRAM. Without this the coordinator would
  /// judge that memory free and let a main-thread load proceed alongside
  /// them — exactly the OOM this class exists to prevent. Registering the
  /// bytes here makes [reconcile] see the true headroom, while keeping these
  /// models strictly out of the unload candidate list.
  final Map<String, int> _external = {};

  /// Total bytes currently declared via [setExternal].
  int get externalBytes => _external.values.fold<int>(0, (a, b) => a + b);

  /// Declare (or update) an external reservation. [bytes] <= 0 clears it.
  void setExternal(String key, int bytes) {
    if (bytes > 0) {
      _external[key] = bytes;
    } else {
      _external.remove(key);
    }
  }

  /// Drop an external reservation (the isolates are gone).
  void clearExternal(String key) => _external.remove(key);

  /// Conservative GPU memory budget (bytes) used when no live probe or
  /// explicit setting is available. Defaults to 3 GiB — enough for one
  /// mid-size GGUF, deliberately conservative so small-GPU machines unload
  /// the previous task's model instead of OOMing. A startup probe
  /// ([probeTotalVram]) or an explicit settings value overrides this.
  int gpuBudgetBytes = 3 * 1024 * 1024 * 1024;

  /// True once [probeTotalVram] has run, so we don't re-probe.
  bool _probed = false;

  /// Best-effort: read the adapter's TOTAL VRAM once and use ~85% of it as
  /// the model budget. Cheap, cached, and purely advisory — any failure
  /// keeps [gpuBudgetBytes] at its default / settings value. Windows reads
  /// `Win32_VideoController.AdapterRAM`; other platforms are left to the
  /// default (no portable free-VRAM read from Dart).
  Future<void> probeTotalVram() async {
    if (_probed) return;
    _probed = true;
    try {
      int? total;
      if (Platform.isWindows) {
        final r = await Process.run('powershell', [
          '-NoProfile',
          '-NonInteractive',
          '-Command',
          '(Get-CimInstance Win32_VideoController '
              '| Measure-Object AdapterRAM -Maximum).Maximum'
        ]);
        if (r.exitCode == 0) total = int.tryParse(r.stdout.toString().trim());
      }
      if (total != null && total > 256 * 1024 * 1024) {
        gpuBudgetBytes = (total * 0.85).round();
        Log.instance.i('model', 'VRAM budget set from adapter',
            fields: {'bytes': gpuBudgetBytes});
      }
    } catch (err, st) {
      Log.instance.d('model', 'VRAM probe failed', error: err, stack: st);
    }
  }

  /// Quantised GGUF weights expand in VRAM (dequant scratch + KV cache +
  /// activations), so a model's true VRAM cost is well above its on-disk
  /// size. We apply this overhead factor to every footprint — mirroring
  /// [MemoryEstimator]'s 1.6× RAM heuristic — so the "is there room?"
  /// decision is not fooled by raw file sizes.
  static const double vramOverhead = 1.6;

  /// The slot whose model is currently resident, or null if none / unknown.
  ModelSlot? get active => _active;

  /// Register (or replace) the release callback + footprint reporter for
  /// [slot]. Idempotent — a later registration for the same slot wins.
  void register(ModelSlot slot, Future<void> Function() release,
      {int? Function()? footprint}) {
    _slots[slot] = _Slot(
      release: release,
      footprint: footprint ?? () => null,
    );
  }

  int _footprintOf(ModelSlot slot) =>
      _slots[slot]?.footprint() ?? 0;

  /// Decide, before [keep] loads a model of [newFootprintBytes], which other
  /// resident models to keep.
  ///
  /// Unloading is bounded by TASK, not by load:
  ///  * the same task loading again (or calling through on every sentence)
  ///    never evicts another task's model — that would tear down models
  ///    still in use without the user having gone anywhere;
  ///  * a switch to a DIFFERENT task is the only trigger, and even then the
  ///    previous task's model is kept when there is room.
  ///
  ///   free = vramFreeProbe?.call() ?? (gpuBudgetBytes - residentFootprintSum
  ///                                    - externalBytes)
  ///   if free >= newFootprintBytes -> keep everyone (no unload needed)
  ///   else unload the previous task first, then largest-first, until freed.
  ///
  /// Individual release failures are logged and swallowed so one stuck task
  /// cannot block another from loading. When coordination is disabled
  /// ([enabled] == false, e.g. the local server driving several models) only
  /// the active slot is updated.
  Future<void> reconcile(ModelSlot keep, int newFootprintBytes) async {
    // ── Task boundary ────────────────────────────────────────────────────
    // Unloading is only ever considered when the user moves to a DIFFERENT
    // task. Repeated loads inside the same task (a translation call per
    // sentence, re-opening the same model with new settings) replace that
    // task's own footprint — they must NOT evict another task's model, or
    // simply translating a few sentences would tear down the transcribe /
    // TTS model that is still in use.
    final switched = _active != keep;
    if (switched) _previous = _active;
    _active = keep;

    if (!enabled) return;
    if (!switched) return;

    final others = _slots.entries
        .where((e) => e.key != keep)
        .where((e) => _footprintOf(e.key) > 0)
        .toList();
    if (others.isEmpty) return;

    int effective(int bytes) => (bytes * vramOverhead).round();

    final residentSum =
        others.fold<int>(0, (s, e) => s + effective(_footprintOf(e.key)));
    // A live probe reports real free VRAM, which already accounts for
    // isolate-held models, so it is used as-is. The budget estimate only
    // knows about registered slots, so subtract the external reservations
    // ourselves or we would spend memory that is already gone.
    final probe = vramFreeProbe?.call();
    final free = probe ?? (gpuBudgetBytes - residentSum - externalBytes);
    final needed = effective(newFootprintBytes);

    if (free >= needed) {
      // Plenty of room for the new model alongside the others — keep them.
      return;
    }

    var deficit = needed - free;
    // The previous task's model goes first — that is the one the switch
    // stranded — then largest-first, shedding the most memory per unload.
    others.sort((a, b) {
      final pa = a.key == _previous ? 1 : 0;
      final pb = b.key == _previous ? 1 : 0;
      if (pa != pb) return pb.compareTo(pa);
      return effective(_footprintOf(b.key))
          .compareTo(effective(_footprintOf(a.key)));
    });
    for (final e in others) {
      try {
        await e.value.release();
      } catch (err, st) {
        Log.instance.w('model', 'reconcile release failed for ${e.key}',
            error: err, stack: st);
      }
      deficit -= effective(_footprintOf(e.key));
      if (deficit <= 0) break;
    }
  }
}
