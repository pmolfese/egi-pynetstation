#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AUDIO TIMING TEST - PsychoPy (PTB audio) -> EGI Net Station.

One identical tone is played every --soa seconds, scheduled on the PTB
audio clock on a fixed absolute grid (never locked to screen flips), and
each ``stm+`` marker is stamped with the tone's scheduled onset mapped into
the Net Station timebase (``send_event(..., start=...)``). Marker vs. audio
onset (EEG auxiliary input / loopback) is analysed externally, e.g. with
MFFTimingInspector.

Examples
--------
    python audio_timing_test.py amp --duration 3600 --staged-drift \
        --warmup 30 --log audio_timing.csv \
        --error-log audio_timing_errors.jsonl

    python audio_timing_test.py amp --sessions 2 --duration 600 \
        --staged-drift --warmup 30 --log audio_timing.csv \
        --error-log audio_timing_errors.jsonl

    python audio_timing_test.py --list-devices

Events sent to Net Station (per recording session)
--------------------------------------------------
  TSTA  start of the tone sequence (sent "now")
  stm+  one per tone, stamped at the scheduled tone onset
  TEND  end of the tone sequence (sent "now")

A small black status window stays open so Escape / q can stop the run; the
recording is closed cleanly and the log is still written. The window is
windowed by default (--fullscreen exists but is not needed for audio).
"""

import csv
import inspect
import statistics
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import egi_pynetstation
from egi_pynetstation.NetStation import NetStation


# ============================================================
# CONSTANTS
# ============================================================

# Exact Windows name of the wired output device on the acquisition PC.
# Used on Windows when --device is not given.
DEFAULT_WINDOWS_DEVICE = "Auriculares (Realtek(R) Audio)"

TONE_CODE = "stm+"
START_CODE = "TSTA"
END_CODE = "TEND"

QUIT_KEYS = ["escape", "q", "Q"]

NETWORK_DEFAULTS = {
    "local": ("127.0.0.1", "216.239.35.4", 9885),
    "amp": ("10.10.10.42", "10.10.10.51", 55513),
}


# ============================================================
# AUDIO DEVICES
# ============================================================

def _output_devices():
    """Output-capable devices, in psychtoolbox enumeration order."""
    import psychtoolbox.audio as pta
    return [d for d in pta.get_devices() if d["NrOutputChannels"] > 0]


def list_devices_and_exit():
    try:
        devices = _output_devices()
    except ImportError:
        print("psychtoolbox is not importable; cannot list devices.")
        raise SystemExit(1)
    print(f"{len(devices)} output device(s):")
    for position, dev in enumerate(devices):
        print(f"  [{position}] {dev['DeviceName']!r} "
              f"({dev['NrOutputChannels']:.0f} ch, "
              f"{dev['DefaultSampleRate']:.0f} Hz)")
    print("\nSelect one with --device <number | exact name | unique part>.")
    raise SystemExit(0)


def _device_error(message, names):
    print(f"ERROR: {message}.", file=sys.stderr)
    print("\nAvailable output devices:", file=sys.stderr)
    for position, name in enumerate(names):
        print(f"  [{position}] {name!r}", file=sys.stderr)
    raise SystemExit(2)


def resolve_device(spec):
    """Turn a --device value (position, exact name or unique part) into a name."""
    try:
        devices = _output_devices()
    except ImportError:
        return spec
    names = [d["DeviceName"] for d in devices]
    text = str(spec).strip()
    if text.isdigit():
        position = int(text)
        if 0 <= position < len(names):
            return names[position]
        _device_error(f"no device at position [{position}]", names)
    if text in names:
        return text
    matches = [n for n in names if text.lower() in n.lower()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        _device_error(f"{text!r} is ambiguous ({len(matches)} matches)", names)
    _device_error(f"no output device matching {text!r}", names)


def setup_audio_prefs(args):
    """Must run before psychopy.sound is imported."""
    from psychopy import prefs
    # Lower-case "ptb": recent PsychoPy versions reject "PTB".
    prefs.hardware["audioLib"] = ["ptb"]
    try:
        prefs.hardware["audioLatencyMode"] = args.latency_mode
    except Exception:
        pass
    device = None
    if args.device:
        device = resolve_device(args.device)
    elif sys.platform == "win32":
        device = DEFAULT_WINDOWS_DEVICE
    if device:
        prefs.hardware["audioDevice"] = device
    return device or "system default"


# ============================================================
# COMMAND LINE  (network / session / drift options mirror the
# photocell test so the same flags mean the same thing)
# ============================================================

def add_common_arguments(parser):
    parser.add_argument("mode", choices=["local", "amp", "custom"])
    parser.add_argument("--ip-cmd", help="Net Station / command IPv4 address")
    parser.add_argument("--ip-clock", help="NTP server / amplifier IPv4 address")
    parser.add_argument("--port", type=int, help="ECI TCP port")

    parser.add_argument(
        "--duration", type=float, default=300.0,
        help="Duration of each recording session in seconds (default: 300)",
    )
    parser.add_argument(
        "--sessions", "--recordings", dest="sessions", type=int, default=1,
        help="Consecutive Net Station recordings on one ECI connection",
    )
    parser.add_argument(
        "--session-break", type=float, default=0.0,
        help="Seconds between end_rec() and the next begin_rec()",
    )
    parser.add_argument(
        "--prep", type=float, default=0.0,
        help="Seconds between connect() and begin_rec() (pre-sync drift "
             "samples are backfilled unless --no-drift-presync)",
    )
    parser.add_argument(
        "--warmup", type=float, default=0.0,
        help="Seconds to wait after begin_rec() before the first tone",
    )

    g = parser.add_argument_group("drift correction")
    g.add_argument("--staged-drift", action="store_true",
                   help="Provisional short model first (drift_warmup=True)")
    g.add_argument("--warmup-model-min-samples", type=int, default=None)
    g.add_argument("--warmup-model-min-span", type=float, default=None)
    g.add_argument("--warmup-sample-interval", type=float, default=None)
    g.add_argument("--sample-interval", type=float, default=None)
    g.add_argument("--drift-min-samples", type=int, default=None)
    g.add_argument("--drift-min-span", type=float, default=None)
    g.add_argument("--drift-max-delay", type=float, default=None)
    g.add_argument("--drift-max-residual", type=float, default=None)
    g.add_argument("--drift-window-minutes", type=float, default=None)
    g.add_argument("--drift-samples", type=int, default=None)
    g.add_argument("--drift-sample-spacing", type=float, default=None)
    g.add_argument("--drift-min-pause", type=float, default=None)
    g.add_argument("--drift-stall-after", type=int, default=None)
    g.add_argument("--drift-excursion-threshold", type=float, default=None)
    g.add_argument("--drift-sample-reject-offset", type=float, default=None)
    g.add_argument("--drift-status-interval", type=float, default=None)
    g.add_argument("--drift-slew", type=float, default=None)
    g.add_argument("--drift-max-model-age", type=float, default=None)
    g.add_argument("--drift-cooperative", action="store_true",
                   help="Sample NTP only in the gaps between tones instead "
                        "of the background thread")
    g.add_argument("--no-drift-presync", action="store_true")
    g.add_argument("--no-drift-correction", action="store_true")

    o = parser.add_argument_group("events and logging")
    o.add_argument("--sync-events", action="store_true",
                   help="send_event(wait=True): write to the socket inline")
    o.add_argument("--debug", action="store_true")
    o.add_argument("--error-log", help="JSON-lines ECI error log path")
    o.add_argument("--log", help="CSV file with one row per tone")


def validate_common(parser, args):
    if args.sessions < 1:
        parser.error("--sessions must be at least 1")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    for name in ("session_break", "prep", "warmup",
                 "drift_window_minutes", "drift_sample_spacing",
                 "drift_slew", "drift_max_model_age", "drift_min_pause",
                 "drift_excursion_threshold", "drift_sample_reject_offset",
                 "drift_status_interval", "warmup_model_min_span"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("drift_max_delay", "drift_max_residual",
                 "warmup_sample_interval"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name, minimum in (("drift_samples", 1), ("drift_stall_after", 1),
                          ("warmup_model_min_samples", 2)):
        value = getattr(args, name)
        if value is not None and value < minimum:
            parser.error(f"--{name.replace('_', '-')} must be at least "
                         f"{minimum}")


def resolve_network(args):
    if args.mode in NETWORK_DEFAULTS:
        cmd, clock, port = NETWORK_DEFAULTS[args.mode]
        return args.ip_cmd or cmd, args.ip_clock or clock, args.port or port
    if not (args.ip_cmd and args.ip_clock and args.port):
        raise SystemExit("custom mode requires --ip-cmd, --ip-clock and --port")
    return args.ip_cmd, args.ip_clock, args.port


def build_parser():
    parser = ArgumentParser(
        description="PTB-scheduled audio timing test for EGI Net Station.",
    )
    add_common_arguments(parser)
    a = parser.add_argument_group("audio")
    a.add_argument("--soa", type=float, default=0.5,
                   help="Onset-to-onset interval in s (default: 0.5)")
    a.add_argument("--tone-freq", type=float, default=1000.0)
    a.add_argument("--tone-duration", type=float, default=0.1)
    a.add_argument("--volume", type=float, default=1.0)
    a.add_argument("--no-hamming", action="store_true",
                   help="Disable the on/off ramp")
    a.add_argument("--device", default=None,
                   help="Output device: number, exact name or unique part "
                        f"(Windows default: {DEFAULT_WINDOWS_DEVICE!r})")
    a.add_argument("--list-devices", action="store_true",
                   help="List output devices and exit (no mode needed)")
    a.add_argument("--latency-mode", type=int, default=3,
                   choices=[0, 1, 2, 3, 4],
                   help="PsychoPy audioLatencyMode (default: 3)")
    a.add_argument("--audio-lead", type=float, default=0.060,
                   help="Hand each tone to PTB this many s before onset")
    a.add_argument("--late-margin", type=float, default=0.010,
                   help="Closer than this to onset -> shift the grid")
    a.add_argument("--recalibrate-every", type=int, default=500,
                   help="Re-measure PTB<->capture offset every N tones")
    a.add_argument("--max-bracket-ms", type=float, default=0.5,
                   help="Warn if the offset measurement is coarser")
    w = parser.add_argument_group("status window")
    w.add_argument("--fullscreen", action="store_true")
    w.add_argument("--screen", type=int, default=0)
    return parser


# ============================================================
# egi-pynetstation
# ============================================================

def check_package():
    required = ["capture_time", "time_at_capture", "rec_start",
                "clock_state", "session_summary", "drift_ready"]
    missing = [n for n in required if not hasattr(NetStation, n)]
    try:
        if "start" not in inspect.signature(NetStation.send_event).parameters:
            missing.append("send_event(start=...)")
    except (TypeError, ValueError):
        missing.append("send_event(start=...)")
    if missing:
        raise SystemExit(
            "egi-pynetstation is too old or the wrong copy is loaded.\n"
            f"  version: {getattr(egi_pynetstation, '__version__', 'unknown')}\n"
            f"  file: {inspect.getfile(NetStation)}\n"
            f"  missing: {', '.join(missing)}"
        )


def connect_with_drift_options(ns, ntp_ip, args):
    try:
        ns.connect(
            ntp_ip=ntp_ip,
            handshake=True,
            drift_correction=not args.no_drift_correction,
            drift_min_samples=args.drift_min_samples,
            drift_min_span=args.drift_min_span,
            drift_warmup=True if args.staged_drift else None,
            drift_warmup_min_samples=args.warmup_model_min_samples,
            drift_warmup_min_span=args.warmup_model_min_span,
            drift_warmup_interval=args.warmup_sample_interval,
            drift_max_delay=args.drift_max_delay,
            drift_max_residual=args.drift_max_residual,
            drift_window_minutes=args.drift_window_minutes,
            drift_samples=args.drift_samples,
            drift_sample_spacing=args.drift_sample_spacing,
            drift_presync=not args.no_drift_presync,
            drift_slew=args.drift_slew,
            drift_max_model_age=args.drift_max_model_age,
            auto_drift=True,
            auto_drift_interval=args.sample_interval,
            auto_drift_min_pause=args.drift_min_pause,
            auto_drift_background=not args.drift_cooperative,
        )
    except TypeError as err:
        raise SystemExit(
            "The imported egi_pynetstation does not support the options this "
            f"test requires.\n  error: {err}\n"
            f"  loaded from: {inspect.getfile(NetStation)}"
        )


def configure_drift_monitoring(ns, args):
    ns.set_drift_stability(stall_after=args.drift_stall_after)
    if any(v is not None for v in (args.drift_excursion_threshold,
                                   args.drift_sample_reject_offset,
                                   args.drift_status_interval)):
        print("Drift monitoring:", ns.set_drift_monitoring(
            excursion_threshold=args.drift_excursion_threshold,
            sample_reject_offset=args.drift_sample_reject_offset,
            status_interval=args.drift_status_interval,
        ))


def run_prep(ns, args):
    """Stand-in for cap application between connect() and begin_rec()."""
    if args.prep <= 0:
        return
    print(f"Simulating {args.prep:g} s of pre-sync prep (Ctrl-C to stop)...")
    deadline = time.monotonic() + args.prep
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if args.drift_cooperative:
            ns.sample_drift_if_due(available_pause=remaining)
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    print(f"  {len(ns.drift_history())} pre-sync drift samples.")


def warm_up_before_trials(ns, seconds, idle, cooperative=False):
    """Keep the process responsive while the clock model warms up."""
    if seconds <= 0:
        return 0.0
    started = ns.capture_time()
    deadline = started + seconds
    print(f"Warming up clock model for {seconds:g} s...")
    while True:
        remaining = deadline - ns.capture_time()
        if remaining <= 0:
            break
        if cooperative:
            ns.sample_drift_if_due(available_pause=remaining)
        idle()
    elapsed = ns.capture_time() - started
    status = ns.drift_ready()
    print(
        f"Warmup complete after {elapsed:.1f} s: "
        f"{status['samples']} valid drift samples over "
        f"{status['span_s']:.0f} s, "
        + ("model ready." if status["ready"]
           else f"model NOT ready ({status['reason']}, "
                f"{status['pending_correction_ms']:.2f} ms pending).")
    )
    return elapsed


# (state key, CSV column); columns ending in _ms are converted from s.
CLOCK_FIELDS = (
    ("drift_samples", "drift_samples"),
    ("drift_valid_samples", "drift_valid_samples"),
    ("drift_rejected_samples", "drift_rejected_samples"),
    ("drift_model_samples", "drift_model_samples"),
    ("drift_model_span", "drift_model_span_s"),
    ("drift_model_stage", "drift_model_stage"),
    ("active_drift_model_stage", "active_drift_model_stage"),
    ("drift_stable_engaged", "drift_stable_engaged"),
    ("drift_model_max_residual", "drift_model_max_residual_ms"),
    ("drift_model_rms_residual", "drift_model_rms_residual_ms"),
    ("drift_max_residual", "drift_max_residual_ms"),
    ("drift_window", "drift_window_s"),
    ("drift_accepted_fits", "drift_accepted_fits"),
    ("drift_rejected_fits", "drift_rejected_fits"),
    ("drift_last_reject_reason", "drift_last_reject_reason"),
    ("drift_pending_error", "drift_pending_error_ms"),
    ("drift_model_age", "drift_model_age_s"),
    ("drift_samples_per_call", "drift_samples_per_call"),
    ("sys_mono_skew", "sys_mono_skew_ms"),
    ("ntp_offset_raw", "ntp_offset_raw"),
)

CLOCK_COLUMNS = (
    ["drift_correction_ms", "drift_slope_ms_per_hour",
     "active_drift_slope_ms_per_hour"]
    + [col for _, col in CLOCK_FIELDS]
    + ["clock_state_error"]
)

NTP_COLUMNS = [
    "ntp_offset", "ntp_delay", "ntp_valid", "ntp_reject_reason",
    "ntp_burst_size", "ntp_burst_ok", "ntp_burst_worst_delay",
    "ntp_offset_mono", "ntp_sample_sys_mono_skew",
]


def add_clock_diagnostics(record, ns):
    """Flat drift-model snapshot. Call only outside the timing-critical path."""
    try:
        state = ns.clock_state()
    except Exception as err:
        record["clock_state_error"] = f"{type(err).__name__}: {err}"
        return
    initial = state.get("ntp_offset")
    predicted = state.get("predicted_ntp_offset")
    if isinstance(initial, (int, float)) and isinstance(predicted, (int, float)):
        record["drift_correction_ms"] = (predicted - initial) * 1000.0
    for key, col in (("drift_slope", "drift_slope_ms_per_hour"),
                     ("active_drift_slope", "active_drift_slope_ms_per_hour")):
        value = state.get(key)
        if isinstance(value, (int, float)):
            record[col] = value * 1000.0 * 3600.0
    for key, col in CLOCK_FIELDS:
        value = state.get(key)
        if value is None:
            continue
        if col.endswith("_ms") and isinstance(value, (int, float)):
            value *= 1000.0
        record[col] = value


def drift_sample_record(ns, label, sample, session, local_time):
    record = {
        "session": session,
        "phase": label,
        "local_time": local_time,
        "package_time": ns.getTime(),
        "ntp_offset": sample.get("offset"),
        "ntp_delay": sample.get("delay"),
        "ntp_valid": sample.get("valid"),
        "ntp_reject_reason": sample.get("reject_reason"),
        "ntp_burst_size": sample.get("burst_size"),
        "ntp_burst_ok": sample.get("burst_ok"),
        "ntp_burst_worst_delay": sample.get("burst_worst_delay"),
        "ntp_offset_mono": sample.get("offset_mono"),
        "ntp_sample_sys_mono_skew": sample.get("sys_mono_skew"),
    }
    add_clock_diagnostics(record, ns)
    return record


def report_end(ns):
    errors = ns.event_errors()
    if errors:
        print(f"WARNING: {len(errors)} asynchronous event sends failed.")
        for item in errors[:5]:
            print("  ", item)
    else:
        print("Asynchronous event send errors: none")
    state = ns.clock_state()
    print(
        f"\nDrift model: {state['drift_accepted_fits']} fits accepted, "
        f"{state['drift_rejected_fits']} rejected"
        + (f", STALLED ({state['drift_consecutive_rejections']} "
           f"consecutive rejections)" if state.get("drift_stalled") else "")
    )
    print(f"Active drift stage: {state.get('active_drift_model_stage')} "
          f"(stable engaged: {state.get('drift_stable_engaged')})")
    if state["drift_accepted_fits"] == 0:
        print("WARNING: drift correction never engaged. Timestamps used the "
              "initial NTP sync only.")
    print("Drift estimate:", ns.drift_estimate())
    summary = ns.session_summary()
    print("\nSession summary:", summary)
    if not summary.get("ok"):
        print("WARNING: this run did not meet every health check. Treat its "
              "timing numbers with care.")


def write_records(path, records, columns):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print("Log written:", path)


def print_stats(label, values, unit="ms", scale=1.0):
    values = [v * scale for v in values if isinstance(v, (int, float))]
    if not values:
        return
    ordered = sorted(values)
    print(f"  {label}: n={len(values)} mean {statistics.mean(values):.3f} "
          f"median {statistics.median(values):.3f} "
          f"p95 {ordered[int(0.95 * (len(values) - 1))]:.3f} "
          f"min {ordered[0]:.3f} max {ordered[-1]:.3f} {unit}")


# ============================================================
# LOG LAYOUT
# ============================================================

CSV_COLUMNS = [
    "trial", "session", "session_trial", "phase", "send_mode",
    "tone_freq_hz", "planned_onset_ptb", "request_ptb", "request_lead_ms",
    "schedule_shift_ms", "ptb_minus_capture_s", "calibration_bracket_ms",
    "onset_capture_time", "package_time", "local_time",
    "send_call_span_ms", "pending_events", "send_error",
] + CLOCK_COLUMNS + NTP_COLUMNS


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--list-devices" in argv:
        list_devices_and_exit()

    parser = build_parser()
    args = parser.parse_args(argv)
    validate_common(parser, args)
    if args.soa <= args.audio_lead:
        parser.error("--soa must be longer than --audio-lead")
    if args.recalibrate_every < 1:
        parser.error("--recalibrate-every must be at least 1")
    n_tones = int(args.duration // args.soa)
    if n_tones < 1:
        parser.error("--duration is shorter than one --soa")

    ip_cmd, ip_clock, port = resolve_network(args)
    device = setup_audio_prefs(args)          # before psychopy.sound

    import psychopy
    from psychopy import sound, visual
    import psychtoolbox as ptb

    check_package()
    send_mode = "sync" if args.sync_events else "async"

    print("")
    print("=" * 50)
    print("AUDIO TIMING TEST")
    print("PsychoPy          :", psychopy.__version__)
    print("egi-pynetstation  :",
          getattr(egi_pynetstation, "__version__", "unknown"))
    print("NetStation.py     :", inspect.getfile(NetStation))
    print("Network           :", f"{ip_cmd}:{port}, clock {ip_clock}")
    print("Audio device      :", device)
    print(f"Tones per session : {n_tones} x {args.tone_freq:.0f} Hz, "
          f"SOA {args.soa:.3f} s ({n_tones * args.soa / 60.0:.1f} min)")
    print(f"Sessions          : {args.sessions}")
    print("=" * 50, flush=True)

    records = []
    ns = None
    win = None
    tone = None
    shifts = 0
    leads = []

    try:
        # --- audio: one preloaded tone, before anything starts ----
        tone = sound.Sound(
            value=args.tone_freq, secs=args.tone_duration,
            stereo=True, hamming=not args.no_hamming,
        )
        tone.setVolume(args.volume)
        print("PTB audio initialised", flush=True)

        # --- Net Station ------------------------------------------
        ns = NetStation(ip_cmd, port, debug=args.debug,
                        error_log=args.error_log)
        connect_with_drift_options(ns, ip_clock, args)
        run_prep(ns, args)
        ns.begin_rec()
        if ns.rec_start() is None:
            raise RuntimeError("begin_rec() returned without a recording "
                               "start time")
        configure_drift_monitoring(ns, args)
        print(f"Event send mode: {send_mode}")
        print("Drift sampling: "
              + ("between tones (cooperative)" if args.drift_cooperative
                 else "background thread (default)"))
        print("Press q or escape in the status window to stop early; "
              "the log is still written.", flush=True)

        # --- status window (also carries the quit keys) -----------
        win = visual.Window(
            fullscr=args.fullscreen, screen=args.screen,
            allowGUI=not args.fullscreen, color="black",
            units="height", checkTiming=False,
        )
        win.mouseVisible = False
        message = visual.TextStim(win=win, text="", font="Arial",
                                  height=0.04, wrapWidth=1.4, color="white")
        should_quit = make_quit_checker()

        def status(text):
            message.text = text

        def idle():
            if should_quit():
                raise KeyboardInterrupt
            message.draw()
            win.flip()

        try:
            ns._append_error_log({
                "record": "audio_timing",
                "audio_device": device,
                "soa": args.soa,
                "tone_freq": args.tone_freq,
                "tone_duration": args.tone_duration,
                "audio_lead": args.audio_lead,
                "tones_per_session": n_tones,
                "sessions": args.sessions,
                "session_duration": args.duration,
                "session_break": args.session_break,
                "clock": None,
            })
        except Exception:
            pass

        status("Clock model warm-up\n\nESCAPE / q = quit")
        warm_up_before_trials(ns, args.warmup, idle,
                              cooperative=args.drift_cooperative)

        # --- PTB <-> capture clock offset -------------------------
        offset = {"value": None, "bracket_ms": None}

        def calibrate(label, session, n=100):
            best = None
            for _ in range(n):
                a = ns.capture_time()
                g = ptb.GetSecs()
                b = ns.capture_time()
                if best is None or (b - a) < best[0]:
                    best = (b - a, g - (a + b) / 2.0)
            previous = offset["value"]
            offset["value"] = best[1]
            offset["bracket_ms"] = best[0] * 1000.0
            change = ("" if previous is None
                      else ", change %.4f ms" % ((best[1] - previous) * 1000.0))
            print("Clock calibration (%s): bracket %.4f ms%s"
                  % (label, offset["bracket_ms"], change), flush=True)
            if offset["bracket_ms"] > args.max_bracket_ms:
                print("WARNING: calibration bracket > %.1f ms - check clock "
                      "resolution (check_clocks)." % args.max_bracket_ms,
                      flush=True)
            records.append({
                "session": session, "phase": "clock_calibration",
                "ptb_minus_capture_s": offset["value"],
                "calibration_bracket_ms": offset["bracket_ms"],
                "local_time": time.time(),
            })

        records.append(drift_sample_record(
            ns, "drift_sample_start", ns.sample_drift(), 1, time.time()))

        trial = 0
        for session in range(1, args.sessions + 1):
            if session > 1:
                ns.send_event(event_type=END_CODE, label=END_CODE)
                ns.flush_events()
                ns.end_rec()
                print(f"Session {session - 1} recording stopped", flush=True)
                if args.session_break:
                    status("Break between recordings\n\nESCAPE / q = quit")
                    print(f"Waiting {args.session_break:g} s between "
                          "recordings...", flush=True)
                    deadline = time.monotonic() + args.session_break
                    while time.monotonic() < deadline:
                        idle()
                ns.begin_rec()
                print(f"Started session {session}/{args.sessions} on the "
                      "existing connection.", flush=True)
                records.append(drift_sample_record(
                    ns, "session_start", ns.sample_drift(), session,
                    time.time()))

            status(f"Audio timing test running\n"
                   f"session {session}/{args.sessions}\n\nESCAPE / q = stop")
            calibrate(f"session {session} start", session)
            ns.send_event(event_type=START_CODE, label=START_CODE)
            print("Event sent:", START_CODE, flush=True)

            first_onset = ptb.GetSecs() + 0.25
            target = first_onset

            for i in range(n_tones):
                if should_quit():
                    raise KeyboardInterrupt
                if i > 0 and i % args.recalibrate_every == 0:
                    calibrate(f"session {session} tone {i}", session)

                target = first_onset + i * args.soa

                # The flip only decides WHEN the request is made, not
                # when the tone sounds or what time its marker carries.
                while ptb.GetSecs() < target - args.audio_lead:
                    idle()

                request_time = ptb.GetSecs()
                shift_ms = None
                if request_time > target - args.late_margin:
                    # Never ask PTB for a time it cannot meet: shift this
                    # and all later onsets so tone and marker stay identical.
                    new_target = request_time + args.audio_lead
                    shift_ms = (new_target - target) * 1000.0
                    first_onset += new_target - target
                    target = new_target
                    shifts += 1
                    print("Late audio request at tone %d: grid shifted "
                          "%.1f ms" % (i + 1, shift_ms), flush=True)

                tone.play(when=target)

                onset_capture = target - offset["value"]
                package_time = ns.time_at_capture(onset_capture)
                record = {
                    "trial": trial + 1,
                    "session": session,
                    "session_trial": i + 1,
                    "phase": "tone",
                    "send_mode": send_mode,
                    "tone_freq_hz": args.tone_freq,
                    "planned_onset_ptb": target,
                    "request_ptb": request_time,
                    "request_lead_ms": (target - request_time) * 1000.0,
                    "schedule_shift_ms": shift_ms,
                    "ptb_minus_capture_s": offset["value"],
                    "calibration_bracket_ms": offset["bracket_ms"],
                    "onset_capture_time": onset_capture,
                    "package_time": package_time,
                }
                span_start = ns.capture_time()
                try:
                    ns.send_event(event_type=TONE_CODE, label=TONE_CODE,
                                  start=package_time,
                                  wait=args.sync_events)
                except Exception as err:
                    record["send_error"] = f"{type(err).__name__}: {err}"
                span_end = ns.capture_time()
                record["send_call_span_ms"] = (span_end - span_start) * 1000.0
                record["pending_events"] = ns.pending_events()
                record["local_time"] = time.time()
                leads.append(record["request_lead_ms"])

                # Off the critical path from here on.
                add_clock_diagnostics(record, ns)
                records.append(record)
                trial += 1

                if args.drift_cooperative and i < n_tones - 1:
                    next_request = (first_onset + (i + 1) * args.soa
                                    - args.audio_lead)
                    pause = next_request - ptb.GetSecs()
                    result = ns.sample_drift_if_due(available_pause=pause)
                    if result.get("sampled"):
                        records.append(drift_sample_record(
                            ns, "drift_sample", result["sample"], session,
                            time.time()))

                if (i + 1) % 100 == 0:
                    print(f"Session {session}: tones played {i + 1} / "
                          f"{n_tones}", flush=True)

            # let the last tone finish before closing the session
            end_wait = target + args.tone_duration + 0.2
            while ptb.GetSecs() < end_wait:
                idle()

        status("Audio timing test complete\n\nThank you.")
        completion_deadline = ptb.GetSecs() + 4.0
        while ptb.GetSecs() < completion_deadline:
            idle()

        ns.flush_events()
        records.append(drift_sample_record(
            ns, "drift_sample_end", ns.sample_drift(), args.sessions,
            time.time()))

        tone_rows = [r for r in records if r.get("phase") == "tone"]
        print(f"\nSequence finished: {len(tone_rows)} tones, "
              f"{shifts} grid shift(s)")
        print_stats("request lead before onset", leads)
        print_stats(f"send_event() call span ({send_mode})",
                    [r.get("send_call_span_ms") for r in tone_rows],
                    unit="us", scale=1000.0)
        pending = [r["pending_events"] for r in tone_rows
                   if isinstance(r.get("pending_events"), (int, float))]
        if pending:
            print(f"  queue depth max {max(pending)} "
                  "(0 means the sender kept up)")
        report_end(ns)
        return 0

    except KeyboardInterrupt:
        print("\nStopped early by the operator.", file=sys.stderr)
        return 130
    finally:
        try:
            if tone is not None:
                tone.stop()
        except Exception:
            pass
        if ns is not None and getattr(ns, "_connected", True):
            try:
                if ns.rec_start() is not None:
                    try:
                        ns.send_event(event_type=END_CODE, label=END_CODE)
                        ns.flush_events()
                        print("Event sent:", END_CODE, flush=True)
                    except Exception as err:
                        print("Could not send end event:", err)
                    ns.end_rec()
                    print("Net Station recording stopped", flush=True)
            except Exception as err:
                print(f"EndRecording failed: {type(err).__name__}: {err}",
                      file=sys.stderr)
            try:
                ns.disconnect()
                print("Net Station disconnected", flush=True)
            except Exception as err:
                print(f"Disconnect failed: {type(err).__name__}: {err}",
                      file=sys.stderr)
        if win is not None:
            try:
                win.close()
            except Exception:
                pass
        write_records(args.log, records, CSV_COLUMNS)


def make_quit_checker():
    """PTB keyboard (no window focus needed) plus pyglet as a fallback."""
    from psychopy import event

    def poll_pyglet():
        return bool(event.getKeys(keyList=QUIT_KEYS))

    try:
        from psychopy.hardware import keyboard
        device = keyboard.Keyboard()
    except Exception as err:
        print(f"Note: PTB keyboard unavailable ({type(err).__name__}: {err}); "
              "quit keys need the status window to have focus.")
        return poll_pyglet

    def poll_both():
        if device.getKeys(keyList=QUIT_KEYS, waitRelease=False):
            return True
        return poll_pyglet()

    return poll_both


if __name__ == "__main__":
    raise SystemExit(main())