"""
Offline realtime screen translator - Chinese -> English.

Watches part of the screen (for example a VM window), reads Chinese text with OCR and
paints the English translation right on top of it in a transparent, click-through
overlay - like pointing a phone camera translator at your monitor.

Everything runs locally: OCR uses RapidOCR (PP-OCR models shipped inside the package) and
translation uses an OPUS-MT model through CTranslate2 (models/opus-mt-zho-eng), with
exact-match overrides from glossary.txt.
Nothing is sent over the network.

Run it with the project's Windows venv (from WSL: .venv/Scripts/python.exe translator.py ...):
    translator.py --select       drag a rectangle around the VM window (recommended, faster)
    translator.py                whole primary monitor
    translator.py --monitor 2    another monitor
    translator.py --no-overlay   only print the translations in the console

Hotkeys: Ctrl+Alt+Shift+T pause/resume, Ctrl+Alt+Shift+Q quit (Ctrl+C in the console also quits).
A VM window that has keyboard focus may swallow the hotkeys - click outside it first.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import queue
import re
import signal
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "opus-mt-zho-eng"
GLOSSARY = HERE / "glossary.txt"
IS_WINDOWS = sys.platform == "win32"
CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
TRANSPARENT_KEY = "#010203"  # overlay pixels of exactly this colour are see-through


# --------------------------------------------------------------------------------------- OCR

@dataclass(frozen=True)
class Line:
    """One line of Chinese text found on screen (coordinates relative to the capture area)."""
    x0: int
    y0: int
    x1: int
    y1: int
    text: str
    bg: str  # background colour around the text, "#rrggbb"
    fg: str  # black or white, whichever contrasts with bg


def sample_colors(crop: np.ndarray) -> tuple[str, str]:
    """Background colour (median of the crop's border) and text colour (the pixels that differ most
    from the background), so red errors stay red, links stay blue, etc."""
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]).astype(int)
    bg_bgr = np.median(border, axis=0)
    pixels = crop.reshape(-1, 3).astype(int)
    dist = np.abs(pixels - bg_bgr).sum(axis=1)
    if dist.max() >= 150:  # text clearly stands out: use its real colour
        fg_bgr = np.median(pixels[dist >= np.percentile(dist, 92)], axis=0)
    else:  # low contrast / unclear: black or white, whichever is readable
        lum = 0.114 * bg_bgr[0] + 0.587 * bg_bgr[1] + 0.299 * bg_bgr[2]
        fg_bgr = np.zeros(3) if lum > 140 else np.full(3, 255)
    bg, fg = (f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}" for c in (bg_bgr, fg_bgr))  # BGR -> #rrggbb
    return ("#000000" if bg == TRANSPARENT_KEY else bg), ("#000001" if fg == TRANSPARENT_KEY else fg)


class ScreenOCR:
    """Runs text detection on every frame, but text recognition only on boxes whose pixels have not
    been seen before. On a mostly static screen that skips ~75% of the OCR work."""

    def __init__(self, min_score: float, cache_size: int = 5000):
        from rapidocr import RapidOCR

        self.engine = RapidOCR(params={"Global.use_cls": False, "Global.log_level": "error"})
        self.min_score = min_score
        self.cache_size = cache_size
        self.cache: OrderedDict[bytes, tuple[str, float]] = OrderedDict()  # crop hash -> (text, score)

    def read(self, frame: np.ndarray) -> list[Line]:
        det = self.engine(frame, use_det=True, use_cls=False, use_rec=False)
        boxes = getattr(det, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        h, w = frame.shape[:2]
        found = []
        unseen: dict[bytes, np.ndarray] = {}
        for box in boxes:
            x0, y0 = np.floor(box.min(axis=0)).astype(int)
            x1, y1 = np.ceil(box.max(axis=0)).astype(int)
            x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), w), min(int(y1), h)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            crop = frame[y0:y1, x0:x1]
            key = hashlib.blake2b(crop.tobytes(), digest_size=16).digest()
            found.append(((x0, y0, x1, y1), key, crop))
            if key not in self.cache:
                unseen[key] = crop

        if unseen:
            # Tall boxes are vertical text: rotate them like RapidOCR does internally.
            crops = [np.ascontiguousarray(np.rot90(c) if c.shape[0] >= 1.5 * c.shape[1] else c)
                     for c in unseen.values()]
            try:
                rec = self.engine.recognize_txt(crops)
                results = list(zip(rec.txts, rec.scores))
            except Exception:  # RapidOCR raises when nothing at all could be recognised
                results = [("", 0.0)] * len(crops)
            for key, (text, score) in zip(unseen, results):
                self.cache[key] = (text.strip(), float(score))

        lines = []
        for (x0, y0, x1, y1), key, crop in found:
            text, score = self.cache[key]
            self.cache.move_to_end(key)
            if score >= self.min_score and CJK.search(text):
                lines.append(Line(x0, y0, x1, y1, text, *sample_colors(crop)))
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return lines


# ------------------------------------------------------------------------------- translation

# The normalisation the OPUS-MT model was trained with (its preprocess.sh): full-width -> ASCII.
_FULLWIDTH = str.maketrans({
    "，": ",", "、": ",", "“": '"', "”": '"', "∶": ":", "：": ":", "？": "?", "《": '"', "》": '"',
    "（": "(", "）": ")", "！": "!", "；": ";", "「": '"', "」": '"', "～": "~", "’": "'", "━": "-",
    "〈": "<", "〉": ">", "【": "[", "】": "]", "％": "%",
    **{chr(0xFF10 + i): str(i) for i in range(10)},
})
# The model sometimes prepends a training label such as ">>cmn_Hans<<", decoded as "⁇ cmn_Hans ⁇".
_LABEL = re.compile(r"⁇\s*[\w-]*\s*⁇|⁇")
_SENTENCE_END = ("。", ".", "！", "!", "？", "?")


def load_glossary(path: Path) -> dict[str, str]:
    """Read `中文 = English` lines (UTF-8, '#' starts a comment line)."""
    glossary: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                zh, en = (part.strip() for part in line.split("=", 1))
                if zh and en:
                    glossary[zh] = en
    return glossary


class OfflineTranslator:
    """Chinese -> English with a local OPUS-MT model (CTranslate2, int8, CPU).
    Lines that exactly match a glossary entry use the glossary instead of the model."""

    def __init__(self, model_dir: Path, glossary: dict[str, str] | None = None):
        import ctranslate2
        import sentencepiece as spm

        self.model = ctranslate2.Translator(str(model_dir), device="cpu", compute_type="int8")
        self.sp_src = spm.SentencePieceProcessor(model_file=str(model_dir / "source.spm"))
        self.sp_tgt = spm.SentencePieceProcessor(model_file=str(model_dir / "target.spm"))
        self.glossary = glossary or {}

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.translate(_FULLWIDTH).replace("。", ". ").replace("．", ". ").replace("…", "...")
        return re.sub(r"\s+", " ", text).strip()

    def translate(self, texts: list[str]) -> list[str]:
        out: list[str | None] = [self.glossary.get(t.strip()) for t in texts]
        todo = [i for i, en in enumerate(out) if en is None]
        if todo:
            batch = [self.sp_src.encode(self._normalize(texts[i]), out_type=str) for i in todo]
            results = self.model.translate_batch(batch, beam_size=4, max_decoding_length=256, max_batch_size=32)
            for i, res in zip(todo, results):
                en = _LABEL.sub("", self.sp_tgt.decode(res.hypotheses[0])).strip()
                # Short UI labels should not get a full stop: "确定" -> "OK", not "OK."
                if en.endswith(".") and not en.endswith("..") and not texts[i].rstrip().endswith(_SENTENCE_END):
                    en = en[:-1]
                out[i] = en
        return out


# ------------------------------------------------------------------------------------ worker

@dataclass(frozen=True)
class Item:
    """A translated line to draw, in overlay (= capture area) coordinates."""
    x0: int
    y0: int
    x1: int
    y1: int
    text: str
    bg: str
    fg: str


class Worker(threading.Thread):
    """capture -> detect change -> OCR -> translate, forever. Publishes the latest Items to `out`."""

    def __init__(self, area: dict, args: argparse.Namespace, out: queue.Queue,
                 stop: threading.Event, paused: threading.Event):
        super().__init__(daemon=True)
        self.area, self.args, self.out, self.stop, self.paused = area, args, out, stop, paused
        self.translations: OrderedDict[str, str] = OrderedDict()
        self.failed = False

    def publish(self, items: list[Item]) -> None:
        try:
            self.out.get_nowait()  # keep only the newest result
        except queue.Empty:
            pass
        self.out.put(items)

    def run(self) -> None:
        try:
            self._loop()
        except BaseException:
            self.failed = True
            self.stop.set()
            raise

    def _loop(self) -> None:
        import mss

        print("Loading OCR and translation models (offline) ...", flush=True)
        ocr = ScreenOCR(self.args.min_score)
        glossary = load_glossary(GLOSSARY)
        translator = OfflineTranslator(Path(self.args.model), glossary)
        translator.translate(["你好"])  # warm-up
        print(f"Glossary: {len(glossary)} entries from {GLOSSARY.name}", flush=True)
        print("Ready - translating. New lines are printed below.\n", flush=True)

        prev = None
        with mss.MSS() as sct:
            while not self.stop.is_set():
                started = time.monotonic()
                if self.paused.is_set():
                    prev = None  # re-scan as soon as we resume
                    self.stop.wait(0.2)
                    continue

                frame = np.ascontiguousarray(np.asarray(sct.grab(self.area))[:, :, :3])  # BGRA -> BGR

                # Cheap change detection on a 1/16 sample: skip OCR while the screen is static.
                small = frame[::4, ::4, 1].astype(np.int16)
                if prev is not None:
                    changed = np.count_nonzero(np.abs(small - prev) > 24)
                    if changed <= 8:
                        self.stop.wait(self.args.interval)
                        continue
                    if changed > 0.05 * small.size:  # big change (scroll, new page): drop stale overlay now
                        self.publish([])
                prev = small

                lines = ocr.read(frame)
                todo = list(dict.fromkeys(ln.text for ln in lines if ln.text not in self.translations))
                if todo:
                    for zh, en in zip(todo, translator.translate(todo)):
                        self.translations[zh] = en
                        print(f"{zh}  ->  {en}", flush=True)
                    while len(self.translations) > 5000:
                        self.translations.popitem(last=False)

                self.publish([Item(ln.x0, ln.y0, ln.x1, ln.y1, en, ln.bg, ln.fg)
                              for ln in lines if (en := self.translations.get(ln.text))])
                self.stop.wait(max(0.0, self.args.interval - (time.monotonic() - started)))


# --------------------------------------------------------------------------- Windows helpers

def enable_dpi_awareness() -> None:
    """Make screen capture and the overlay both use physical pixels. Must run before Tk/mss start."""
    if not IS_WINDOWS:
        return
    user32 = ctypes.windll.user32
    if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):  # per-monitor aware v2
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        user32.SetProcessDPIAware()


def setup_overlay_window(hwnd: int) -> bool:
    """Click-through, no taskbar/Alt-Tab entry, never takes focus, and excluded from screen capture
    so our own English text is never fed back into the OCR. Returns False if the exclusion failed."""
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    GWL_EXSTYLE = -20
    WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_LAYERED, WS_EX_NOACTIVATE = 0x20, 0x80, 0x80000, 0x08000000
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                          style | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_NOACTIVATE)
    WDA_EXCLUDEFROMCAPTURE = 0x11  # Windows 10 2004 or newer
    return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))


def start_hotkeys(stop: threading.Event, paused: threading.Event) -> None:
    """Global hotkeys: Ctrl+Alt+Shift+T pause/resume, Ctrl+Alt+Shift+Q quit."""
    from ctypes import wintypes

    def loop() -> None:
        user32 = ctypes.windll.user32
        mods = 0x1 | 0x2 | 0x4 | 0x4000  # Alt | Ctrl | Shift | no auto-repeat
        if not (user32.RegisterHotKey(None, 1, mods, ord("T")) and user32.RegisterHotKey(None, 2, mods, ord("Q"))):
            print("Warning: could not register the hotkeys (in use by another app?). Use Ctrl+C to quit.")
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == 0x0312 and msg.wParam == 1:  # WM_HOTKEY
                if paused.is_set():
                    paused.clear()
                    print("Resumed.", flush=True)
                else:
                    paused.set()
                    print("Paused (Ctrl+Alt+Shift+T to resume).", flush=True)
            elif msg.message == 0x0312 and msg.wParam == 2:
                stop.set()

    threading.Thread(target=loop, daemon=True).start()


# ----------------------------------------------------------------------------------- overlay

class Overlay:
    """Transparent always-on-top window covering the capture area; draws the translations."""

    MIN_PX, MAX_PX = 11, 40

    def __init__(self, root, area: dict, opacity: float):
        import tkinter as tk
        import tkinter.font as tkfont

        self.root, self.tkfont = root, tkfont
        root.overrideredirect(True)
        root.geometry(f"{area['width']}x{area['height']}+{area['left']}+{area['top']}")
        root.configure(bg=TRANSPARENT_KEY)
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", TRANSPARENT_KEY)
        root.attributes("-alpha", opacity)
        self.canvas = tk.Canvas(root, bg=TRANSPARENT_KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        root.deiconify()
        root.update()
        self.capture_excluded = setup_overlay_window(int(root.wm_frame(), 16))
        self.fonts: dict[int, object] = {}
        self.items: list[Item] = []

    def _font(self, px: int):
        if px not in self.fonts:
            self.fonts[px] = self.tkfont.Font(root=self.root, family="Segoe UI Semibold", size=-px)
        return self.fonts[px]

    def render(self, items: list[Item]) -> None:
        if items == self.items:
            return  # unchanged: don't redraw (avoids flicker)
        self.items = items
        c = self.canvas
        c.delete("all")
        for it in items:
            w, h = it.x1 - it.x0, it.y1 - it.y0
            # Largest font that fits the original box width (English is usually longer than Chinese).
            px = max(self.MIN_PX, min(self.MAX_PX, int(h * 0.7)))
            while px > self.MIN_PX and self._font(px).measure(it.text) > w:
                px -= 1
            font = self._font(px)
            if font.measure(it.text) <= w:
                tid = c.create_text(it.x0 + 1, (it.y0 + it.y1) // 2, text=it.text, font=font,
                                    fill=it.fg, anchor="w")
            else:  # still too long at the minimum size: wrap it
                tid = c.create_text(it.x0 + 1, it.y0, text=it.text, font=font, fill=it.fg,
                                    anchor="nw", width=max(w, 240))
            tx0, ty0, tx1, ty1 = c.bbox(tid)
            rid = c.create_rectangle(min(it.x0, tx0 - 2), min(it.y0, ty0), max(it.x1, tx1 + 2), max(it.y1, ty1),
                                     fill=it.bg, outline="")
            c.tag_lower(rid, tid)  # background box (hides the Chinese) goes under its text


def select_region(root, monitors: list[dict]) -> dict | None:
    """Dim the screen and let the user drag a rectangle. Returns the area in screen pixels."""
    import tkinter as tk

    v = monitors[0]  # bounding box of all monitors
    top = tk.Toplevel(root)
    top.overrideredirect(True)
    top.geometry(f"{v['width']}x{v['height']}+{v['left']}+{v['top']}")
    top.attributes("-topmost", True)
    top.attributes("-alpha", 0.4)
    canvas = tk.Canvas(top, bg="black", highlightthickness=0, cursor="crosshair")
    canvas.pack(fill="both", expand=True)
    p = next((m for m in monitors[1:] if m.get("is_primary")), monitors[1] if len(monitors) > 1 else v)
    canvas.create_text(p["left"] - v["left"] + p["width"] // 2, p["top"] - v["top"] + 60,
                       text="Drag a rectangle around the area to translate   (Esc = cancel)",
                       fill="white", font=("Segoe UI", 20, "bold"))
    rect = canvas.create_rectangle(0, 0, 0, 0, outline="#ff3030", width=3, fill="white")
    state: dict = {}

    def press(e):
        state["start"] = (e.x, e.y)

    def drag(e):
        if "start" in state:
            canvas.coords(rect, *state["start"], e.x, e.y)

    def release(e):
        if "start" in state:
            (sx, sy), ex, ey = state["start"], e.x, e.y
            x0, x1 = sorted((sx, ex))
            y0, y1 = sorted((sy, ey))
            if x1 - x0 >= 20 and y1 - y0 >= 20:
                state["area"] = dict(left=v["left"] + x0, top=v["top"] + y0, width=x1 - x0, height=y1 - y0)
        top.destroy()

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", release)
    canvas.bind("<ButtonPress-3>", lambda e: top.destroy())
    top.bind("<Escape>", lambda e: top.destroy())
    top.update()
    top.focus_force()
    root.wait_window(top)
    root.update()  # make sure the dimmer is gone before the first capture
    return state.get("area")


# -------------------------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline realtime Chinese -> English screen translator.")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--select", action="store_true",
                       help="drag a rectangle to choose the area to translate (e.g. the VM window)")
    where.add_argument("--region", metavar="LEFT,TOP,WIDTH,HEIGHT", help="area to translate, in screen pixels")
    where.add_argument("--monitor", type=int, default=1,
                       help="monitor to translate: 1 = primary (default), 2 = second, ...")
    p.add_argument("--interval", type=float, default=0.25, help="seconds between screen checks (default 0.25)")
    p.add_argument("--min-score", type=float, default=0.6, help="minimum OCR confidence 0-1 (default 0.6)")
    p.add_argument("--opacity", type=float, default=1.0, help="overlay opacity 0.2-1 (default 1)")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="CTranslate2 translation model folder")
    p.add_argument("--no-overlay", action="store_true", help="don't draw on screen, only print translations")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # Chinese in redirected/piped output
        except (AttributeError, ValueError):
            pass
    if not (Path(args.model) / "model.bin").exists():
        sys.exit(f"Translation model not found in {args.model}")
    if not IS_WINDOWS and not args.no_overlay:
        sys.exit("The overlay needs Windows: run this with Windows Python (.venv/Scripts/python.exe) "
                 "or use --no-overlay.")

    enable_dpi_awareness()
    import mss

    with mss.MSS() as sct:
        monitors = sct.monitors

    root = None
    if args.select or not args.no_overlay:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

    if args.select:
        area = select_region(root, monitors)
        if area is None:
            sys.exit("No area selected.")
    elif args.region:
        try:
            left, top, width, height = (int(v) for v in args.region.split(","))
        except ValueError:
            sys.exit("--region must look like LEFT,TOP,WIDTH,HEIGHT, e.g. 100,80,1280,720")
        area = dict(left=left, top=top, width=width, height=height)
    else:
        if not 1 <= args.monitor < len(monitors):
            sys.exit(f"--monitor must be between 1 and {len(monitors) - 1}")
        area = {k: monitors[args.monitor][k] for k in ("left", "top", "width", "height")}

    print(f"Area: {area['width']}x{area['height']} at ({area['left']}, {area['top']})")
    print("Hotkeys: Ctrl+Alt+Shift+T pause/resume, Ctrl+Alt+Shift+Q quit (or Ctrl+C here).")

    stop, paused = threading.Event(), threading.Event()
    results: queue.Queue = queue.Queue(maxsize=1)
    worker = Worker(area, args, results, stop, paused)
    worker.start()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if IS_WINDOWS:
        start_hotkeys(stop, paused)

    if args.no_overlay:
        if root is not None:
            root.destroy()
        while not stop.is_set():
            stop.wait(0.2)
    else:
        overlay = Overlay(root, area, max(0.2, min(1.0, args.opacity)))
        if not overlay.capture_excluded:
            print("Warning: this Windows version can't hide the overlay from screen capture; "
                  "translations may flicker.")
        last_topmost = time.monotonic()

        def poll() -> None:
            nonlocal last_topmost
            if stop.is_set():
                root.destroy()
                return
            if paused.is_set():
                overlay.render([])
            else:
                try:
                    overlay.render(results.get_nowait())
                except queue.Empty:
                    pass
            if time.monotonic() - last_topmost > 2:  # stay above windows that grabbed "topmost" later
                root.attributes("-topmost", True)
                last_topmost = time.monotonic()
            root.after(50, poll)

        poll()
        root.mainloop()

    stop.set()
    if worker.failed:
        sys.exit(1)
    print("Bye.")


if __name__ == "__main__":
    main()
