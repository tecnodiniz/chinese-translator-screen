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

A floating toolbar has pause/resume, resize (choose a new area), font size -/+,
line spacing -/+ and quit.
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
FONT_SCALE_MIN, FONT_SCALE_MAX, FONT_SCALE_STEP = 0.5, 3.0, 0.1
LINE_SPACING_MIN, LINE_SPACING_MAX, LINE_SPACING_STEP = 1.0, 3.0, 0.1


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
    fg: str  # text colour, "#rrggbb"


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
        return merge_split_lines(lines)


def merge_split_lines(lines: list[Line]) -> list[Line]:
    """Text detection sometimes cuts one line into overlapping pieces, e.g. "系统将在5分钟后自动重启，请"
    + "请保存您的工作". Join pieces that sit on the same row and overlap/touch, so the sentence is
    translated (and drawn) as a whole. Separate items such as menu entries have clear gaps and stay apart."""
    out: list[Line] = []
    for ln in sorted(lines, key=lambda line: line.x0):
        for i, m in enumerate(out):
            h = min(m.y1 - m.y0, ln.y1 - ln.y0)
            same_row = min(m.y1, ln.y1) - max(m.y0, ln.y0) > 0.6 * h
            if same_row and ln.x0 - m.x1 < 0.25 * h:
                if ln.text in m.text:
                    text = m.text
                elif m.text in ln.text:
                    text = ln.text
                else:  # drop characters read twice where the boxes overlap
                    k = next((k for k in range(min(3, len(m.text), len(ln.text)), 0, -1)
                              if ln.x0 < m.x1 and m.text.endswith(ln.text[:k])), 0)
                    text = m.text + ln.text[k:]
                colors = (m.bg, m.fg) if len(m.text) >= len(ln.text) else (ln.bg, ln.fg)
                out[i] = Line(m.x0, min(m.y0, ln.y0), max(m.x1, ln.x1), max(m.y1, ln.y1), text, *colors)
                break
        else:
            out.append(ln)
    return sorted(out, key=lambda line: (line.y0, line.x0))


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
    """capture -> detect change -> OCR -> translate, forever. Publishes `(area, items)` to `out`,
    tagged with the area they were computed for. Assign `worker.area` to move to a new area."""

    def __init__(self, area: dict, args: argparse.Namespace, out: queue.Queue,
                 stop: threading.Event, paused: threading.Event):
        super().__init__(daemon=True)
        self.area, self.args, self.out, self.stop, self.paused = area, args, out, stop, paused
        self.translations: OrderedDict[str, str] = OrderedDict()
        self.ready = threading.Event()  # set once the models are loaded
        self.failed = False

    def publish(self, area: dict, items: list[Item]) -> None:
        try:
            self.out.get_nowait()  # keep only the newest result
        except queue.Empty:
            pass
        self.out.put((area, items))

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
        self.ready.set()

        prev, prev_area = None, None
        with mss.MSS() as sct:
            while not self.stop.is_set():
                started = time.monotonic()
                if self.paused.is_set():
                    prev = None  # re-scan as soon as we resume
                    self.stop.wait(0.2)
                    continue

                area = self.area  # read once: the toolbar may swap it at any time
                if area is not prev_area:
                    prev, prev_area = None, area
                frame = np.ascontiguousarray(np.asarray(sct.grab(area))[:, :, :3])  # BGRA -> BGR

                # Cheap change detection on a 1/16 sample: skip OCR while the screen is static.
                small = frame[::4, ::4, 1].astype(np.int16)
                if prev is not None:
                    changed = np.count_nonzero(np.abs(small - prev) > 24)
                    if changed <= 8:
                        self.stop.wait(self.args.interval)
                        continue
                    if changed > 0.05 * small.size:  # big change (scroll, new page): drop stale overlay now
                        self.publish(area, [])
                prev = small

                lines = ocr.read(frame)
                todo = list(dict.fromkeys(ln.text for ln in lines if ln.text not in self.translations))
                if todo:
                    for zh, en in zip(todo, translator.translate(todo)):
                        self.translations[zh] = en
                        print(f"{zh}  ->  {en}", flush=True)
                    while len(self.translations) > 5000:
                        self.translations.popitem(last=False)

                self.publish(area, [Item(ln.x0, ln.y0, ln.x1, ln.y1, en, ln.bg, ln.fg)
                                    for ln in lines if (en := self.translations.get(ln.text))])
                self.stop.wait(max(0.0, self.args.interval - (time.monotonic() - started)))


# --------------------------------------------------------------------------- Windows helpers

def enable_dpi_awareness() -> None:
    """Make screen capture and the windows all use physical pixels. Must run before Tk/mss start."""
    if not IS_WINDOWS:
        return
    user32 = ctypes.windll.user32
    if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):  # per-monitor aware v2
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        user32.SetProcessDPIAware()


def setup_window(hwnd: int, click_through: bool) -> bool:
    """No taskbar/Alt-Tab entry, never takes keyboard focus (so the VM keeps it), optionally
    click-through, and excluded from screen capture so it is never fed back into the OCR.
    Returns False if the capture exclusion failed."""
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.restype = ctypes.c_long
    user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    GWL_EXSTYLE = -20
    WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_LAYERED, WS_EX_NOACTIVATE = 0x20, 0x80, 0x80000, 0x08000000
    extra = WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | (WS_EX_TRANSPARENT | WS_EX_LAYERED if click_through else 0)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, user32.GetWindowLongW(hwnd, GWL_EXSTYLE) | extra)
    WDA_EXCLUDEFROMCAPTURE = 0x11  # Windows 10 2004 or newer
    return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))


def keep_on_top(hwnd: int) -> None:
    """Re-assert always-on-top WITHOUT activating the window (that would steal focus from the VM)."""
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    HWND_TOPMOST, SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = -1, 0x1, 0x2, 0x10
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)


def toggle_pause(paused: threading.Event) -> None:
    if paused.is_set():
        paused.clear()
        print("Resumed.", flush=True)
    else:
        paused.set()
        print("Paused.", flush=True)


def start_hotkeys(stop: threading.Event, paused: threading.Event) -> None:
    """Global hotkeys: Ctrl+Alt+Shift+T pause/resume, Ctrl+Alt+Shift+Q quit."""
    from ctypes import wintypes

    def loop() -> None:
        user32 = ctypes.windll.user32
        mods = 0x1 | 0x2 | 0x4 | 0x4000  # Alt | Ctrl | Shift | no auto-repeat
        if not (user32.RegisterHotKey(None, 1, mods, ord("T")) and user32.RegisterHotKey(None, 2, mods, ord("Q"))):
            print("Warning: could not register the hotkeys (in use by another app?). Use the toolbar or Ctrl+C.")
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == 0x0312 and msg.wParam == 1:  # WM_HOTKEY
                toggle_pause(paused)
            elif msg.message == 0x0312 and msg.wParam == 2:
                stop.set()

    threading.Thread(target=loop, daemon=True).start()


# ----------------------------------------------------------------------------------- overlay

def same_row(a: Item, b: Item) -> bool:
    return min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5 * min(a.y1 - a.y0, b.y1 - b.y0)


def wrap_text(text: str, font, width: float) -> list[str]:
    """Split text into lines no wider than `width` pixels (breaking between words)."""
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and font.measure(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current] if current else (lines or [text])


@dataclass
class Block:
    """Where one translation is drawn: its font, wrapped lines and position."""
    item: Item
    font: object
    lines: list[str]
    top: float     # top of the first line
    pitch: float   # distance between the tops of consecutive lines (line height x spacing)
    line_h: int
    x0: int        # horizontal extent of original box + English text
    x1: int

    @property
    def bottom(self) -> float:
        return self.top + (len(self.lines) - 1) * self.pitch + self.line_h


class Overlay:
    """Transparent always-on-top window covering the capture area; draws the translations."""

    MIN_PX, MAX_PX = 11, 40

    def __init__(self, root, area: dict, opacity: float, scale: float, spacing: float, screen_bottom: int):
        import tkinter as tk
        import tkinter.font as tkfont

        self.root, self.tkfont = root, tkfont
        root.overrideredirect(True)
        root.configure(bg=TRANSPARENT_KEY)
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", TRANSPARENT_KEY)
        root.attributes("-alpha", opacity)
        self.canvas = tk.Canvas(root, bg=TRANSPARENT_KEY, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.fonts: dict[int, object] = {}
        self.items: list[Item] | None = []
        self.scale, self.spacing = scale, spacing
        self.screen_bottom = screen_bottom
        self.move(area)
        root.deiconify()
        root.update()
        self.hwnd = int(root.wm_frame(), 16)
        self.capture_excluded = setup_window(self.hwnd, click_through=True)

    def move(self, area: dict) -> None:
        """Cover a new capture area (old translations are cleared). The window reaches down to the
        bottom of the screen, so rows pushed below the area by a big font / spacing stay readable."""
        self.area = area
        height = max(area["height"], self.screen_bottom - area["top"])
        self.root.geometry(f"{area['width']}x{height}+{area['left']}+{area['top']}")
        self.root.update_idletasks()  # apply now (see Toolbar.place_near)
        self.canvas.delete("all")
        self.items = []

    def set_scale(self, scale: float) -> None:
        self.scale = round(min(FONT_SCALE_MAX, max(FONT_SCALE_MIN, scale)), 2)
        self.redraw()

    def set_spacing(self, spacing: float) -> None:
        self.spacing = round(min(LINE_SPACING_MAX, max(LINE_SPACING_MIN, spacing)), 2)
        self.redraw()

    def redraw(self) -> None:
        items, self.items = self.items, None
        self.render(items or [])

    def _font(self, px: int):
        if px not in self.fonts:
            self.fonts[px] = self.tkfont.Font(root=self.root, family="Segoe UI Semibold", size=-px)
        return self.fonts[px]

    def _fit(self, it: Item, items: list[Item]):
        """Font size and wrapped lines for one translation."""
        s = self.scale
        min_px, max_px = max(8, round(self.MIN_PX * s)), round(self.MAX_PX * s)
        w, h = it.x1 - it.x0, it.y1 - it.y0
        # Free space to the right: up to the next text on the same row, or the edge of the area.
        right = min((o.x0 for o in items if o is not it and o.x0 >= it.x1 - 2 and same_row(o, it)),
                    default=self.area["width"])
        limit = max(w, right - it.x0 - 4)
        room = max(w, min(w * s, limit))  # wider than the original only when the font is enlarged
        # Largest font that fits (English is usually longer than Chinese).
        px = max(min_px, min(max_px, int(h * 0.7 * s)))
        while px > min_px and self._font(px).measure(it.text) > room:
            px -= 1
        font = self._font(px)
        if font.measure(it.text) <= room:
            return font, [it.text]
        return font, wrap_text(it.text, font, max(room, min(240 * s, limit)))  # too long: wrap

    def render(self, items: list[Item]) -> None:
        if items == self.items:
            return  # unchanged: don't redraw (avoids flicker)
        self.items = items

        # Group translations that sit side by side (table cells, menu entries) into rows.
        rows: list[list[Item]] = []
        for it in sorted(items, key=lambda i: (i.y0, i.x0)):
            row = next((r for r in rows if any(same_row(it, o) for o in r)), None)
            if row is None:
                rows.append([it])
            else:
                row.append(it)

        # Lay out row by row, top to bottom. A translation that needs more height than its original
        # line (big font, wrapping, extra line spacing) pushes the rows below it down instead of
        # overlapping them. A whole row moves together, so table columns stay aligned.
        blocks: list[Block] = []
        for row in rows:
            placed, push = [], 0.0
            for it in row:
                font, lines = self._fit(it, items)
                line_h = font.metrics("linespace")
                x1 = max(it.x1, it.x0 + 1 + max(font.measure(line) for line in lines))
                top = max(0.0, (it.y0 + it.y1) / 2 - line_h / 2)  # first line centred on the original
                for b in blocks:
                    if b.x0 < x1 and it.x0 < b.x1:  # shares columns with a translation placed above
                        push = max(push, b.top + len(b.lines) * b.pitch - top)
                placed.append(Block(it, font, lines, top, line_h * self.spacing, line_h, it.x0, x1))
            for b in placed:
                b.top += push
            blocks.extend(placed)

        c = self.canvas
        c.delete("all")
        for b in blocks:  # 1) backgrounds: hide the original Chinese and make room for the English
            c.create_rectangle(b.x0 - 1, min(b.item.y0, b.top), b.x1 + 2, max(b.item.y1, b.bottom),
                               fill=b.item.bg, outline="")
        for b in blocks:  # 2) all text above all backgrounds, so no translation can hide another
            for k, line in enumerate(b.lines):
                c.create_text(b.item.x0 + 1, b.top + k * b.pitch, text=line, font=b.font,
                              fill=b.item.fg, anchor="nw")


# ----------------------------------------------------------------------------------- toolbar

class Toolbar:
    """Floating, draggable control bar: pause/resume, resize (pick a new area), font size -/+,
    line spacing -/+, quit.
    It stays on top, never takes keyboard focus away from the VM and is hidden from screen capture."""

    BG, BUTTON, HOVER, TEXT, BORDER = "#202124", "#303134", "#4a4c50", "#e8eaed", "#5f6368"
    STATE_COLORS = {"loading": "#8ab4f8", "running": "#34a853", "paused": "#f9ab00"}

    def __init__(self, app: App):
        import tkinter as tk

        self.tk, self.app = tk, app
        self.win = win = tk.Toplevel(app.root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.configure(bg=self.BORDER)
        bar = tk.Frame(win, bg=self.BG)
        bar.pack(padx=1, pady=1)

        grip = tk.Label(bar, text="⠿", bg=self.BG, fg="#9aa0a6", font=("Segoe UI Symbol", 13), cursor="fleur")
        grip.pack(side="left", padx=(6, 2))
        self.dot = tk.Label(bar, text="●", bg=self.BG, fg=self.STATE_COLORS["loading"], font=("Segoe UI", 9))
        self.dot.pack(side="left", padx=(0, 4))
        self.pause_button = self._button(bar, "❚❚  Pause", app.toggle_pause, width=9)
        self._button(bar, "⤢  Resize", app.reselect_area)
        self._button(bar, "A−", lambda: app.change_font(-1), width=3, pad_left=8)
        self.scale_label = tk.Label(bar, bg=self.BG, fg=self.TEXT, font=("Segoe UI", 10), width=5)
        self.scale_label.pack(side="left")
        self._button(bar, "A+", lambda: app.change_font(+1), width=3)
        self._button(bar, "↕−", lambda: app.change_spacing(-1), width=3, pad_left=8)
        self.spacing_label = tk.Label(bar, bg=self.BG, fg=self.TEXT, font=("Segoe UI", 10), width=4)
        self.spacing_label.pack(side="left")
        self._button(bar, "↕+", lambda: app.change_spacing(+1), width=3)
        self._button(bar, "✕", app.quit, width=3, hover="#c5221f", pad_left=8)

        for widget in (bar, grip, self.dot, self.scale_label, self.spacing_label):  # drag by non-buttons
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._drag)
        self.moved_by_user = False
        self.state: str | None = None
        self.show_scale(app.overlay.scale)
        self.show_spacing(app.overlay.spacing)
        win.update()
        self.hwnd = int(win.wm_frame(), 16)
        setup_window(self.hwnd, click_through=False)

    def _button(self, parent, text: str, command, width: int | None = None, hover: str | None = None,
                pad_left: int = 2):
        b = self.tk.Button(parent, text=text, command=command, width=width, font=("Segoe UI", 10),
                           bg=self.BUTTON, fg=self.TEXT, activebackground=hover or self.HOVER,
                           activeforeground=self.TEXT, relief="flat", bd=0, padx=8, pady=3,
                           highlightthickness=0, cursor="hand2")
        b.bind("<Enter>", lambda e: b.configure(bg=hover or self.HOVER))
        b.bind("<Leave>", lambda e: b.configure(bg=self.BUTTON))
        b.pack(side="left", padx=(pad_left, 2), pady=4)
        return b

    def _start_drag(self, e) -> None:
        self._offset = (e.x_root - self.win.winfo_x(), e.y_root - self.win.winfo_y())

    def _drag(self, e) -> None:
        self.win.geometry(f"+{e.x_root - self._offset[0]}+{e.y_root - self._offset[1]}")
        self.win.update_idletasks()
        self.moved_by_user = True

    def place_near(self, area: dict, screen_top: int) -> None:
        """Centre the bar just above the area, or just inside its top edge if there is no room above."""
        self.win.update_idletasks()
        w, h = self.win.winfo_reqwidth(), self.win.winfo_reqheight()
        x = area["left"] + (area["width"] - w) // 2
        y = area["top"] - h - 6
        if y < screen_top:
            y = area["top"] + 6
        self.win.geometry(f"+{x}+{y}")
        # Apply the move right away: if Windows reports the old position first (e.g. after a
        # "stay on top" refresh), Tk would otherwise drop the pending move.
        self.win.update_idletasks()

    def show_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.dot.configure(fg=self.STATE_COLORS[state])
            self.pause_button.configure(text="▶  Resume" if state == "paused" else "❚❚  Pause")

    def show_scale(self, scale: float) -> None:
        self.scale_label.configure(text=f"{round(scale * 100)}%")

    def show_spacing(self, spacing: float) -> None:
        self.spacing_label.configure(text=f"{spacing:.1f}×")


# --------------------------------------------------------------------------------------- app

class App:
    """Main (Tk) thread: owns the overlay and toolbar and connects them to the worker thread."""

    def __init__(self, root, area: dict, monitors: list[dict], args: argparse.Namespace, worker: Worker,
                 results: queue.Queue, stop: threading.Event, paused: threading.Event):
        self.root, self.monitors, self.worker = root, monitors, worker
        self.results, self.stop, self.paused = results, stop, paused
        self.selecting = False
        self.overlay = Overlay(root, area, max(0.2, min(1.0, args.opacity)),
                               min(FONT_SCALE_MAX, max(FONT_SCALE_MIN, args.font_scale)),
                               min(LINE_SPACING_MAX, max(LINE_SPACING_MIN, args.line_spacing)),
                               screen_bottom=monitors[0]["top"] + monitors[0]["height"])
        if not self.overlay.capture_excluded:
            print("Warning: this Windows version can't hide the overlay from screen capture; "
                  "translations may flicker.")
        self.toolbar = None
        if not args.no_toolbar:
            self.toolbar = Toolbar(self)
            self.toolbar.place_near(area, monitors[0]["top"])
        self.last_on_top = time.monotonic()

    # --- toolbar actions
    def toggle_pause(self) -> None:
        toggle_pause(self.paused)

    def change_font(self, direction: int) -> None:
        self.overlay.set_scale(self.overlay.scale + direction * FONT_SCALE_STEP)
        if self.toolbar:
            self.toolbar.show_scale(self.overlay.scale)

    def change_spacing(self, direction: int) -> None:
        self.overlay.set_spacing(self.overlay.spacing + direction * LINE_SPACING_STEP)
        if self.toolbar:
            self.toolbar.show_spacing(self.overlay.spacing)

    def quit(self) -> None:
        self.stop.set()

    def reselect_area(self) -> None:
        """Dim the screen, let the user drag a new area, then move the overlay and capture there."""
        if self.selecting:
            return
        self.selecting = True
        was_paused = self.paused.is_set()
        self.paused.set()  # stop capturing while the dimmer is on screen
        self.overlay.render([])
        try:
            area = select_region(self.root, self.monitors)
        finally:
            self.selecting = False
        if self.stop.is_set():
            return
        if area:
            self.overlay.move(area)
            self.worker.area = area
            if self.toolbar and not self.toolbar.moved_by_user:
                self.toolbar.place_near(area, self.monitors[0]["top"])
            print(f"Area: {area['width']}x{area['height']} at ({area['left']}, {area['top']})", flush=True)
        if not was_paused:
            self.paused.clear()

    # --- main loop
    def poll(self) -> None:
        if self.stop.is_set() and not self.selecting:
            self.root.destroy()
            return
        if self.paused.is_set():
            self.overlay.render([])
        else:
            try:
                area, items = self.results.get_nowait()
            except queue.Empty:
                pass
            else:
                if area == self.overlay.area:  # ignore results computed for an area we just left
                    self.overlay.render(items)
        if self.toolbar:
            self.toolbar.show_state("paused" if self.paused.is_set() and not self.selecting
                                    else "running" if self.worker.ready.is_set() else "loading")
        if not self.selecting and time.monotonic() - self.last_on_top > 2:
            # Stay above windows that grabbed "topmost" later; toolbar above the overlay.
            keep_on_top(self.overlay.hwnd)
            if self.toolbar:
                keep_on_top(self.toolbar.hwnd)
            self.last_on_top = time.monotonic()
        self.root.after(50, self.poll)


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
    root.update()  # make sure the dimmer is gone before the next capture
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
    p.add_argument("--font-scale", type=float, default=1.0,
                   help="starting font size of the translations, 0.5-3 (default 1 = fit the original text)")
    p.add_argument("--line-spacing", type=float, default=1.0,
                   help="starting space between translated rows, 1-3 x the text height (default 1)")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="CTranslate2 translation model folder")
    p.add_argument("--no-toolbar", action="store_true", help="hide the floating toolbar (hotkeys still work)")
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
        app = App(root, area, monitors, args, worker, results, stop, paused)
        app.poll()
        root.mainloop()

    stop.set()
    if worker.failed:
        sys.exit(1)
    print("Bye.")


if __name__ == "__main__":
    main()
