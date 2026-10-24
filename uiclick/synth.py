"""Procedural UI screenshots with exact ground-truth boxes and *controllable spurious cues*.

Every screenshot has one target element (named by the instruction) and several distractors.
Two cues that carry no information about the instruction can be correlated with the target:

* position cue: target sits in the left half of the screen  (``p_pos``)
* ring cue    : target wears the blue "focus ring"           (``p_ring``)

``None`` means the cue is independent of the target. Train with high p, test with p = 0 and a
model that learned the cue instead of reading the instruction falls apart. Everything is a pure
function of (seed, p_pos, p_ring), so splits are reproducible and disjoint by seed range.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

W, H = 672, 448  # both multiples of 28 -> Qwen2-VL keeps the resolution (24 x 16 = 384 visual tokens)
RING = (0, 110, 255)

BUTTONS = ["Save", "Cancel", "Submit", "Delete", "Next", "Back", "Apply", "Reset", "Download", "Upload", "Send",
           "Open", "Close", "Continue", "Confirm", "Search", "Export", "Import", "Refresh", "Share", "Edit", "Add",
           "Remove", "Log in", "Sign up", "Print", "Copy", "Paste", "Undo", "Redo", "Help", "Finish", "Skip", "Retry"]
LINKS = ["Forgot password?", "Terms of Service", "Privacy Policy", "Learn more", "Contact us", "View all",
         "Skip for now", "Need help?", "Documentation", "Release notes", "Resend code", "Change plan"]
CHECKS = ["Remember me", "Subscribe to newsletter", "I agree to the terms", "Enable notifications", "Dark mode",
          "Auto-save", "Show password", "Keep me signed in", "Share analytics", "Email receipts"]
INPUTS = ["Email", "Password", "Username", "Full name", "Phone number", "Address", "Search...", "City", "Zip code",
          "Company", "Country", "Coupon code", "Job title", "Website"]
TABS = ["Overview", "Activity", "Billing", "Security", "Team", "Reports", "Inbox", "Drafts", "Sent", "Archive"]
HEADINGS = ["Account details", "Welcome back", "Project settings", "Create workspace", "Payment", "Your profile",
            "Notifications", "Checkout", "Invite members", "Preferences"]
ICON_COLORS = {"red": (220, 40, 40), "green": (40, 160, 70), "orange": (245, 140, 20), "purple": (140, 60, 200),
               "yellow": (235, 200, 20), "pink": (240, 100, 170), "teal": (20, 160, 160)}
ICON_SHAPES = ["circle", "square", "triangle", "diamond", "star", "plus", "cross", "hexagon"]

THEMES = [
    dict(bg=(236, 239, 244), panel=(255, 255, 255), text=(28, 30, 36), muted=(150, 156, 168), btn=(226, 230, 236),
         btn_text=(40, 44, 54), border=(196, 202, 212), bar=(222, 226, 233)),
    dict(bg=(18, 20, 25), panel=(33, 36, 44), text=(230, 233, 238), muted=(118, 124, 138), btn=(58, 63, 74),
         btn_text=(236, 238, 242), border=(78, 84, 97), bar=(26, 28, 34)),
    dict(bg=(246, 240, 230), panel=(253, 250, 244), text=(52, 40, 30), muted=(168, 154, 140), btn=(234, 224, 208),
         btn_text=(60, 46, 34), border=(214, 200, 180), bar=(236, 227, 212)),
    dict(bg=(226, 236, 232), panel=(244, 250, 247), text=(22, 48, 40), muted=(130, 156, 148), btn=(208, 226, 219),
         btn_text=(22, 52, 42), border=(176, 204, 194), bar=(210, 226, 220)),
]
FILL_COLORS = [(46, 160, 90), (214, 70, 60), (238, 150, 40), (120, 80, 200), (30, 150, 160)]  # never ring-blue

_FONT_FILES = ["arial.ttf", "segoeui.ttf", "calibri.ttf", "verdana.ttf", "tahoma.ttf", "DejaVuSans.ttf",
               "LiberationSans-Regular.ttf"]
_fonts: dict = {}


def _available_fonts() -> list[str]:
    ok = []
    for f in _FONT_FILES:
        try:
            ImageFont.truetype(f, 12)
            ok.append(f)
        except OSError:
            pass
    return ok or ["<default>"]


FONTS = _available_fonts()


def font(name: str, size: int):
    if (name, size) not in _fonts:
        try:
            _fonts[(name, size)] = ImageFont.truetype(name, size)
        except OSError:
            _fonts[(name, size)] = ImageFont.load_default(size)
    return _fonts[(name, size)]


@dataclass
class Element:
    kind: str       # button | link | checkbox | input | tab | icon
    label: str      # text label, or "<color> <shape>" for icons
    w: int          # layout extent
    h: int
    box_off: tuple  # target box relative to the extent origin
    draw_fn: object = None
    box: tuple = (0, 0, 0, 0)  # absolute pixels, set by the layout

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


@dataclass
class Sample:
    seed: int
    image: Image.Image
    instruction: str
    elements: list
    target: int
    ring: int

    @property
    def box(self) -> tuple:
        """Target box in the 0-1000 normalised coordinates the model is asked to answer in."""
        return norm_box(self.elements[self.target].box)

    @property
    def kind(self) -> str:
        return self.elements[self.target].kind

    @property
    def target_in_left(self) -> bool:
        return self.elements[self.target].center[0] < W / 2


def norm_box(b) -> tuple:
    return (b[0] / W * 1000, b[1] / H * 1000, b[2] / W * 1000, b[3] / H * 1000)


def _star(cx, cy, r):
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _shape(d, shape, box, fill):
    x0, y0, x1, y1 = box
    cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2
    if shape == "circle":
        d.ellipse(box, fill=fill)
    elif shape == "square":
        d.rounded_rectangle(box, 3, fill=fill)
    elif shape == "triangle":
        d.polygon([(cx, y0), (x1, y1), (x0, y1)], fill=fill)
    elif shape == "diamond":
        d.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=fill)
    elif shape == "star":
        d.polygon(_star(cx, cy, r * 1.1), fill=fill)
    elif shape == "plus":
        t = r * 0.38
        d.rectangle((cx - t, y0, cx + t, y1), fill=fill)
        d.rectangle((x0, cy - t, x1, cy + t), fill=fill)
    elif shape == "cross":
        d.line((x0, y0, x1, y1), fill=fill, width=max(3, int(r * 0.45)))
        d.line((x0, y1, x1, y0), fill=fill, width=max(3, int(r * 0.45)))
    elif shape == "hexagon":
        d.polygon([(cx + r * math.cos(a), cy + r * math.sin(a)) for a in [i * math.pi / 3 for i in range(6)]],
                  fill=fill)


def _text(d, xy, s, f, fill):
    bb = d.textbbox((0, 0), s, font=f)
    d.text((xy[0] - bb[0], xy[1] - bb[1]), s, font=f, fill=fill)


def _measure(s, f):
    bb = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), s, font=f)
    return bb[2] - bb[0], bb[3] - bb[1]


def _make_element(rng, kind, label, th, f, fname, fs):
    tw, thh = _measure(label, f) if kind != "icon" else (0, 0)
    if kind == "button":
        style = rng.choice(["flat", "flat", "outline", "fill"])
        fill = rng.choice(FILL_COLORS)
        w, h = tw + rng.randint(22, 36), thh + rng.randint(14, 20)

        def draw(d, x, y):
            if style == "flat":
                d.rounded_rectangle((x, y, x + w, y + h), 6, fill=th["btn"], outline=th["border"])
                c = th["btn_text"]
            elif style == "outline":
                d.rounded_rectangle((x, y, x + w, y + h), 6, outline=th["text"], width=2)
                c = th["text"]
            else:
                d.rounded_rectangle((x, y, x + w, y + h), 6, fill=fill)
                c = (255, 255, 255)
            _text(d, (x + (w - tw) / 2, y + (h - thh) / 2), label, f, c)
        return Element(kind, label, w, h, (0, 0, w, h), draw)
    if kind == "link":
        w, h = tw, thh + 4

        def draw(d, x, y):
            _text(d, (x, y), label, f, (70, 130, 235))
            d.line((x, y + thh + 3, x + tw, y + thh + 3), fill=(70, 130, 235))
        return Element(kind, label, w, h, (-2, -2, w + 2, h + 2), draw)
    if kind == "checkbox":
        cs = rng.randint(18, 24)
        checked = rng.random() < 0.4
        w, h = cs + 8 + tw, max(cs, thh)

        def draw(d, x, y):
            yy = y + (h - cs) / 2
            d.rounded_rectangle((x, yy, x + cs, yy + cs), 4, outline=th["text"], width=2,
                                fill=(70, 130, 235) if checked else th["panel"])
            if checked:
                d.line((x + 4, yy + cs / 2, x + cs * 0.42, yy + cs - 5, x + cs - 4, yy + 5), fill=(255, 255, 255), width=3)
            _text(d, (x + cs + 8, y + (h - thh) / 2), label, f, th["text"])
        return Element(kind, label, w, h, (0, (h - cs) / 2, cs, (h + cs) / 2), draw)
    if kind == "input":
        w, ih = rng.randint(150, 230), rng.randint(30, 36)
        above = rng.random() < 0.5
        lh = thh + 6 if above else 0
        h = ih + lh

        def draw(d, x, y):
            d.rounded_rectangle((x, y + lh, x + w, y + lh + ih), 5, fill=th["panel"], outline=th["border"], width=2)
            if above:
                _text(d, (x, y), label, f, th["text"])
            else:
                _text(d, (x + 10, y + (ih - thh) / 2), label, f, th["muted"])
        return Element(kind, label, w, h, (0, lh, w, lh + ih), draw)
    if kind == "tab":
        w, h = tw + 28, thh + 16
        sel = rng.random() < 0.3

        def draw(d, x, y):
            if sel:
                d.line((x, y + h - 1, x + w, y + h - 1), fill=(70, 130, 235), width=3)
            _text(d, (x + 14, y + 8), label, f, th["text"] if sel else th["muted"])
        return Element(kind, label, w, h, (0, 0, w, h), draw)
    # icon: label is "<color> <shape>"
    color, shape = label.split()
    s = rng.randint(28, 38)
    cell = rng.random() < 0.5

    def draw(d, x, y):
        if cell:
            d.rounded_rectangle((x, y, x + s, y + s), 7, fill=th["btn"], outline=th["border"])
        _shape(d, shape, (x + 6, y + 6, x + s - 6, y + s - 6), ICON_COLORS[color])
    return Element(kind, label, s, s, (0, 0, s, s), draw)


def _instruction(rng, e: Element) -> str:
    l = e.label
    return rng.choice({
        "button": [f'Click the "{l}" button.', f'Press the {l} button.', f'Select the button labeled "{l}".'],
        "link": [f'Click the "{l}" link.', f'Follow the "{l}" link.'],
        "checkbox": [f'Click the checkbox next to "{l}".', f'Tick the "{l}" checkbox.'],
        "input": [f'Click the "{l}" input field.', f'Focus the {l} text box.'],
        "tab": [f'Open the "{l}" tab.', f'Switch to the "{l}" tab.'],
        "icon": [f"Click the {l} icon.", f"Tap the {l} icon."],
    }[e.kind])


def style_pools(style: str):
    """'all' = everything; 'seen' / 'heldout' split the visual style (colour theme + font family) so a model can be
    trained on one "set of apps" and tested on apps it never saw. Held-out = last theme + last available font."""
    if style == "all":
        return THEMES, FONTS
    if style == "seen":
        return THEMES[:-1], FONTS[:-1] or FONTS
    if style == "heldout":
        return THEMES[-1:], FONTS[-1:]
    raise ValueError(style)


def make_sample(seed: int, p_pos: float | None = None, p_ring: float | None = None, style: str = "all") -> Sample:
    """Deterministic in (seed, p_pos, p_ring, style). style='all' is the original behaviour, bit for bit."""
    rng = random.Random(seed * 7919 + 13)
    pools = style_pools(style)
    while True:
        s = _try_make(rng, seed, p_pos, p_ring, pools)
        if s is not None:
            return s


def _try_make(rng, seed, p_pos, p_ring, pools=(THEMES, FONTS)):
    th = rng.choice(pools[0])
    fname, fs = rng.choice(pools[1]), rng.randint(13, 18)
    f = font(fname, fs)
    img = Image.new("RGB", (W, H), th["bg"])
    d = ImageDraw.Draw(img)
    mx, my = rng.randint(6, 30), rng.randint(6, 24)
    win = (mx, my, W - mx, H - my)
    d.rounded_rectangle(win, 8, fill=th["panel"], outline=th["border"])
    d.rounded_rectangle((win[0], win[1], win[2], win[1] + 26), 8, fill=th["bar"])
    for i in range(3):
        d.ellipse((win[0] + 10 + i * 16, win[1] + 8, win[0] + 20 + i * 16, win[1] + 18), fill=th["muted"])
    cx0, cy0, cx1, cy1 = win[0] + 20, win[1] + 40, win[2] - 20, win[3] - 14

    # ---- choose elements with unique labels ------------------------------------------------------------
    n = rng.randint(6, 13)
    kinds = rng.choices(["button", "link", "checkbox", "input", "icon", "tab"], [5, 2, 2, 2, 3, 0], k=n)
    pools = {"button": rng.sample(BUTTONS, len(BUTTONS)), "link": rng.sample(LINKS, len(LINKS)),
             "checkbox": rng.sample(CHECKS, len(CHECKS)), "input": rng.sample(INPUTS, len(INPUTS)),
             "tab": rng.sample(TABS, len(TABS))}
    icon_pool = [f"{c} {s}" for c in ICON_COLORS for s in ICON_SHAPES]
    rng.shuffle(icon_pool)
    elems = []
    for k in kinds:
        label = icon_pool.pop() if k == "icon" else pools[k].pop()
        elems.append(_make_element(rng, k, label, th, f, fname, fs))
    tabs = []
    if rng.random() < 0.5:
        tabs = [_make_element(rng, "tab", pools["tab"].pop(), th, f, fname, fs) for _ in range(rng.randint(2, 4))]

    # ---- flow layout: optional heading, optional tab strip, then justified rows ---------------------------
    y = cy0
    if rng.random() < 0.6:
        hf = font(fname, fs + 6)
        _text(d, (cx0, y), rng.choice(HEADINGS), hf, th["text"])
        y += _measure("Ag", hf)[1] + rng.randint(12, 20)
    placed = []
    if tabs:
        x = cx0
        for t in tabs:
            t.box = (x + t.box_off[0], y + t.box_off[1], x + t.box_off[2], y + t.box_off[3])
            t.draw_fn(d, x, y)
            placed.append(t)
            x += t.w + 6
        d.line((cx0, y + tabs[0].h, cx1, y + tabs[0].h), fill=th["border"])
        y += tabs[0].h + rng.randint(16, 26)
    rows, row, rw = [], [], 0
    gap = rng.randint(14, 28)
    for e in elems:
        if row and rw + gap + e.w > cx1 - cx0:
            rows.append(row)
            row, rw = [], 0
        row.append(e)
        rw += e.w + (gap if len(row) > 1 else 0)
    rows.append(row)
    # decorative skeleton-text rows (never targets), then spread all rows over the free height so that
    # vertical position is not a free shortcut either
    rows = [r for r in rows if r]
    for i in range(len(rows), 0, -1):
        if rng.random() < 0.2:
            rows.insert(i, "deco")
    min_gap = 12
    while len(rows) > 1 and sum(24 if r == "deco" else max(e.h for e in r) for r in rows) \
            + min_gap * (len(rows) - 1) > cy1 - y:
        rows.pop()
    heights = [24 if r == "deco" else max(e.h for e in r) for r in rows]
    slack = max(0, cy1 - y - sum(heights) - min_gap * (len(rows) - 1))
    y += rng.uniform(0, slack * 0.25)
    vgap = min_gap + min(50, (slack * 0.75 * rng.uniform(0.6, 1.0)) / max(1, len(rows) - 1))
    for row in rows:
        if row == "deco":
            for j in range(2):
                d.rounded_rectangle((cx0, y + j * 14, cx0 + rng.randint(120, 320), y + j * 14 + 8), 4, fill=th["btn"])
            y += 24 + vgap
            continue
        rh = max(e.h for e in row)
        total = sum(e.w for e in row)
        mode = rng.choice(["left", "center", "right", "spread"])
        if mode == "spread" and len(row) > 1:
            g = (cx1 - cx0 - total) / (len(row) - 1)
            x = cx0
        else:
            g = gap
            full = total + g * (len(row) - 1)
            x = {"left": cx0, "spread": cx0, "center": (cx0 + cx1 - full) / 2, "right": cx1 - full}[mode]
        for e in row:
            ey = y + (rh - e.h) / 2
            e.box = (x + e.box_off[0], ey + e.box_off[1], x + e.box_off[2], ey + e.box_off[3])
            e.draw_fn(d, x, ey)
            placed.append(e)
            x += e.w + g
        y += rh + vgap
    if len(placed) < 5:
        return None

    # ---- pick target / ring according to the cue probabilities -----------------------------------------
    if p_pos is None:
        target = rng.choice(placed)
    else:
        left = [e for e in placed if e.center[0] < W / 2]
        right = [e for e in placed if e.center[0] >= W / 2]
        pool = left if rng.random() < p_pos else right
        target = rng.choice(pool or placed)
    if p_ring is None:
        ring = rng.choice(placed)
    elif rng.random() < p_ring:
        ring = target
    else:
        ring = rng.choice([e for e in placed if e is not target])
    x0, y0, x1, y1 = ring.box
    d.rounded_rectangle((x0 - 5, y0 - 5, x1 + 5, y1 + 5), 6, outline=RING, width=3)
    return Sample(seed, img, _instruction(rng, target), placed, placed.index(target), placed.index(ring))


# split recipes: (p_pos, p_ring). Train/test use disjoint seed ranges.
SPLITS = {
    "clean": (None, None),         # cues independent of the target (no shortcut available)
    "biased": (0.9, 0.9),          # both cues predict the target 90% of the time
    "unbiased_test": (None, None),
    "iid_biased_test": (0.9, 0.9),
    "adversarial_test": (0.0, 0.0),  # both cues point *away* from the target
    # cue isolation (2x2): one cue aligned / anti-aligned with the target, the other independent of it
    "pos_aligned_test": (1.0, None),
    "pos_anti_test": (0.0, None),
    "ring_aligned_test": (None, 1.0),
    "ring_anti_test": (None, 0.0),
}
TEST_SEED_BASE = 10_000_000
