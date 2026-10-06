#!/usr/bin/env python3
"""fieldcard — a pocket field card written by an open-weight model that runs on your own machine.

Two commands, and the screen is the shortest part of both:

    python fieldcard.py quest --place "riverside park" --minutes 45 --lat 37.53 --lon 126.93
        -> asks a local Gemma model (via Ollama) for five things to notice outside today,
           and writes a printable A6 card. Print it, close the laptop, go.

    python fieldcard.py check out/card.json path/to/photos/
        -> when you are back, the same local model looks at your photos (or photos of your
           pencil sketches) and writes a one-page field journal: what you found, and what to
           look closer at next time.

Everything talks to http://127.0.0.1:11434 (Ollama). With --airplane the process refuses
any non-loopback connection, so you can prove nothing leaves the machine.

Standard library only. Pillow is optional (used to shrink photos before inference).
License: MIT.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import io
import ipaddress
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

__version__ = "0.1.0"
DEFAULT_MODEL = os.environ.get("FIELDCARD_MODEL", "gemma4:e4b")
DEFAULT_HOST = os.environ.get("FIELDCARD_OLLAMA", "http://127.0.0.1:11434")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
HABITATS = ["park", "forest", "city", "coast", "mountain", "garden", "riverside", "backyard"]

# ---------------------------------------------------------------------------
# Airplane mode: refuse every connection that is not to this machine.
# ---------------------------------------------------------------------------

_real_create_connection = socket.create_connection
_real_getaddrinfo = socket.getaddrinfo


def _is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def enable_airplane_mode() -> None:
    """Monkeypatch the socket layer so only loopback hosts can be reached."""

    def guarded_getaddrinfo(host, *a, **k):
        if host is not None and not _is_loopback(str(host)):
            raise ConnectionRefusedError(f"[airplane] blocked lookup of {host!r}: fieldcard only talks to this machine")
        return _real_getaddrinfo(host, *a, **k)

    def guarded_create_connection(address, *a, **k):
        if not _is_loopback(str(address[0])):
            raise ConnectionRefusedError(f"[airplane] blocked connection to {address[0]!r}")
        return _real_create_connection(address, *a, **k)

    socket.getaddrinfo = guarded_getaddrinfo
    socket.create_connection = guarded_create_connection


# ---------------------------------------------------------------------------
# Sky math (no API): season and sunset from date + coordinates.
# ---------------------------------------------------------------------------

def season_for(date: dt.date, lat: float) -> str:
    """Meteorological season, flipped for the southern hemisphere."""
    north = ["winter", "winter", "spring", "spring", "spring", "summer",
             "summer", "summer", "autumn", "autumn", "autumn", "winter"]
    s = north[date.month - 1]
    if lat < 0:
        s = {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}[s]
    return s


def sun_times(date: dt.date, lat: float, lon: float, tz_hours: float):
    """Sunrise/sunset in local clock time using the NOAA general solar position formulas.

    Returns (sunrise, sunset) as datetime.time, or (None, None) during polar day/night.
    Accuracy is a couple of minutes, which is all a walk needs.
    """
    n = date.timetuple().tm_yday
    gamma = 2 * math.pi / 365 * (n - 1)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    zenith = math.radians(90.833)
    latr = math.radians(lat)
    cos_ha = math.cos(zenith) / (math.cos(latr) * math.cos(decl)) - math.tan(latr) * math.tan(decl)
    if cos_ha < -1 or cos_ha > 1:
        return None, None
    ha = math.degrees(math.acos(cos_ha))

    def to_time(minutes_utc: float) -> dt.time:
        m = round(minutes_utc + tz_hours * 60) % 1440
        return dt.time(m // 60, m % 60)

    sunrise = 720 - 4 * (lon + ha) - eqtime
    sunset = 720 - 4 * (lon - ha) - eqtime
    return to_time(sunrise), to_time(sunset)


# ---------------------------------------------------------------------------
# Talking to the local model (Ollama HTTP API).
# ---------------------------------------------------------------------------

class ModelError(RuntimeError):
    pass


def ollama_chat(host: str, model: str, messages: list, schema: dict, temperature: float = 0.7,
                timeout: int = 600) -> tuple[dict, dict]:
    """One non-streaming chat call constrained to a JSON schema. Returns (parsed, stats)."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema,
        "think": False,
        "options": {"temperature": temperature},
        "keep_alive": "10m",
    }
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise ModelError(f"Ollama answered {e.code}: {e.read().decode(errors='replace')[:300]}") from e
    except (urllib.error.URLError, ConnectionError) as e:
        raise ModelError(f"Could not reach Ollama at {host} ({e}). Is `ollama serve` running?") from e
    text = data.get("message", {}).get("content", "")
    try:
        parsed = parse_json_reply(text)
    except ValueError as e:
        raise ModelError(f"Model did not return JSON: {text[:200]!r}") from e
    stats = {
        "model": data.get("model", model),
        "seconds": round(data.get("total_duration", 0) / 1e9, 2),
        "eval_tokens": data.get("eval_count", 0),
        "tokens_per_second": round(data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1e9), 1)
        if data.get("eval_duration") else None,
    }
    return parsed, stats


def parse_json_reply(text: str) -> dict:
    """Parse a model reply that should be JSON, tolerating code fences or stray prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("no JSON object found")
        return json.loads(m.group(0))


# ---------------------------------------------------------------------------
# quest: the card
# ---------------------------------------------------------------------------

QUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "find": {"type": "string"},
                    "why": {"type": "string"},
                    "proof": {"type": "string", "enum": ["photo", "sketch", "tally"]},
                },
                "required": ["find", "why", "proof"],
            },
        },
        "sit_spot": {"type": "string"},
    },
    "required": ["title", "items", "sit_spot"],
}

# Things a card must never ask for. Checked in code, not left to the model's good manners.
UNSAFE = re.compile(
    r"\b(eat|eats|eating|taste|tasting|lick|forage|foraging|pick (?:a |the |some )?(?:mushroom|berr)|"
    r"touch (?:a |the )?(?:bird|animal|insect|snake|frog|spider|wildlife)|feed (?:the |a )?(?:bird|duck|animal|wildlife)|"
    r"climb (?:a |the |up )?(?:tree|rock|wall|fence|cliff)|trespass|disturb (?:a |the )?nest|go off[- ]trail|phone|screen|app\b)",
    re.IGNORECASE,
)


COUNTABLE = re.compile(r"\b(how many|count|number of|tally)\b", re.I)


def build_quest_messages(ctx: dict) -> list:
    system = (
        "You write pocket field cards that get people outside and looking closely. "
        "Each card has exactly 5 things to notice on a walk. Rules: "
        "(1) every item must be findable on foot in the given habitat, season and region within the walk time; "
        "(2) observe only: never ask people to eat, taste, forage, pick fungi or berries, touch or feed animals, climb, "
        "go off-trail, or catch anything; "
        "(3) never mention phones, screens or apps; "
        "(4) be concrete and sensory (a shape, a colour, a sound, a smell, a texture), not generic like 'a tree'; "
        "(5) 'find' is at most 14 words, 'why' is one sentence of real natural history, at most 22 words; "
        "(6) 'proof' is photo, sketch, or tally; tally means counting something countable ('how many crows'), never a sound or a smell; "
        "(8) only name plants or animals that are common in that region; when unsure, describe shape and colour instead of a species name; "
        "(7) 'sit_spot' is one sentence inviting a 60-second still, silent pause somewhere specific. "
        "Write in plain English."
    )
    user = (
        f"Date: {ctx['date']} ({ctx['season']} in the {ctx['hemisphere']} hemisphere). "
        f"Place: {ctx['place']}. Habitat: {ctx['habitat']}. Walk length: {ctx['minutes']} minutes. "
        f"Daylight: sunset around {ctx['sunset'] or 'unknown'}. "
        + (f"Walker's note: {ctx['note']}. " if ctx.get("note") else "")
        + "Write the card as JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_quest(q: dict) -> list[str]:
    """Return a list of problems. Empty list means the card is printable."""
    problems = []
    items = q.get("items") or []
    if len(items) != 5:
        problems.append(f"expected 5 items, got {len(items)}")
    for i, it in enumerate(items, 1):
        find, why = str(it.get("find", "")).strip(), str(it.get("why", "")).strip()
        if not find:
            problems.append(f"item {i} is empty")
        if len(find.split()) > 18:
            problems.append(f"item {i} 'find' is too long ({len(find.split())} words)")
        if len(why.split()) > 30:
            problems.append(f"item {i} 'why' is too long ({len(why.split())} words)")
        if it.get("proof") not in ("photo", "sketch", "tally"):
            problems.append(f"item {i} has proof={it.get('proof')!r}")
        if it.get("proof") == "tally" and not COUNTABLE.search(find):
            problems.append(f"item {i} is a tally but does not say what to count (start it with 'How many' or 'Count')")
        for field in (find, why):
            m = UNSAFE.search(field)
            if m:
                problems.append(f"item {i} asks for something we never print: {m.group(0)!r}")
    if UNSAFE.search(str(q.get("sit_spot", ""))):
        problems.append("sit_spot breaks a safety rule")
    finds = [str(it.get("find", "")).lower() for it in items]
    if len(set(finds)) != len(finds):
        problems.append("duplicate items")
    return problems


def make_quest(args) -> dict:
    today = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    tz = args.tz if args.tz is not None else -time.timezone / 3600 + (1 if time.daylight and time.localtime().tm_isdst else 0)
    sunrise, sunset = sun_times(today, args.lat, args.lon, tz)
    ctx = {
        "date": today.isoformat(),
        "season": season_for(today, args.lat),
        "hemisphere": "northern" if args.lat >= 0 else "southern",
        "place": args.place,
        "habitat": args.habitat,
        "minutes": args.minutes,
        "sunrise": sunrise.strftime("%H:%M") if sunrise else None,
        "sunset": sunset.strftime("%H:%M") if sunset else None,
        "note": args.note,
        "lat": args.lat,
        "lon": args.lon,
    }
    messages = build_quest_messages(ctx)
    attempts = []
    for attempt in range(1, args.retries + 2):
        q, stats = ollama_chat(args.host, args.model, messages, QUEST_SCHEMA, temperature=args.temperature)
        problems = validate_quest(q)
        attempts.append({"attempt": attempt, "problems": problems, **stats})
        log(f"  attempt {attempt}: {stats['seconds']}s, {stats['eval_tokens']} tokens"
            + (f" -> rejected: {'; '.join(problems)}" if problems else " -> ok"))
        if not problems:
            break
        messages = messages + [
            {"role": "assistant", "content": json.dumps(q)},
            {"role": "user", "content": "That card breaks these rules: " + "; ".join(problems)
             + ". Rewrite the whole card as JSON, following every rule."},
        ]
    else:
        raise ModelError("The model could not produce a card that passes the safety and format checks: "
                         + "; ".join(problems))
    return {
        "fieldcard": __version__,
        "context": ctx,
        "title": q["title"].strip(),
        "items": [{"n": i, **{k: str(v).strip() for k, v in it.items()}} for i, it in enumerate(q["items"], 1)],
        "sit_spot": q["sit_spot"].strip(),
        "generated_by": {"model": attempts[-1]["model"], "host": args.host, "attempts": attempts},
    }


# ---------------------------------------------------------------------------
# check: the journal
# ---------------------------------------------------------------------------

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "seen": {"type": "string"},
        "matches": {"type": "integer"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "look_closer": {"type": "string"},
    },
    "required": ["seen", "matches", "confidence", "reason", "look_closer"],
}


def load_image_b64(path: Path, max_side: int = 896) -> str:
    """Base64 of the image, shrunk with Pillow when available (faster, same answer)."""
    try:
        from PIL import Image, ImageOps  # optional dependency

        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        return base64.b64encode(path.read_bytes()).decode()


def build_check_messages(card: dict, image_b64: str) -> list:
    listing = "\n".join(f"{it['n']}. {it['find']}" for it in card["items"])
    system = (
        "You are a friendly naturalist reading someone's field card after their walk. "
        "You will see one photo (it may be a photo of a pencil sketch). Describe only what is visible; "
        "do not guess species names unless the features make it obvious, and say 'looks like' when unsure. "
        "If an item names a specific plant or animal and the photo clearly shows a different one, "
        "it does not match (or give it a confidence below 0.5). "
        "A match means the thing the item names is itself visible. Something merely nearby or similar "
        "(a fungus for moss, a puddle for leaf litter, a bare branch for a bird) is not a match: answer 0. "
        "Decide which single card item (1-5) the photo is evidence for, or 0 if none. "
        "'seen': one sentence of what is in the picture. 'reason': one sentence on why it does or does not match. "
        "'confidence': 0.0-1.0. 'look_closer': one specific thing to notice next time at this same spot, "
        "something you would see with eyes, ears or nose, not with a phone."
    )
    user = f"The card:\n{listing}\n\nWhich item does this photo show? Answer as JSON."
    return [{"role": "system", "content": system},
            {"role": "user", "content": user, "images": [image_b64]}]


def load_audio_b64(path: Path) -> str:
    """Base64 of a 16 kHz mono WAV. Uses ffmpeg (local) to convert phone recordings."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        wav = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
                             capture_output=True, check=True).stdout
        return base64.b64encode(wav).decode()
    if path.suffix.lower() != ".wav":
        raise SystemExit(f"{path.name}: install ffmpeg to read {path.suffix} voice notes (or record WAV)")
    return base64.b64encode(path.read_bytes()).decode()


def build_voice_messages(card: dict, audio_b64: str) -> list:
    listing = "\n".join(f"{it['n']}. {it['find']}" for it in card["items"])
    system = (
        "You are a friendly naturalist reading someone's field card after their walk. "
        "You will hear one short voice note they recorded outside instead of looking at a screen. "
        "'seen': the exact transcript of the voice note. "
        "Decide which single card item (1-5) the note is evidence for, or 0 if none. "
        "Trust what they describe, but if they only say a number without describing anything, use confidence 0.3. "
        "'reason': one sentence on why it does or does not match. 'confidence': 0.0-1.0. "
        "'look_closer': one specific thing to notice next time, with eyes, ears or nose."
    )
    user = f"The card:\n{listing}\n\nWhich item does this voice note describe? Answer as JSON."
    return [{"role": "system", "content": system},
            {"role": "user", "content": user, "images": [audio_b64]}]


def check_photos(args) -> dict:
    card = json.loads(Path(args.card).read_text(encoding="utf-8"))
    evidence = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in IMAGE_EXT | AUDIO_EXT)
    if not evidence:
        raise SystemExit(f"No photos or voice notes found in {args.photos}")
    results = []
    for p in evidence:
        kind = "voice" if p.suffix.lower() in AUDIO_EXT else "photo"
        messages = (build_voice_messages(card, load_audio_b64(p)) if kind == "voice"
                    else build_check_messages(card, load_image_b64(p)))
        verdict, stats = ollama_chat(args.host, args.model, messages, CHECK_SCHEMA, temperature=0.2)
        n = int(verdict.get("matches") or 0)
        if not 0 <= n <= len(card["items"]):
            n = 0
        conf = max(0.0, min(1.0, float(verdict.get("confidence") or 0)))
        results.append({"photo": p.as_posix(), "file": p.name, "kind": kind, "item": n, "confidence": conf,
                        "seen": verdict.get("seen", "").strip(), "reason": verdict.get("reason", "").strip(),
                        "look_closer": verdict.get("look_closer", "").strip(), **stats})
        log(f"  {p.name} [{kind}]: item {n or '-'} ({conf:.2f}) in {stats['seconds']}s")
    # Each card item keeps its most convincing photo; everything else is a bonus find.
    found = {}
    for r in sorted(results, key=lambda r: -r["confidence"]):
        if r["item"] and r["confidence"] >= args.threshold and r["item"] not in found:
            found[r["item"]] = r["file"]
    for r in results:
        r["counted"] = found.get(r["item"]) == r["file"]
    return {"fieldcard": __version__, "card": card, "results": results, "found": found,
            "score": f"{len(found)}/{len(card['items'])}", "model": args.model}


# ---------------------------------------------------------------------------
# HTML output (print-first, no external assets, no JavaScript)
# ---------------------------------------------------------------------------

CSS = """
@page { size: A6; margin: 7mm; }
* { box-sizing: border-box; }
body { margin: 0; background: #e9e4d8; font: 13px/1.38 Georgia, 'Times New Roman', serif; color: #23301f; }
.card { width: 105mm; min-height: 148mm; margin: 12px auto; padding: 7mm; background: #fbf8ef;
        border: 1px solid #c9bfa6; }
h1 { font-size: 17px; margin: 0 0 2px; letter-spacing: .2px; }
.meta { font: 10.5px/1.3 'Courier New', monospace; color: #5d6b4f; margin-bottom: 9px; }
ol { padding: 0; margin: 0; list-style: none; }
li { display: grid; grid-template-columns: 16px 1fr; gap: 7px; padding: 6px 0; border-top: 1px dashed #c9bfa6; }
.box { width: 13px; height: 13px; border: 1.5px solid #23301f; margin-top: 2px; }
.find { font-weight: bold; }
.why { font-size: 11.5px; color: #4b5a40; }
.proof { font: 9.5px 'Courier New', monospace; text-transform: uppercase; color: #8a6d2f; }
.sit { margin-top: 8px; padding: 6px 8px; border: 1.5px solid #23301f; font-style: italic; }
.foot { margin-top: 8px; font: 9.5px/1.3 'Courier New', monospace; color: #5d6b4f; }
.journal { width: 180mm; }
.row { display: grid; grid-template-columns: 120px 1fr; gap: 12px; padding: 10px 0; border-top: 1px dashed #c9bfa6; }
.row img { width: 120px; height: 90px; object-fit: cover; border: 1px solid #c9bfa6; }
.tick { font-weight: bold; color: #2f6b2f; font-family: "Segoe UI Symbol", "DejaVu Sans", sans-serif; }
.next { font-size: 11.5px; font-style: italic; color: #8a6d2f; }
.miss { color: #8a6d2f; }
.score { font-size: 28px; float: right; }
.voice { width: 120px; height: 90px; border: 1px solid #c9bfa6; display: flex; flex-direction: column; justify-content: center;
         align-items: center; text-align: center; font: 11px 'Courier New', monospace; color: #4b5a40; }
.voice span { font-size: 9px; word-break: break-all; padding: 0 4px; }
@media print { body { background: none; } .card { margin: 0; border: none; } }
"""


def esc(s) -> str:
    return html.escape(str(s))


def render_card_html(card: dict) -> str:
    c = card["context"]
    proof_label = {"photo": "photo", "sketch": "sketch it", "tally": "count it"}
    items = "".join(
        f'<li><div class="box"></div><div><div class="find">{esc(it["find"])}</div>'
        f'<div class="why">{esc(it["why"])}</div><div class="proof">{proof_label.get(it["proof"], it["proof"])}</div></div></li>'
        for it in card["items"])
    daylight = f"sunset {c['sunset']}" if c.get("sunset") else "check the light"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(card['title'])}</title>
<style>{CSS}</style></head><body><div class="card">
<h1>{esc(card['title'])}</h1>
<div class="meta">{esc(c['date'])} · {esc(c['season'])} · {esc(c['place'])} · {c['minutes']} min · {esc(daylight)}</div>
<ol>{items}</ol>
<div class="sit">Sit spot — {esc(card['sit_spot'])}</div>
<div class="foot">Leave it as you found it. Print this, pocket it, close the laptop.<br>
written offline by {esc(card['generated_by']['model'])} · fieldcard {esc(card['fieldcard'])}</div>
</div></body></html>"""


def thumb_data_uri(path: str) -> str:
    return "data:image/jpeg;base64," + load_image_b64(Path(path), max_side=360)


def next_time(r: dict) -> str:
    """The model likes to start with 'Next time, ...'; the label already says it."""
    t = re.sub(r"^(next time)[,:]?\s*", "", r.get("look_closer", ""), flags=re.I)
    return t[:1].upper() + t[1:]


def quote_if_voice(r: dict) -> str:
    return f'&ldquo;{esc(r["seen"])}&rdquo;' if r.get("kind") == "voice" else esc(r["seen"])


def media_cell(r: dict) -> str:
    if r.get("kind") == "voice":
        return f'<div class="voice">&#127908; voice note<br><span>{esc(r["file"])}</span></div>'
    return f'<img src="{thumb_data_uri(r["photo"])}" alt="{esc(r["file"])}">'


def render_journal_html(j: dict) -> str:
    card = j["card"]
    by_item = {r["file"]: r for r in j["results"]}
    rows = []
    for it in card["items"]:
        f = j["found"].get(it["n"]) or j["found"].get(str(it["n"]))
        if f:
            r = by_item[f]
            rows.append(f'<div class="row">{media_cell(r)}'
                        f'<div><div><span class="tick">&#10003; {it["n"]}.</span> <b>{esc(it["find"])}</b></div>'
                        f'<div>{quote_if_voice(r)}</div><div class="why">{esc(r["reason"])}</div>'
                        f'<div class="next">Next time: {esc(next_time(r))}</div></div></div>')
        else:
            rows.append(f'<div class="row"><div class="miss">not yet</div><div><b>{it["n"]}. {esc(it["find"])}</b>'
                        f'<div class="why">Still out there. Bring the card back.</div></div></div>')
    bonus = [r for r in j["results"] if not r["counted"]]
    if bonus:
        rows.append('<h1 style="margin-top:14px">Bonus finds</h1>')
        for r in bonus:
            rows.append(f'<div class="row">{media_cell(r)}'
                        f'<div><div>{quote_if_voice(r)}</div><div class="why">{esc(r["reason"])}</div>'
                        f'<div class="next">Next time: {esc(next_time(r))}</div></div></div>')
    secs = round(sum(r["seconds"] for r in j["results"]), 1)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Field journal — {esc(card['title'])}</title>
<style>{CSS}</style></head><body><div class="card journal">
<div class="score">{esc(j['score'])}</div><h1>Field journal — {esc(card['title'])}</h1>
<div class="meta">{esc(card['context']['date'])} · {esc(card['context']['place'])} · {len(j['results'])} photos and voice notes read on this machine by {esc(j['model'])} in {secs}s</div>
{''.join(rows)}
<div class="foot">No photo or recording left this computer. fieldcard {esc(j['fieldcard'])}</div>
</div></body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "card"


def cmd_quest(args) -> None:
    log(f"fieldcard quest · {args.model} @ {args.host}")
    card = make_quest(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{card['context']['date']}-{slug(card['title'])}"
    (out / f"{stem}.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{stem}.html").write_text(render_card_html(card), encoding="utf-8")
    print(f"\n  {card['title']}\n")
    for it in card["items"]:
        print(f"  [ ] {it['n']}. {it['find']}  ({it['proof']})")
    print(f"\n  sit spot: {card['sit_spot']}")
    if card["context"]["sunset"]:
        print(f"  sunset {card['context']['sunset']}")
    print(f"\n  card: {out / (stem + '.html')}  (print it, then go)")


def cmd_check(args) -> None:
    log(f"fieldcard check · {args.model} @ {args.host}")
    j = check_photos(args)
    card_path = Path(args.card)
    out = Path(args.out) if args.out else card_path.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = card_path.stem + "-journal"
    (out / f"{stem}.json").write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{stem}.html").write_text(render_journal_html(j), encoding="utf-8")
    print(f"\n  found {j['score']}")
    for it in j["card"]["items"]:
        f = j["found"].get(it["n"])
        print(f"  [{'x' if f else ' '}] {it['n']}. {it['find']}" + (f"  <- {f}" if f else ""))
    print(f"\n  journal: {out / (stem + '.html')}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fieldcard", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama URL (default {DEFAULT_HOST})")
    ap.add_argument("--airplane", action="store_true",
                    help="refuse every non-loopback network connection for this run")
    ap.add_argument("--version", action="version", version=f"fieldcard {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quest", help="write today's field card")
    q.add_argument("--place", default="a local park", help="free text, e.g. 'Han river park, Seoul'")
    q.add_argument("--habitat", default="park", choices=HABITATS)
    q.add_argument("--minutes", type=int, default=45)
    q.add_argument("--lat", type=float, default=37.53, help="latitude, for season and sunset")
    q.add_argument("--lon", type=float, default=126.93, help="longitude, for sunset")
    q.add_argument("--tz", type=float, default=None, help="UTC offset in hours (default: this computer's)")
    q.add_argument("--date", default=None, help="YYYY-MM-DD (default today)")
    q.add_argument("--note", default=None, help="anything the card should know ('with a 6-year-old')")
    q.add_argument("--temperature", type=float, default=0.8)
    q.add_argument("--retries", type=int, default=2)
    q.add_argument("--out", default="out")
    q.set_defaults(func=cmd_quest)

    c = sub.add_parser("check", help="read your walk photos and voice notes against a card and write a journal")
    c.add_argument("card", help="the card .json written by `quest`")
    c.add_argument("photos", help="folder of photos, photos of sketches, and/or voice notes (.wav/.m4a/.mp3)")
    c.add_argument("--threshold", type=float, default=0.5, help="minimum confidence to tick an item")
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    if args.airplane:
        enable_airplane_mode()
        log("airplane mode: only loopback connections allowed")
    if args.airplane and not _is_loopback(re.sub(r"^\w+://|:\d+.*$", "", args.host)):
        raise SystemExit("--airplane needs a local --host")
    try:
        args.func(args)
    except ModelError as e:
        raise SystemExit(f"fieldcard: {e}")


if __name__ == "__main__":
    main()
