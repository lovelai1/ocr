import json
import os
import re
import time
import threading
import random
import queue
from pathlib import Path
from urllib.parse import quote
from collections import deque

import mss
import pytesseract
import requests
from PIL import Image, ImageOps, ImageFilter


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

# Ники: только A-Z a-z 0-9, whitespace убираем полностью
ALLOWED_RE = re.compile(r"[^A-Za-z0-9]+")
SPACES_RE = re.compile(r"\s+")
RESOLVED_RE = re.compile(r"resolved=(.+)$", re.IGNORECASE)


# =========================
# Helpers / State
# =========================
def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Не найден {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def norm_text(s: str) -> str:
    s = (s or "").replace("\x0c", "")
    s = s.strip()
    s = SPACES_RE.sub("", s)     # убрать пробелы/переносы
    s = ALLOWED_RE.sub("", s)    # оставить только A-Za-z0-9
    return s


def key_of(tag: str) -> str:
    return norm_text(tag).lower()


def load_processed(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "entries": {}}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "entries": {}}

    if isinstance(data, dict) and data.get("version") == 1 and isinstance(data.get("entries"), dict):
        return data

    migrated = {"version": 1, "entries": {}}
    if isinstance(data, dict):
        for k in ("added", "notfound", "privacy403", "privacy_403"):
            arr = data.get(k)
            if isinstance(arr, list):
                for tag in arr:
                    if isinstance(tag, str) and tag.strip():
                        migrated["entries"][key_of(tag)] = {"tag": norm_text(tag), "status": k, "ts": time.time()}
    return migrated


def save_processed(path: Path, state: dict) -> None:
    state.setdefault("version", 1)
    state.setdefault("entries", {})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def is_done(state: dict, tag: str) -> bool:
    k = key_of(tag)
    if not k:
        return True
    v = state.get("entries", {}).get(k)
    if isinstance(v, dict):
        st = str(v.get("status", "")).lower()
        return st in ("added", "notfound", "privacy403", "privacy_403")
    return False


def mark(state: dict, tag: str, status: str, note: str | None = None, xuid: str | None = None) -> None:
    k = key_of(tag)
    if not k:
        return

    state.setdefault("entries", {})
    prev = state["entries"].get(k)
    if isinstance(prev, dict) and str(prev.get("status", "")).lower() == "added" and status != "added":
        return

    state["entries"][k] = {
        "tag": norm_text(tag),
        "status": status,
        "ts": time.time(),
        "note": note,
        "xuid": xuid,
    }


# =========================
# Learning cache
# =========================
def _extract_resolved(note: str | None) -> str | None:
    if not note:
        return None
    m = RESOLVED_RE.search(str(note).strip())
    if not m:
        return None
    resolved = norm_text(m.group(1))
    return resolved if resolved else None


class ResolutionCache:
    def __init__(self):
        self.by_key: dict[str, tuple[str, str]] = {}

    def ingest_state(self, state: dict) -> None:
        entries = state.get("entries", {})
        if not isinstance(entries, dict):
            return

        for _, v in entries.items():
            if not isinstance(v, dict):
                continue
            if str(v.get("status", "")).lower() != "added":
                continue

            tag = v.get("tag")
            xuid = v.get("xuid")
            if not isinstance(tag, str) or not isinstance(xuid, str):
                continue

            tag_n = norm_text(tag)
            xuid_n = str(xuid).strip()
            if not tag_n or not xuid_n:
                continue

            resolved = _extract_resolved(v.get("note")) or tag_n
            self.by_key[key_of(tag_n)] = (resolved, xuid_n)

    def learn(self, raw_tag: str, resolved_tag: str, xuid: str) -> None:
        self.by_key[key_of(raw_tag)] = (norm_text(resolved_tag), str(xuid).strip())

    def lookup(self, raw_tag: str) -> tuple[str, str] | None:
        return self.by_key.get(key_of(raw_tag) or "")


# =========================
# Rate limiter
# =========================
class RequestLimiter:
    def __init__(self, min_interval_sec: float = 2.2, jitter_sec: float = 0.15):
        self.min_interval = max(0.05, float(min_interval_sec))
        self.jitter = max(0.0, float(jitter_sec))
        self._lock = threading.Lock()
        self._next_ts = 0.0  # monotonic

    def wait(self) -> None:
        now = time.monotonic()
        with self._lock:
            target = max(now, self._next_ts)
            delay = target - now
            self._next_ts = target + self.min_interval + (random.random() * self.jitter)
        if delay > 0:
            time.sleep(delay)


def limited_get(session: requests.Session, limiter: RequestLimiter, url: str, **kwargs) -> requests.Response:
    limiter.wait()
    return session.get(url, **kwargs)


def limited_post(session: requests.Session, limiter: RequestLimiter, url: str, **kwargs) -> requests.Response:
    limiter.wait()
    return session.post(url, **kwargs)


# =========================
# OCR retry candidates (BFS)
# =========================
def _replace_one_at_a_time(s: str, old: str, new: str, limit: int = 6) -> list[str]:
    out: list[str] = []
    if not s or old == new:
        return out
    idxs = [i for i, ch in enumerate(s) if ch == old]
    for i in idxs[:limit]:
        v = s[:i] + new + s[i + 1 :]
        if v != s and v not in out:
            out.append(v)
    return out


def gen_retry_candidates(tag: str, max_candidates: int = 8, max_depth: int = 2) -> list[str]:
    if not tag:
        return []

    RULES: list[tuple[str, str]] = [
        ("e", "a"), ("E", "A"),
        ("a", "e"), ("A", "E"),

        ("s", "a"), ("S", "A"),
        ("a", "s"), ("A", "S"),

        ("S", "5"), ("s", "5"), ("5", "S"),

        ("O", "D"), ("D", "O"),
        ("O", "R"), ("R", "O"),
        ("o", "d"), ("d", "o"),
        ("o", "r"), ("r", "o"),

        ("M", "H"), ("H", "M"),
        ("m", "h"), ("h", "m"),
    ]

    PER_RULE_LIMIT = 4
    seen: set[str] = {tag}
    out: list[str] = []
    q = deque([(tag, 0)])

    while q and len(out) < max_candidates:
        cur, depth = q.popleft()
        if depth >= max_depth:
            continue

        for old, new in RULES:
            if old not in cur:
                continue
            for v in _replace_one_at_a_time(cur, old, new, limit=PER_RULE_LIMIT):
                if v in seen:
                    continue
                seen.add(v)
                out.append(v)
                q.append((v, depth + 1))
                if len(out) >= max_candidates:
                    break
            if len(out) >= max_candidates:
                break

    return out


# =========================
# Overlay (optional)
# =========================
def start_overlay(left: int, top: int, width: int, height: int, border_px: int, label: str, click_through: bool) -> None:
    import tkinter as tk
    import ctypes
    from ctypes import wintypes

    TRANSPARENT_COLOR = "magenta"

    root = tk.Tk()
    root.overrideredirect(True)
    root.geometry(f"{width}x{height}+{left}+{top}")
    root.attributes("-topmost", True)

    try:
        root.wm_attributes("-transparentcolor", TRANSPARENT_COLOR)
    except Exception:
        pass

    canvas = tk.Canvas(root, width=width, height=height, highlightthickness=0, bg=TRANSPARENT_COLOR)
    canvas.pack(fill="both", expand=True)

    canvas.create_rectangle(
        border_px // 2,
        border_px // 2,
        width - border_px // 2 - 1,
        height - border_px // 2 - 1,
        outline="red",
        width=border_px,
    )

    if label:
        canvas.create_text(6, 6, anchor="nw", fill="red", text=label, font=("Consolas", 10, "bold"))

    if click_through:
        try:
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x80000
            WS_EX_TRANSPARENT = 0x20

            user32 = ctypes.windll.user32
            get_long = user32.GetWindowLongW
            set_long = user32.SetWindowLongW
            get_long.argtypes = [wintypes.HWND, ctypes.c_int]
            set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_int]

            style = get_long(hwnd, GWL_EXSTYLE)
            set_long(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
        except Exception:
            pass

    root.mainloop()


# =========================
# Xbox API
# =========================
def load_xsts_from_tokens(tokens_path: Path) -> tuple[str, str]:
    data = json.loads(tokens_path.read_text(encoding="utf-8"))
    xsts = data.get("xboxLiveXstsToken", {})
    token = xsts.get("token")
    user_hash = xsts.get("userHash")
    if not token or not user_hash:
        raise RuntimeError(f"В {tokens_path.name} не найден xboxLiveXstsToken")
    return user_hash, token


def auth_headers(user_hash: str, xsts_token: str) -> dict:
    return {
        "Authorization": f"XBL3.0 x={user_hash};{xsts_token}",
        "x-xbl-contract-version": "2",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


class RateLimitError(RuntimeError):
    def __init__(self, retry_after: float):
        super().__init__(f"RATE_LIMIT retry_after={retry_after}")
        self.retry_after = retry_after


def retry_after_seconds(resp: requests.Response) -> float:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return 2.0
    try:
        v = float(ra)
        return max(0.5, min(60.0, v))
    except Exception:
        return 2.0


def get_my_profile(session: requests.Session, limiter: RequestLimiter, user_hash: str, xsts_token: str) -> tuple[str, str]:
    url = "https://profile.xboxlive.com/users/me/profile/settings?settings=Gamertag"
    r = limited_get(session, limiter, url, headers=auth_headers(user_hash, xsts_token), timeout=15)

    if r.status_code == 429:
        raise RateLimitError(retry_after_seconds(r))
    if r.status_code == 401:
        raise RuntimeError("AUTH_401")
    r.raise_for_status()

    data = r.json()
    user_data = data["profileUsers"][0]
    xuid = str(user_data["id"])

    gamertag = ""
    for setting in user_data.get("settings", []):
        if setting.get("id") == "Gamertag":
            gamertag = setting.get("value", "")
            break

    return xuid, gamertag


def get_xuid_by_gamertag(session: requests.Session, limiter: RequestLimiter, user_hash: str, xsts_token: str, gamertag: str) -> str:
    url = f"https://profile.xboxlive.com/users/gt({quote(gamertag)})/profile/settings?settings=Gamertag"
    r = limited_get(session, limiter, url, headers=auth_headers(user_hash, xsts_token), timeout=15)

    if r.status_code == 429:
        raise RateLimitError(retry_after_seconds(r))
    if r.status_code == 401:
        raise RuntimeError("AUTH_401")
    if r.status_code == 404:
        raise RuntimeError("NOTFOUND")
    r.raise_for_status()

    data = r.json()
    return str(data["profileUsers"][0]["id"])


def add_friend(session: requests.Session, limiter: RequestLimiter, user_hash: str, xsts_token: str, my_xuid: str, target_xuid: str) -> None:
    url = f"https://social.xboxlive.com/users/xuid({my_xuid})/people/xuids?method=add"
    r = limited_post(
        session,
        limiter,
        url,
        headers=auth_headers(user_hash, xsts_token),
        json={"xuids": [str(target_xuid)]},
        timeout=15,
    )

    if r.status_code == 429:
        raise RateLimitError(retry_after_seconds(r))
    if r.status_code == 401:
        raise RuntimeError("AUTH_401")
    if r.status_code == 403:
        raise RuntimeError("PRIVACY_403")
    if r.status_code not in (200, 204):
        raise RuntimeError(f"ADD_FAILED HTTP {r.status_code}: {r.text}")


# =========================
# Accounts
# =========================
class SessionInfo:
    def __init__(self, path: Path, user_hash: str, xsts_token: str):
        self.path = path
        self.user_hash = user_hash
        self.xsts_token = xsts_token

        self.http = requests.Session()  # <- ВАЖНО: отдельная сессия на аккаунт

        self.my_xuid = ""
        self.my_gamertag = ""
        self.blocked_until = 0.0  # monotonic

    @property
    def log_prefix(self) -> str:
        if self.my_gamertag:
            return f"[{self.path.stem}][{self.my_gamertag}]"
        return f"[{self.path.stem}]"


class AccountManager:
    def __init__(self, token_paths: list[str], switch_interval: float):
        self.sessions: list[SessionInfo] = []
        self.switch_interval = float(switch_interval)
        self.current_idx = 0
        self.last_switch_time = time.monotonic()

        for tp in token_paths:
            p = BASE_DIR / tp
            if not p.exists():
                print(f"[WARN] Файл токена не найден: {tp}")
                continue
            uh, xt = load_xsts_from_tokens(p)
            self.sessions.append(SessionInfo(p, uh, xt))

        if not self.sessions:
            raise RuntimeError("Не загружено ни одного файла токенов!")

        print(f"[INFO] Найдено аккаунтов: {len(self.sessions)}")
        print(f"[INFO] Интервал переключения: {self.switch_interval} сек.")

    def get_active(self) -> SessionInfo:
        now = time.monotonic()
        if now - self.last_switch_time >= self.switch_interval:
            prev = self.current_idx
            self.current_idx = (self.current_idx + 1) % len(self.sessions)
            self.last_switch_time = now
            if self.current_idx != prev:
                print(f"[SWITCH] Аккаунт изменен на: {self.sessions[self.current_idx].log_prefix}")
        return self.sessions[self.current_idx]

    def get_available(self) -> SessionInfo:
        now = time.monotonic()
        self.get_active()

        for i in range(len(self.sessions)):
            idx = (self.current_idx + i) % len(self.sessions)
            if now >= self.sessions[idx].blocked_until:
                if idx != self.current_idx:
                    self.current_idx = idx
                    self.last_switch_time = now
                    print(f"[SWITCH] Аккаунт изменен на: {self.sessions[self.current_idx].log_prefix}")
                return self.sessions[idx]

        best_idx = min(range(len(self.sessions)), key=lambda j: self.sessions[j].blocked_until)
        if best_idx != self.current_idx:
            self.current_idx = best_idx
            self.last_switch_time = now
            print(f"[SWITCH] Аккаунт изменен на: {self.sessions[self.current_idx].log_prefix}")
        return self.sessions[self.current_idx]


# =========================
# OCR
# =========================
def ocr_one_line(
    img: Image.Image,
    lang: str,
    psm: int,
    oem: int,
    scale: float,
    autocontrast: bool,
    blur_radius: float,
) -> str:
    g = ImageOps.grayscale(img)
    if autocontrast:
        g = ImageOps.autocontrast(g)

    if scale and scale > 1:
        w, h = g.size
        g = g.resize((int(w * scale), int(h * scale)), resample=Image.NEAREST)

    if blur_radius and blur_radius > 0:
        g = g.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    # whitelist + отключение словарей часто помогает для геймертегов
    cfg = (
        f"--psm {int(psm)} --oem {int(oem)} "
        f"-c preserve_interword_spaces=0 "
        f"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
        f"-c load_system_dawg=0 -c load_freq_dawg=0"
    )
    text = pytesseract.image_to_string(g, lang=lang, config=cfg)
    return norm_text(text)


# =========================
# Queue helpers
# =========================
def enqueue_tag(task_q: queue.PriorityQueue, seq_counter: list[int], tag: str, not_before: float = 0.0) -> None:
    seq_counter[0] += 1
    task_q.put((not_before, seq_counter[0], tag))


# =========================
# Worker (per-account session + refresh on 401)
# =========================
def api_worker(
    stop_event: threading.Event,
    task_q: queue.PriorityQueue,
    pending: set[str],
    pending_lock: threading.Lock,
    state: dict,
    state_lock: threading.Lock,
    processed_path: Path,
    cache: ResolutionCache,
    cache_lock: threading.Lock,
    account_mgr: AccountManager,
    limiter: RequestLimiter,
    max_candidates: int,
    max_depth: int,
    cache_variant_check: int,
) -> None:
    while not stop_event.is_set():
        try:
            not_before, _, tag = task_q.get(timeout=0.5)
        except queue.Empty:
            continue

        requeue_at: float | None = None
        drop_pending = False

        try:
            now_m = time.monotonic()
            if not_before and not_before > now_m:
                requeue_at = not_before
                time.sleep(min(0.25, not_before - now_m))
                continue

            tag = norm_text(tag)
            if not tag:
                drop_pending = True
                continue

            with state_lock:
                if is_done(state, tag):
                    drop_pending = True
                    continue

            acc = account_mgr.get_available()
            now_m = time.monotonic()
            if now_m < acc.blocked_until:
                requeue_at = acc.blocked_until
                continue

            # ensure profile for THIS account using THIS account session
            def ensure_profile() -> bool:
                try:
                    uid, gt = get_my_profile(acc.http, limiter, acc.user_hash, acc.xsts_token)
                    acc.my_xuid = uid
                    acc.my_gamertag = gt
                    return True
                except RateLimitError as e:
                    acc.blocked_until = time.monotonic() + float(e.retry_after)
                    print(f"{acc.log_prefix}[RATE] 429 init, блок на {e.retry_after:.1f}s.")
                    return False
                except RuntimeError as e:
                    if str(e) == "AUTH_401":
                        print(f"{acc.log_prefix}[AUTH] 401 (profile). Проверь токен.")
                        return False
                    raise

            if not acc.my_xuid:
                ok = ensure_profile()
                if not ok:
                    requeue_at = acc.blocked_until if acc.blocked_until > time.monotonic() else (time.monotonic() + 10.0)
                    continue

            # 1) cache first
            with cache_lock:
                cached = cache.lookup(tag)

            if not cached:
                variants = gen_retry_candidates(tag, max_candidates=max_candidates, max_depth=max_depth)
                for alt in variants[:max(0, cache_variant_check)]:
                    with cache_lock:
                        cached = cache.lookup(alt)
                    if cached:
                        break

            def add_by_xuid(target_xuid: str) -> None:
                add_friend(acc.http, limiter, acc.user_hash, acc.xsts_token, acc.my_xuid, target_xuid)

            def resolve_xuid(gt: str) -> str:
                return get_xuid_by_gamertag(acc.http, limiter, acc.user_hash, acc.xsts_token, gt)

            # helper: retry once on AUTH_401 (refresh profile)
            def call_with_refresh(fn, *args):
                try:
                    return fn(*args)
                except RuntimeError as e:
                    if str(e) != "AUTH_401":
                        raise
                    # refresh profile and retry once
                    ok = ensure_profile()
                    if not ok:
                        raise RuntimeError("AUTH_401")
                    return fn(*args)

            if cached:
                resolved_tag, xuid_cached = cached
                try:
                    call_with_refresh(add_by_xuid, xuid_cached)

                    print(f"{acc.log_prefix}[ADDED:CACHE] {tag} (resolved={resolved_tag})")
                    with state_lock:
                        mark(state, tag, "added", note=f"resolved={resolved_tag}", xuid=xuid_cached)
                        mark(state, resolved_tag, "added", note=f"resolved={resolved_tag}", xuid=xuid_cached)
                        save_processed(processed_path, state)

                    with cache_lock:
                        cache.learn(tag, resolved_tag, xuid_cached)
                        cache.learn(resolved_tag, resolved_tag, xuid_cached)

                    drop_pending = True
                    continue

                except RateLimitError as e:
                    acc.blocked_until = time.monotonic() + float(e.retry_after)
                    print(f"{acc.log_prefix}[RATE] 429, блок на {e.retry_after:.1f}s.")
                    requeue_at = acc.blocked_until
                    continue
                except RuntimeError as e:
                    msg = str(e)
                    if msg == "PRIVACY_403":
                        print(f"{acc.log_prefix}[PRIVACY] {tag}")
                        with state_lock:
                            mark(state, tag, "privacy_403", note="PRIVACY_403")
                            save_processed(processed_path, state)
                        drop_pending = True
                        continue
                    print(f"{acc.log_prefix}[ERR] {tag}: {msg}")
                    drop_pending = True
                    continue

            # 2) resolve exact -> retry if NOTFOUND
            try:
                xuid = call_with_refresh(resolve_xuid, tag)
                call_with_refresh(add_by_xuid, xuid)

                print(f"{acc.log_prefix}[ADDED] {tag}")
                with state_lock:
                    mark(state, tag, "added", note=f"resolved={tag}", xuid=xuid)
                    save_processed(processed_path, state)

                with cache_lock:
                    cache.learn(tag, tag, xuid)

                drop_pending = True
                continue

            except RateLimitError as e:
                acc.blocked_until = time.monotonic() + float(e.retry_after)
                print(f"{acc.log_prefix}[RATE] 429, блок на {e.retry_after:.1f}s.")
                requeue_at = acc.blocked_until
                continue

            except RuntimeError as e:
                msg = str(e)

                if msg == "PRIVACY_403":
                    print(f"{acc.log_prefix}[PRIVACY] {tag}")
                    with state_lock:
                        mark(state, tag, "privacy_403", note="PRIVACY_403")
                        save_processed(processed_path, state)
                    drop_pending = True
                    continue

                if msg != "NOTFOUND":
                    print(f"{acc.log_prefix}[ERR] {tag}: {msg}")
                    drop_pending = True
                    continue

                # NOTFOUND -> retry candidates
                candidates = gen_retry_candidates(tag, max_candidates=max_candidates, max_depth=max_depth)
                did = False

                for alt in candidates:
                    try:
                        print(f"{acc.log_prefix}[RETRY] {tag} -> {alt}")

                        with cache_lock:
                            cached_alt = cache.lookup(alt)

                        if cached_alt:
                            resolved_tag, xuid2 = cached_alt
                        else:
                            xuid2 = call_with_refresh(resolve_xuid, alt)
                            resolved_tag = alt

                        call_with_refresh(add_by_xuid, xuid2)

                        print(f"{acc.log_prefix}[ADDED] {tag} (resolved={resolved_tag})")
                        with state_lock:
                            mark(state, tag, "added", note=f"resolved={resolved_tag}", xuid=xuid2)
                            mark(state, resolved_tag, "added", note=f"resolved={resolved_tag}", xuid=xuid2)
                            save_processed(processed_path, state)

                        with cache_lock:
                            cache.learn(tag, resolved_tag, xuid2)
                            cache.learn(resolved_tag, resolved_tag, xuid2)

                        did = True
                        break

                    except RateLimitError as e2:
                        acc.blocked_until = time.monotonic() + float(e2.retry_after)
                        print(f"{acc.log_prefix}[RATE] 429 (retry), блок на {e2.retry_after:.1f}s.")
                        requeue_at = acc.blocked_until
                        break

                    except RuntimeError as e2:
                        msg2 = str(e2)
                        if msg2 == "PRIVACY_403":
                            print(f"{acc.log_prefix}[PRIVACY] {tag} (as {alt})")
                            with state_lock:
                                mark(state, tag, "privacy_403", note=f"PRIVACY_403_RETRY:{alt}")
                                mark(state, alt, "privacy_403", note=f"PRIVACY_403_RETRY:{tag}")
                                save_processed(processed_path, state)
                            did = True
                            break
                        if msg2 == "NOTFOUND":
                            continue
                        print(f"{acc.log_prefix}[ERR] {tag} (retry as {alt}): {msg2}")
                        continue

                if requeue_at is not None:
                    continue

                if not did:
                    print(f"{acc.log_prefix}[NOTFOUND] {tag}")
                    with state_lock:
                        mark(state, tag, "notfound", note="NOTFOUND")
                        save_processed(processed_path, state)

                drop_pending = True
                continue

        finally:
            # requeue если надо
            if requeue_at is not None:
                try:
                    task_q.put((requeue_at, random.randint(1, 10_000_000), tag))
                except Exception:
                    with pending_lock:
                        pending.discard(key_of(tag))
            else:
                if drop_pending:
                    with pending_lock:
                        pending.discard(key_of(tag))

            # task_done строго 1 раз на item
            try:
                task_q.task_done()
            except Exception:
                pass


# =========================
# Main
# =========================
def main() -> None:
    cfg = load_config()

    print("[INFO] script:", Path(__file__).resolve())
    print("[INFO] config:", CONFIG_PATH.resolve())

    # tesseract paths
    tesseract_cmd = cfg.get("tesseract_cmd", "")
    tessdata_dir = cfg.get("tessdata_dir", "")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    if tessdata_dir:
        os.environ["TESSDATA_PREFIX"] = tessdata_dir

    processed_path = BASE_DIR / cfg.get("processed_path", "processed.json")
    state = load_processed(processed_path)
    state_lock = threading.Lock()

    cache = ResolutionCache()
    cache.ingest_state(state)
    cache_lock = threading.Lock()

    # capture region
    region = cfg["capture_region"]
    left = int(region["left"])
    top = int(region["top"])
    width = int(region["width"])
    height = int(region["height"])

    # overlay
    overlay = cfg.get("overlay", {})
    if bool(overlay.get("enabled", True)):
        th = threading.Thread(
            target=start_overlay,
            args=(
                left, top, width, height,
                int(overlay.get("border_px", 2)),
                str(overlay.get("label", "")),
                bool(overlay.get("click_through", True)),
            ),
            daemon=True,
        )
        th.start()

    # ocr config
    ocr_cfg = cfg.get("ocr", {})
    lang = str(ocr_cfg.get("lang", "mc"))
    psm = int(ocr_cfg.get("psm", 7))
    oem = int(ocr_cfg.get("oem", 1))
    scale = float(ocr_cfg.get("scale", 4.0))
    autocontrast = bool(ocr_cfg.get("autocontrast", True))
    blur_radius = float(ocr_cfg.get("blur_radius", 0.6))

    # logic
    logic = cfg.get("logic", {})
    stable_reads = int(logic.get("stable_reads", 2))
    min_len = int(logic.get("min_len", 3))
    max_len = int(logic.get("max_len", 15))

    # debug
    debug = cfg.get("debug", {})
    print_ocr = bool(debug.get("print_ocr", True))
    print_events = bool(debug.get("print_events", True))

    # api config
    api_cfg = cfg.get("api", {})
    api_enabled = bool(api_cfg.get("enabled", True))
    switch_sec = float(api_cfg.get("switch_interval_sec", 6.0))
    min_req_interval = float(api_cfg.get("min_request_interval_sec", 2.2))
    max_candidates = int(api_cfg.get("max_candidates", 8))
    max_depth = int(api_cfg.get("max_depth", 2))
    cache_variant_check = int(api_cfg.get("cache_variant_check", 4))
    queue_maxsize = int(api_cfg.get("queue_maxsize", 800))

    limiter = RequestLimiter(min_interval_sec=min_req_interval, jitter_sec=0.15)

    account_mgr = None
    if api_enabled:
        paths = cfg.get("tokens_paths") or []
        if not paths:
            single = cfg.get("tokens_path", "tokens.json")
            paths = [single] if single else []

        account_mgr = AccountManager(paths, switch_sec)

        # preload profile per account on its own session
        print("\n=== [INIT] Загрузка профилей аккаунтов ===")
        for sess in account_mgr.sessions:
            print(f"Профиль: {sess.path.name} ... ", end="", flush=True)
            try:
                uid, gt = get_my_profile(sess.http, limiter, sess.user_hash, sess.xsts_token)
                sess.my_xuid = uid
                sess.my_gamertag = gt
                print(f"OK -> {sess.log_prefix}")
            except Exception as e:
                print(f"FAIL: {e}")
        print("=========================================\n")
    else:
        print("[INFO] API выключен — будет только OCR вывод.")

    # queue + pending
    task_q: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_maxsize)
    seq_counter = [0]
    pending: set[str] = set()
    pending_lock = threading.Lock()

    stop_event = threading.Event()

    # start worker
    if api_enabled and account_mgr:
        worker = threading.Thread(
            target=api_worker,
            args=(
                stop_event,
                task_q,
                pending,
                pending_lock,
                state,
                state_lock,
                processed_path,
                cache,
                cache_lock,
                account_mgr,
                limiter,
                max_candidates,
                max_depth,
                cache_variant_check,
            ),
            daemon=True,
        )
        worker.start()
        print("[INFO] API worker started")

    capture_region = {"left": left, "top": top, "width": width, "height": height}
    print(f"OCR region: left={left} top={top} width={width} height={height}")
    print("Started. Ctrl+C to stop.")

    last_raw = ""
    same_count = 0
    last_committed = ""

    with mss.mss() as sct:
        try:
            while not stop_event.is_set():
                shot = sct.grab(capture_region)
                img = Image.frombytes("RGB", shot.size, shot.rgb)

                tag = ocr_one_line(
                    img,
                    lang=lang,
                    psm=psm,
                    oem=oem,
                    scale=scale,
                    autocontrast=autocontrast,
                    blur_radius=blur_radius,
                )

                if not (min_len <= len(tag) <= max_len):
                    tag = ""

                acc_info = ""
                if api_enabled and account_mgr:
                    acc_info = account_mgr.get_active().log_prefix

                if print_ocr:
                    print(f"[OCR]{acc_info} {tag}")

                if tag and tag == last_raw:
                    same_count += 1
                else:
                    last_raw = tag
                    same_count = 1 if tag else 0

                if tag and same_count >= stable_reads and tag != last_committed:
                    last_committed = tag
                    if print_events:
                        print(f"[EVENT] NEW LINE: {tag}")

                    if api_enabled and account_mgr:
                        k = key_of(tag)
                        if k:
                            with state_lock:
                                done = is_done(state, tag)
                            if not done:
                                with pending_lock:
                                    if k not in pending:
                                        pending.add(k)
                                        try:
                                            enqueue_tag(task_q, seq_counter, tag, not_before=time.monotonic())
                                            # print(f"[QUEUE] + {tag} (qsize={task_q.qsize()})")
                                        except queue.Full:
                                            pending.discard(k)

                time.sleep(0.01)

        except KeyboardInterrupt:
            stop_event.set()
            print("Stopping...")

    stop_event.set()
    time.sleep(0.3)
    print("Stopped.")


if __name__ == "__main__":
    main()
