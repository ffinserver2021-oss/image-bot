"""Telegram image-grabber bot.

Send a web page link -> the bot finds every image on the page, downloads them
and sends them back as albums (10 per album), keeping the page order.

Environment variables:
  BOT_TOKEN      (required)  token from @BotFather
  PORT           (optional)  health-check port, default 8000
  ALLOWED_USERS  (optional)  comma separated Telegram user ids; empty = everyone
  MAX_IMAGES     (optional)  max images per page, default 150
  MIN_SIDE       (optional)  skip images smaller than this many px, default 150
"""
import asyncio
import hashlib
import ipaddress
import logging
import os
import re
import socket
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import aiohttp
from aiohttp import web
from bs4 import BeautifulSoup
from PIL import Image
from telegram import InputMediaDocument, InputMediaPhoto, Update
from telegram.error import RetryAfter, TelegramError, TimedOut
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("imgbot")

# ---------------------------------------------------------------- settings
BOT_TOKEN = os.environ["BOT_TOKEN"]
PORT = int(os.environ.get("PORT", "8000"))
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "150"))
MIN_SIDE = int(os.environ.get("MIN_SIDE", "150"))
ALLOWED_USERS = {int(x) for x in re.findall(r"\d+", os.environ.get("ALLOWED_USERS", ""))}

MAX_FILE_BYTES = 10 * 1024 * 1024   # Telegram photo limit
MAX_HTML_BYTES = 5 * 1024 * 1024
BATCH = 10                           # images per album
PARALLEL = 4                         # simultaneous downloads (low = less RAM)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
ACCEPT_HTML = "text/html,application/xhtml+xml,*/*;q=0.8"
ACCEPT_IMG = "image/jpeg,image/png,image/webp,image/*;q=0.8,*/*;q=0.5"

IMG_ATTRS = ("data-original", "data-src", "data-lazy-src", "data-lazy", "data-url", "src")
SRCSET_ATTRS = ("srcset", "data-srcset")
IMG_EXT = re.compile(r"\.(jpe?g|png|webp|gif|bmp)(\?|$)", re.I)
CSS_URL = re.compile(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", re.I)
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)

JOB_LOCK = asyncio.Lock()


# ------------------------------------------------------------ safe fetching
async def is_public_host(host: str) -> bool:
    """Block localhost / private network addresses (SSRF protection)."""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return True


@asynccontextmanager
async def safe_get(session, url, accept, referer=None):
    """GET with manual redirect handling so every hop is checked."""
    headers = {"User-Agent": UA, "Accept": accept, "Accept-Language": "en-US,en;q=0.9"}
    if referer:
        headers["Referer"] = referer
    for _ in range(6):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("লিংকটা ঠিক নেই।")
        if not await is_public_host(parsed.hostname):
            raise ValueError("এই লিংক ব্যবহার করা যাবে না।")
        async with session.get(url, headers=headers, allow_redirects=False) as resp:
            location = resp.headers.get("Location")
            if resp.status in (301, 302, 303, 307, 308) and location:
                url = urljoin(url, location)
                continue
            resp.request_url = url
            yield resp
            return
    raise ValueError("লিংকটা বারবার রিডাইরেক্ট হচ্ছে।")


async def fetch_html(session, url):
    async with safe_get(session, url, ACCEPT_HTML) as resp:
        if resp.status != 200:
            raise ValueError(f"পেজ খুলতে পারিনি (HTTP {resp.status})।")
        buf = bytearray()
        async for chunk in resp.content.iter_chunked(65536):
            buf += chunk
            if len(buf) >= MAX_HTML_BYTES:
                break
        try:
            text = bytes(buf).decode(resp.charset or "utf-8", errors="replace")
        except LookupError:
            text = bytes(buf).decode("utf-8", errors="replace")
        return text, resp.request_url


async def download_image(session, sem, url, referer, dest):
    """Download one image to `dest`. Returns dest or None."""
    async with sem:
        try:
            async with safe_get(session, url, ACCEPT_IMG, referer) as resp:
                if resp.status != 200:
                    return None
                ctype = resp.headers.get("Content-Type", "").lower()
                if "svg" in ctype or not (ctype.startswith("image/") or "octet-stream" in ctype):
                    return None
                size = 0
                with open(dest, "wb") as f:
                    async for chunk in resp.content.iter_chunked(65536):
                        size += len(chunk)
                        if size > MAX_FILE_BYTES:
                            return None
                        f.write(chunk)
            return dest
        except Exception:
            return None


# ------------------------------------------------------- finding the images
def best_from_srcset(value):
    best, best_w = None, -1.0
    for part in re.split(r",\s+", value.strip()):
        bits = part.strip().split()
        if not bits:
            continue
        w = 0.0
        if len(bits) > 1:
            m = re.match(r"([\d.]+)[wx]", bits[1])
            if m:
                try:
                    w = float(m.group(1))
                except ValueError:
                    w = 0.0
        if w >= best_w:
            best, best_w = bits[0], w
    return best


def extract_images(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    base_tag = soup.find("base", href=True)
    base = urljoin(base_url, base_tag["href"]) if base_tag else base_url
    found = []

    def add(raw):
        if not raw:
            return
        raw = raw.strip()
        if not raw or raw.startswith(("data:", "blob:", "javascript:")):
            return
        full, _ = urldefrag(urljoin(base, raw))
        if urlparse(full).scheme in ("http", "https"):
            found.append(full)

    for img in soup.find_all("img"):
        link = img.find_parent("a")
        if link is not None and link.get("href") and IMG_EXT.search(link["href"]):
            add(link["href"])  # thumbnail linking to the full-size image
            continue
        chosen = None
        for attr in SRCSET_ATTRS:
            if img.get(attr):
                chosen = best_from_srcset(img[attr])
                if chosen:
                    break
        if not chosen:
            for attr in IMG_ATTRS:
                v = img.get(attr)
                if v and not v.strip().startswith("data:"):
                    chosen = v
                    break
        add(chosen)

    for tag in soup.find_all(style=True):
        for m in CSS_URL.finditer(tag["style"]):
            if IMG_EXT.search(m.group(1)):
                add(m.group(1))

    return list(dict.fromkeys(found))  # remove duplicate links, keep order


# ------------------------------------------------------- checking the files
def classify(path, seen):
    """Return (new_path, 'photo'|'doc') or None if the file should be skipped."""
    try:
        digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()
        if digest in seen:
            return None
        with Image.open(path) as im:
            w, h = im.size
            fmt = (im.format or "").upper()
            animated = bool(getattr(im, "is_animated", False))
    except Exception:
        return None
    if min(w, h) < MIN_SIDE:
        return None
    seen.add(digest)
    ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif"}.get(fmt, fmt.lower() or "img")
    new_path = f"{path}.{ext}"
    os.rename(path, new_path)
    photo_ok = (
        fmt in ("JPEG", "PNG", "WEBP")
        and not animated
        and w + h <= 10000
        and max(w, h) / min(w, h) <= 20
    )
    return new_path, ("photo" if photo_ok else "doc")


def group_runs(items):
    runs = []
    for path, kind in items:
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(path)
        else:
            runs.append((kind, [path]))
    return runs


# ------------------------------------------------------------ sending files
def _secs(value):
    return value.total_seconds() if hasattr(value, "total_seconds") else float(value)


async def send_run(bot, chat_id, kind, paths, caption):
    """Send one album (or a single file). Returns how many files were sent."""
    opts = dict(read_timeout=120, write_timeout=120, connect_timeout=30)
    for _ in range(3):
        try:
            if len(paths) == 1:
                if kind == "photo":
                    await bot.send_photo(chat_id, Path(paths[0]), caption=caption, **opts)
                else:
                    await bot.send_document(chat_id, Path(paths[0]), caption=caption, **opts)
            else:
                cls = InputMediaPhoto if kind == "photo" else InputMediaDocument
                media = [cls(Path(p), caption=caption if i == 0 else None) for i, p in enumerate(paths)]
                await bot.send_media_group(chat_id, media, **opts)
            return len(paths)
        except RetryAfter as e:
            await asyncio.sleep(_secs(e.retry_after) + 1)
        except TimedOut:
            return len(paths)  # upload probably went through; avoid duplicates
        except TelegramError as e:
            log.warning("album failed (%s), sending files one by one", e)
            break
    sent = 0
    for p in paths:
        for _ in range(2):
            try:
                await bot.send_document(chat_id, Path(p), **opts)
                sent += 1
                break
            except RetryAfter as e:
                await asyncio.sleep(_secs(e.retry_after) + 1)
            except TelegramError:
                break
    return sent


# ------------------------------------------------------------------- the job
async def run_job(msg, bot, url):
    chat_id = msg.chat_id
    status = await msg.reply_text("⏳ পেজ খুলছি...")

    async def say(text):
        try:
            await status.edit_text(text)
        except TelegramError:
            pass

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
            html, page_url = await fetch_html(session, url)
            urls = extract_images(html, page_url)
            if not urls:
                await say(
                    "😕 এই পেজে কোনো ছবি পাওয়া যায়নি।\n"
                    "(পেজের ছবি JavaScript দিয়ে লোড হলে আমি ধরতে পারি না।)"
                )
                return
            total_found = len(urls)
            urls = urls[:MAX_IMAGES]
            await say(f"🔎 {len(urls)} টি ছবির লিংক পাওয়া গেছে। নামাচ্ছি...")

            sem = asyncio.Semaphore(PARALLEL)
            seen, sent, skipped = set(), 0, 0
            for start in range(0, len(urls), BATCH):
                chunk = urls[start:start + BATCH]
                with tempfile.TemporaryDirectory() as tmp:
                    dests = [os.path.join(tmp, f"{start + i:04d}") for i in range(len(chunk))]
                    results = await asyncio.gather(
                        *(download_image(session, sem, u, page_url, d) for u, d in zip(chunk, dests))
                    )
                    items = []
                    for res in results:
                        info = classify(res, seen) if res else None
                        if info is None:
                            skipped += 1
                        else:
                            items.append(info)
                    caption = f"📷 ছবি {start + 1}–{start + len(chunk)} / {len(urls)}"
                    for kind, paths in group_runs(items):
                        sent += await send_run(bot, chat_id, kind, paths, caption)
                        caption = None
                        await asyncio.sleep(1)
                await say(f"📤 পাঠানো হচ্ছে... {min(start + BATCH, len(urls))}/{len(urls)}")

            text = f"✅ শেষ! {sent} টি ছবি পাঠানো হয়েছে।"
            if skipped:
                text += f"\n({skipped} টি বাদ গেছে: খুব ছোট, ডুপ্লিকেট বা নামানো যায়নি)"
            if total_found > len(urls):
                text += f"\n(পেজে মোট {total_found} টি ছিল, প্রথম {len(urls)} টি নেওয়া হয়েছে)"
            await say(text)
    except ValueError as e:
        await say(f"❌ {e}")
    except (aiohttp.ClientError, asyncio.TimeoutError):
        await say("❌ পেজ খুলতে পারিনি (সাইট সাড়া দিচ্ছে না বা ব্লক করেছে)।")
    except Exception:
        log.exception("job failed")
        await say("❌ অপ্রত্যাশিত সমস্যা হয়েছে। একটু পরে আবার চেষ্টা করো।")


# ----------------------------------------------------------------- handlers
def allowed(update: Update) -> bool:
    if not ALLOWED_USERS:
        return True
    user = update.effective_user
    return bool(user and user.id in ALLOWED_USERS)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        await update.message.reply_text("⛔ এই বট ব্যবহারের অনুমতি তোমার নেই।")
        return
    await update.message.reply_text(
        "👋 যেকোনো ওয়েবসাইটের পেজের লিংক পাঠাও।\n"
        "আমি পেজের সব ছবি নামিয়ে ১০টা করে অ্যালবামে পাঠিয়ে দেব।"
    )


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"তোমার ইউজার আইডি: {update.effective_user.id}")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text:
        return
    if not allowed(update):
        await msg.reply_text("⛔ এই বট ব্যবহারের অনুমতি তোমার নেই।")
        return
    m = URL_RE.search(msg.text)
    if not m:
        await msg.reply_text("কোনো ওয়েবসাইটের পেজের লিংক পাঠাও (http বা https দিয়ে শুরু)।")
        return
    url = m.group(0).rstrip(").,;]")
    if JOB_LOCK.locked():
        await msg.reply_text("⏳ আরেকটা কাজ চলছে। শেষ হলে লিংকটা আবার পাঠাও।")
        return
    async with JOB_LOCK:
        await run_job(msg, context.bot, url)


# --------------------------------------------------------------------- main
async def health(_request):
    return web.Response(text="ok")


async def main():
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)
        .read_timeout(60)
        .write_timeout(120)
        .connect_timeout(30)
        .pool_timeout(30)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    web_app = web.Application()
    web_app.router.add_get("/", health)
    web_app.router.add_get("/health", health)
    runner = web.AppRunner(web_app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", PORT).start()
    log.info("health server on port %s", PORT)

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    log.info("bot started")
    try:
        await asyncio.Event().wait()
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
