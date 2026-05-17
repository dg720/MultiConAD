from __future__ import annotations

import argparse
import getpass
import http.cookiejar
import json
import os
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse, urlunparse
from urllib.request import HTTPCookieProcessor, Request, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = Path(__file__).with_name("talkbank_media_sources.json")
DEFAULT_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".wma")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    destination: Path
    depth: int
    enabled: bool


def make_opener(username: str, password: str, timeout: int):
    jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))
    payload = json.dumps({"email": username, "pswd": password}).encode("utf-8")
    req = Request(
        "https://sla2.talkbank.org:443/logInUser",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "MultiConAD-audio-downloader/1.0"},
        method="POST",
    )
    with opener.open(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8", errors="ignore"))
    auth_status = result.get("authStatus") or {}
    if not result.get("success") or not auth_status.get("loggedIn") or not auth_status.get("authorized"):
        raise RuntimeError(f"TalkBank login failed: {result.get('respMsg') or 'unauthorized'}")
    return opener


def request_url(url: str, opener, timeout: int) -> bytes:
    req = Request(url, headers={"User-Agent": "MultiConAD-audio-downloader/1.0"})
    with opener.open(req, timeout=timeout) as response:
        return response.read()


def load_sources(path: Path, only: set[str] | None, include_disabled: bool) -> list[Source]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for item in raw["sources"]:
        if only and item["name"] not in only:
            continue
        if not include_disabled and not bool(item.get("enabled", True)):
            continue
        sources.append(
            Source(
                name=item["name"],
                url=item["url"].rstrip("/") + "/",
                destination=(PROJECT_ROOT / item["destination"]).resolve(),
                depth=int(item.get("depth", 2)),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return sources


def is_under(url: str, base_url: str) -> bool:
    parsed = urlparse(url)
    base = urlparse(base_url)
    parsed_host = parsed.hostname or ""
    base_host = base.hostname or ""
    parsed_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base_port = base.port or (443 if base.scheme == "https" else 80)
    return (
        parsed.scheme == base.scheme
        and parsed_host == base_host
        and parsed_port == base_port
        and parsed.path.startswith(base.path)
    )


def is_media_url(url: str, extensions: tuple[str, ...]) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in extensions)


def save_media_url(url: str) -> str:
    parsed = urlparse(url)
    if "f=save" in parsed.query:
        return url
    query = f"{parsed.query}&f=save" if parsed.query else "f=save"
    return urlunparse(parsed._replace(query=query))


def relative_destination(url: str, source: Source) -> Path:
    rel = unquote(urlparse(url).path[len(urlparse(source.url).path) :]).lstrip("/")
    return source.destination / rel


def discover_media(
    source: Source,
    opener,
    extensions: tuple[str, ...],
    timeout: int,
) -> list[str]:
    queue: list[tuple[str, int]] = [(source.url, 0)]
    seen: set[str] = set()
    media: list[str] = []

    while queue:
        url, depth = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        try:
            payload = request_url(url, opener, timeout)
        except HTTPError as exc:
            print(f"[{source.name}] cannot list {url}: HTTP {exc.code}", file=sys.stderr)
            continue
        except URLError as exc:
            print(f"[{source.name}] cannot list {url}: {exc.reason}", file=sys.stderr)
            continue

        parser = LinkParser()
        parser.feed(payload.decode("utf-8", errors="ignore"))

        for href in parser.links:
            if href.startswith("?") or href.startswith("#") or href in {"../", "/"}:
                continue
            child = urljoin(url, href)
            child = child.split("#", 1)[0]
            if not is_under(child, source.url):
                continue
            if is_media_url(child, extensions):
                media.append(save_media_url(child))
            elif depth < source.depth:
                path = urlparse(child).path
                if child.endswith("/") or not Path(path).suffix:
                    queue.append((child.rstrip("/") + "/", depth + 1))

    return sorted(set(media))


def download_file(url: str, destination: Path, opener, timeout: int, overwrite: bool, min_bytes: int) -> str:
    if destination.exists() and destination.stat().st_size >= min_bytes and not overwrite:
        return "skip"

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    req = Request(url, headers={"User-Agent": "MultiConAD-audio-downloader/1.0"})
    with opener.open(req, timeout=timeout) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    last_error: PermissionError | None = None
    for _ in range(10):
        try:
            tmp.replace(destination)
            break
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.5)
    else:
        raise last_error or PermissionError(f"Could not move {tmp} to {destination}")
    return "downloaded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror authorized TalkBank/DementiaBank media into data/.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--only", nargs="*", help="Source names to download, e.g. english-delaware english-pitt.")
    parser.add_argument("--include-disabled", action="store_true", help="Include sources marked enabled=false.")
    parser.add_argument("--dry-run", action="store_true", help="List matching media URLs without downloading.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local media files.")
    parser.add_argument("--min-bytes", type=int, default=1024, help="Redownload existing media smaller than this.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit files per source for smoke tests.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between downloads.")
    parser.add_argument("--extensions", nargs="*", default=list(DEFAULT_EXTENSIONS))
    parser.add_argument("--username", default=os.environ.get("TALKBANK_USER", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    username = args.username.strip()
    if not username:
        username = input("TalkBank username: ").strip()
    password = os.environ.get("TALKBANK_PASSWORD", "")
    if not password:
        password = getpass.getpass("TalkBank password: ")

    opener = make_opener(username, password, args.timeout)
    extensions = tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions)
    sources = load_sources(args.sources, set(args.only) if args.only else None, args.include_disabled)
    if not sources:
        raise SystemExit("No sources selected.")

    for source in sources:
        print(f"\n[{source.name}] listing {source.url}")
        media = discover_media(source, opener, extensions, args.timeout)
        if args.max_files:
            media = media[: args.max_files]
        print(f"[{source.name}] matched {len(media)} media files")
        for url in media:
            destination = relative_destination(url, source)
            if args.dry_run:
                print(f"DRY\t{url}\t->\t{destination}")
                continue
            try:
                status = download_file(url, destination, opener, args.timeout, args.overwrite, args.min_bytes)
                print(f"{status}\t{destination}")
            except HTTPError as exc:
                print(f"error\tHTTP {exc.code}\t{url}", file=sys.stderr)
            except URLError as exc:
                print(f"error\t{exc.reason}\t{url}", file=sys.stderr)
            if args.sleep:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
