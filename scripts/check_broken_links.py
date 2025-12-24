import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# -------------------------
# Config
# -------------------------
INPUT_CSV = os.environ.get("OUTLINKS_CSV", "data/all_outlinks.csv")
OUT_DIR = os.environ.get("OUT_DIR", "reports/broken-links")
MAX_URLS = int(os.environ.get("MAX_URLS", "2000"))          # 安全装置
TIMEOUT = int(os.environ.get("TIMEOUT", "12"))
USER_AGENT = os.environ.get("USER_AGENT", "SEO-BrokenLinkChecker/1.0")


# -------------------------
# Helpers
# -------------------------
def detect_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise ValueError(f"Missing required column. Tried: {candidates}. Found: {list(df.columns)}")

def is_http(url: str) -> bool:
    try:
        u = urlparse(str(url))
        return u.scheme in ("http", "https")
    except Exception:
        return False

def is_internal(url: str, allowed_hosts: set[str]) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return host in allowed_hosts
    except Exception:
        return False

def normalize_url(url: str) -> str:
    # 軽い正規化だけ（必要なら後で強化）
    return str(url).strip()

def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})

    retry = Retry(
        total=2,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def fetch_status(session: requests.Session, url: str) -> dict:
    """
    可能なら HEAD → ダメなら GET
    allow_redirects=True で最終到達先も取る
    """
    t0 = time.time()
    try:
        r = session.head(url, allow_redirects=True, timeout=TIMEOUT)
        # 一部サーバは HEAD を拒否するので GET にフォールバック
        if r.status_code in (403, 405) or r.status_code >= 500:
            r = session.get(url, allow_redirects=True, timeout=TIMEOUT)
        elapsed = time.time() - t0
        return {
            "status_code": int(r.status_code),
            "final_url": r.url,
            "elapsed_sec": round(elapsed, 2),
        }
    except requests.exceptions.RequestException as e:
        elapsed = time.time() - t0
        return {
            "status_code": None,
            "final_url": None,
            "elapsed_sec": round(elapsed, 2),
            "error": str(e)[:250],
        }


# -------------------------
# Main
# -------------------------
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ INPUT_CSV not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    # Screaming Frog / Sitebulb の列名揺れを吸収
    src_col = detect_col(df, ["Source", "Source URL", "From", "Address", "URL"])
    dst_col = detect_col(df, ["Destination", "Destination URL", "To", "Outlinks", "Link"])

    df = df[[src_col, dst_col]].rename(columns={src_col: "source_url", dst_col: "dest_url"})
    df["source_url"] = df["source_url"].map(normalize_url)
    df["dest_url"] = df["dest_url"].map(normalize_url)

    # http(s)だけ
    df = df[df["dest_url"].map(is_http)].copy()
    df = df[df["source_url"].map(is_http)].copy()

    # internal判定（sourceのホストを許可ホストにする）
    allowed_hosts = set(df["source_url"].map(lambda u: urlparse(u).netloc.lower()).unique())
    df = df[df["dest_url"].map(lambda u: is_internal(u, allowed_hosts))].copy()

    # 重複削除（同一source→destは1回チェックでOK）
    df = df.drop_duplicates(["source_url", "dest_url"]).copy()

    if len(df) > MAX_URLS:
        df = df.head(MAX_URLS).copy()

    session = build_session()

    rows = []
    for i, r in enumerate(df.itertuples(index=False), start=1):
        res = fetch_status(session, r.dest_url)
        rows.append({
            "source_url": r.source_url,
            "dest_url": r.dest_url,
            **res,
        })
        if i % 100 == 0:
            print(f"Checked {i}/{len(df)}")

    out = pd.DataFrame(rows)

    # 分類
    out["is_broken"] = out["status_code"].isna() | out["status_code"].isin([404, 410]) | (out["status_code"] >= 500)
    out["is_redirect"] = out["status_code"].isin([301, 302, 307, 308]) | (
        out["final_url"].notna() & (out["final_url"] != out["dest_url"])
    )

    broken = out[out["is_broken"]].copy().sort_values(["status_code", "elapsed_sec"], ascending=[True, False])
    redirects = out[out["is_redirect"] & ~out["is_broken"]].copy()

    os.makedirs(OUT_DIR, exist_ok=True)

    broken_csv = os.path.join(OUT_DIR, "broken_links.csv")
    redirects_csv = os.path.join(OUT_DIR, "redirects.csv")
    broken.to_csv(broken_csv, index=False)
    redirects.to_csv(redirects_csv, index=False)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    def md_table(d: pd.DataFrame, n=30) -> str:
        if d.empty:
            return "_No issues found_"
        view = d.head(n)[["source_url", "dest_url", "status_code", "final_url", "elapsed_sec"]].copy()
        return view.to_markdown(index=False)

    md = f"""# Internal Broken Link Report

- Generated: {now}
- Input: `{INPUT_CSV}`
- Checked pairs: **{len(out):,}**
- Broken links: **{len(broken):,}**
- Redirects: **{len(redirects):,}**

## Broken links (Top 30)
{md_table(broken, 30)}

## Redirects (Top 30)
{md_table(redirects, 30)}

## Files
- `broken_links.csv`
- `redirects.csv`
"""

    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print("✅ Done:", os.path.join(OUT_DIR, "README.md"))


if __name__ == "__main__":
    main()
