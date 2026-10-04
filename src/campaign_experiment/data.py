"""Acquire, verify and validate the Hillstrom email campaign dataset.

The file is small (4 MB raw, 0.4 MB gzipped), so it is downloaded once into
a git-ignored cache and verified by SHA-256 on every load. Nothing is
committed. The canonical source is plain HTTP, which is why integrity rests
on the pinned digest rather than on the transport.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from campaign_experiment.config import Config
from campaign_experiment.logger import get_logger

log = get_logger(__name__)

COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]
BINARY_COLUMNS = ["mens", "womens", "newbie", "visit", "conversion"]
CATEGORY_DOMAINS = {
    # "Surburban" is the source's own spelling. It is kept, not corrected, so
    # every figure here can be checked against the published file.
    "zip_code": {"Rural", "Surburban", "Urban"},
    "channel": {"Multichannel", "Phone", "Web"},
}
# history_segment labels and the [lower, upper) dollar band each one claims.
HISTORY_BANDS = {
    "1) $0 - $100": (0, 100),
    "2) $100 - $200": (100, 200),
    "3) $200 - $350": (200, 350),
    "4) $350 - $500": (350, 500),
    "5) $500 - $750": (500, 750),
    "6) $750 - $1,000": (750, 1000),
    "7) $1,000 +": (1000, float("inf")),
}


class DataDownloadError(RuntimeError):
    """No source produced a file with the expected digest."""


class DataValidationError(ValueError):
    """The data does not satisfy the schema or an integrity rule."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("data validation failed:\n- " + "\n- ".join(problems))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, timeout: int, retries: int) -> None:
    """Download to a temporary file and rename, so a failure never leaves a partial file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as out:
                while block := response.read(1 << 16):
                    out.write(block)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                wait = 2 ** (attempt - 1)
                log.warning(
                    "download attempt %d/%d failed (%s); retrying in %ds",
                    attempt,
                    retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
    raise DataDownloadError(f"could not download {url}: {last_error}")


def _sources(cfg: Config) -> list[tuple[str, str, str]]:
    """(url, expected sha256, cache file name) for the primary source, then the mirror."""
    d = cfg.data
    return [
        (d.primary_url, d.primary_sha256, "hillstrom.csv"),
        (d.mirror_url, d.mirror_sha256, "hillstrom_mirror.csv.gz"),
    ]


def ensure_dataset(cfg: Config) -> Path:
    """Return a verified local copy, downloading only when no verified copy is cached."""
    cache = cfg.resolve(cfg.data.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    for _url, expected, name in _sources(cfg):
        path = cache / name
        if path.exists():
            if sha256_of(path) == expected:
                log.info("using cached %s", path.name)
                return path
            log.warning("cached %s fails its checksum; deleting it", path.name)
            path.unlink()

    errors = []
    for url, expected, name in _sources(cfg):
        path = cache / name
        try:
            log.info("downloading %s", url)
            _fetch(url, path, cfg.data.timeout_seconds, cfg.data.retries)
        except DataDownloadError as exc:
            errors.append(str(exc))
            continue
        actual = sha256_of(path)
        if actual != expected:
            path.unlink()
            errors.append(f"{url}: sha256 {actual} does not match pinned {expected}")
            log.error("checksum mismatch for %s; file deleted", url)
            continue
        log.info("verified %s (sha256 %s...)", path.name, actual[:12])
        return path
    raise DataDownloadError("no source produced a verified file:\n" + "\n".join(errors))


def read_raw(path: Path) -> pd.DataFrame:
    """Read the CSV (plain or gzipped) without interpreting any value yet."""
    return pd.read_csv(path)


def validate(df: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """Check schema and integrity rules; collect every problem before raising.

    Rows are never dropped. There is no customer id, so identical rows are
    distinct customers who happen to share attributes, and removing them
    would silently change the experiment's sample.
    """
    problems: list[str] = []

    missing = [c for c in COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in COLUMNS]
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        problems.append(f"unexpected columns: {extra}")
    if problems:
        raise DataValidationError(problems)

    if df.empty:
        raise DataValidationError(["dataset has no rows"])

    nulls = df[COLUMNS].isna().sum()
    for col, n in nulls[nulls > 0].items():
        problems.append(f"{col}: {n} missing values")

    for col in ["recency", "history", "spend", *BINARY_COLUMNS]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col}: expected numeric, got {df[col].dtype}")
    if problems:
        raise DataValidationError(problems)

    for col in BINARY_COLUMNS:
        bad = ~df[col].isin([0, 1])
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} values outside {{0, 1}}")

    unknown_arms = set(df["segment"].unique()) - set(arms)
    if unknown_arms:
        problems.append(f"segment: unknown arms {sorted(unknown_arms)}")
    absent_arms = set(arms) - set(df["segment"].unique())
    if absent_arms:
        problems.append(f"segment: arms with no rows {sorted(absent_arms)}")

    for col, domain in CATEGORY_DOMAINS.items():
        unknown = set(df[col].dropna().unique()) - domain
        if unknown:
            problems.append(f"{col}: unknown values {sorted(unknown)}")

    if not df["recency"].between(1, 12).all():
        problems.append("recency: values outside 1..12 months")
    if (df["history"] < 0).any():
        problems.append("history: negative pre-period spend")
    if (df["spend"] < 0).any():
        problems.append("spend: negative values")

    # Outcome consistency: a purchase requires a visit, and spend is positive
    # exactly when there was a purchase.
    if ((df["conversion"] == 1) & (df["visit"] == 0)).any():
        problems.append("conversion without a visit")
    if ((df["spend"] > 0) != (df["conversion"] == 1)).any():
        problems.append("spend > 0 must coincide with conversion == 1")

    # Every customer bought from at least one product line in the pre-period.
    if ((df["mens"] == 0) & (df["womens"] == 0)).any():
        problems.append("customers with neither a mens nor a womens purchase")

    unknown_bands = set(df["history_segment"].unique()) - set(HISTORY_BANDS)
    if unknown_bands:
        problems.append(f"history_segment: unknown labels {sorted(unknown_bands)}")
    else:
        lower = df["history_segment"].map(lambda s: HISTORY_BANDS[s][0])
        upper = df["history_segment"].map(lambda s: HISTORY_BANDS[s][1])
        inconsistent = ~((df["history"] >= lower) & (df["history"] < upper))
        if inconsistent.any():
            problems.append(
                f"history_segment disagrees with history on {int(inconsistent.sum())} rows"
            )

    if problems:
        raise DataValidationError(problems)
    return df


def add_features(df: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """Derived pre-period columns used for segments and balance checks."""
    out = df.copy()
    out["arm"] = pd.Categorical(out["segment"], categories=arms, ordered=True)
    out["purchase_line"] = "both"
    out.loc[(out["mens"] == 1) & (out["womens"] == 0), "purchase_line"] = "mens_only"
    out.loc[(out["mens"] == 0) & (out["womens"] == 1), "purchase_line"] = "womens_only"
    out["newbie"] = out["newbie"].astype(int)
    for col in ["visit", "conversion"]:
        out[col] = out[col].astype(int)
    out["spend"] = out["spend"].astype(float)
    out["history"] = out["history"].astype(float)
    return out


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Download (if needed), verify, validate and enrich: the single entry point."""
    path = ensure_dataset(cfg)
    df = validate(read_raw(path), cfg.experiment.arms)
    log.info("validated %d rows across %d arms", len(df), df["segment"].nunique())
    return add_features(df, cfg.experiment.arms)
