"""Dynamic STRING retrieval with batching and local cache."""

from __future__ import annotations

import hashlib
import logging
import time
from io import StringIO

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from pgm.config.schemas import ProjectConfig
from pgm.utils.paths import string_cache_dir

logger = logging.getLogger("pgm.kg.string")

STRING_NETWORK = "https://string-db.org/api/tsv/network"


def cache_key(identifiers: list[str], cfg: ProjectConfig) -> str:
    base = "|".join(sorted(identifiers))
    blob = (
        f"{base}|sp={cfg.kg.species_id}|thr={cfg.kg.confidence_threshold}|"
        f"b={cfg.kg.batch_size}"
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


class StringAPIClient:
    """Minimal STRING REST client honoring plan limits."""

    def __init__(self, cfg: ProjectConfig) -> None:
        self.cfg = cfg
        self.cache_dir = string_cache_dir(cfg)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _get(self, params: dict) -> str:
        resp = requests.get(
            STRING_NETWORK,
            params=params,
            timeout=self.cfg.kg.request_timeout_seconds,
            headers={"User-Agent": "pgm-scrna/0.1 (research pipeline)"},
        )
        resp.raise_for_status()
        return resp.text

    def fetch_interactions(self, identifiers: list[str]) -> pd.DataFrame:
        """
        Retrieve interaction rows for identifiers (genes / proteins).

        Batches identifiers to respect ``batch_size``.
        Parses ``preferredName_A/B`` and ``score`` (normalized to [0,1]).
        """
        uniq = sorted({str(s).strip() for s in identifiers if str(s).strip()})
        t_net = time.perf_counter()
        if self.cfg.kg.cache_enabled:
            ck = cache_key(uniq, self.cfg)
            cpath = self.cache_dir / f"{ck}_network.parquet"
            if cpath.is_file():
                df_hit = pd.read_parquet(cpath)
                logger.info(
                    "STRING cache hit %s rows=%d %.3fs",
                    cpath.name,
                    len(df_hit),
                    time.perf_counter() - t_net,
                )
                return df_hit

        max_g = min(len(uniq), self.cfg.kg.max_genes_for_query)
        if len(uniq) > max_g:
            logger.warning(
                "STRING query truncated %d → %d genes", len(uniq), max_g
            )
            uniq = uniq[:max_g]

        logger.info(
            "STRING network request start genes=%d batch_size=%d required_score=%d cache_write=%s",
            len(uniq),
            max(5, min(self.cfg.kg.batch_size, 420)),
            int(round(self.cfg.kg.confidence_threshold * 1000)),
            self.cfg.kg.cache_enabled,
        )
        parts: list[pd.DataFrame] = []
        bs = max(5, min(self.cfg.kg.batch_size, 420))
        required = int(round(self.cfg.kg.confidence_threshold * 1000))
        for i in range(0, len(uniq), bs):
            chunk = uniq[i : i + bs]
            ident = "\r".join(chunk)
            payload = self._get(
                {
                    "identifiers": ident,
                    "species": self.cfg.kg.species_id,
                    "required_score": required,
                }
            )
            df = pd.read_csv(StringIO(payload), sep="\t")
            if not df.empty:
                parts.append(df)
            logger.info("STRING batch %d-%d ok", i, i + len(chunk))

        if not parts:
            frame = pd.DataFrame(
                columns=["preferredName_A", "preferredName_B", "score"]
            )
        else:
            frame = pd.concat(parts, ignore_index=True)
            cols = [
                c
                for c in ("preferredName_A", "preferredName_B", "score")
                if c in frame.columns
            ]
            frame = frame[cols].drop_duplicates()
            if "score" in frame.columns:
                mx = float(frame["score"].max()) if len(frame) else 1.0
                if mx > 1.5:
                    frame["score"] = frame["score"] / 1000.0
                frame["score"] = frame["score"].clip(0, 1)

        if self.cfg.kg.cache_enabled:
            ck = cache_key(uniq, self.cfg)
            cpath = self.cache_dir / f"{ck}_network.parquet"
            frame.to_parquet(cpath)
            logger.info("Wrote STRING cache %s", cpath)
        logger.info(
            "STRING fetch done %.3fs rows=%d n_batches=%d",
            time.perf_counter() - t_net,
            len(frame),
            int((len(uniq) + bs - 1) // bs) if uniq else 0,
        )
        return frame


def fetch_string_for_genes(genes: list[str], cfg: ProjectConfig) -> pd.DataFrame:
    return StringAPIClient(cfg).fetch_interactions(genes)

