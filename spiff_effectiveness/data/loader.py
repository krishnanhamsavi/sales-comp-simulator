"""DuckDB views over the Parquet tables.

The estimator only ever sees what a real commercial-ops team would have.
latent_skill is generated but never selected here, and the planted ground
truth lives in a separate folder that this loader does not touch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from .. import config as C

REP_COLUMNS = ["rep_id", "segment", "region", "tenure_months", "quota_annual"]


@dataclass(frozen=True)
class PanelData:
    reps: pd.DataFrame
    bookings: pd.DataFrame
    programs: pd.DataFrame
    payouts: pd.DataFrame

    def __hash__(self):          # lets Streamlit cache on the object identity
        return id(self)


def ensure_data(path: Path = C.DATA_DIR):
    if not (path / "bookings_monthly.parquet").exists():
        from ..generate import programs
        programs.write(programs.build(), path)


def load(path: Path = C.DATA_DIR) -> PanelData:
    ensure_data(path)
    con = duckdb.connect()
    p = lambda name: str(Path(path) / f"{name}.parquet").replace("\\", "/")  # noqa: E731
    con.execute(f"CREATE VIEW reps AS SELECT {', '.join(REP_COLUMNS)} FROM read_parquet('{p('reps')}')")
    for t in ("bookings_monthly", "spiff_programs", "spiff_payouts"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{p(t)}')")
    data = PanelData(
        reps=con.execute("SELECT * FROM reps ORDER BY rep_id").df(),
        bookings=con.execute("SELECT * FROM bookings_monthly ORDER BY rep_id, period").df(),
        programs=con.execute("SELECT * FROM spiff_programs ORDER BY program_id").df(),
        payouts=con.execute("SELECT * FROM spiff_payouts ORDER BY program_id, rep_id, period").df(),
    )
    con.close()
    return _normalise(data)


def from_tables(tables) -> PanelData:
    """In-memory path used by the simulation tests. Same column contract as load()."""
    return _normalise(PanelData(reps=tables.reps[REP_COLUMNS].copy(), bookings=tables.bookings_monthly.copy(),
                                programs=tables.spiff_programs.copy(), payouts=tables.spiff_payouts.copy()))


def _normalise(d: PanelData) -> PanelData:
    for df, cols in ((d.bookings, ["period"]), (d.payouts, ["period"]), (d.programs, ["start_period", "end_period"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c])
    return d
