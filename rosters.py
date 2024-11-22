from io import StringIO
import re
import pprint

import statsapi
import polars as pl
from tqdm import tqdm

from pybaseball import team_ids

team_mappings = {
    "TOR": 141,
    "SEA": 136,
    "TB": 139,
    "HOU": 117,
    "ATL": 144,
    "CHC": 112,
    "CWS": 145,
    "AZ": 109,
    "CLE": 114,
    "PHI": 143,
    "KC": 118,
    "SD": 135,
    "MIL": 158,
    "TEX": 140,
    "LAA": 108,
    "MIA": 146,
    "OAK": 133,
    "SF": 137,
    "STL": 138,
    "PIT": 134,
    "COL": 115,
    "NYY": 147,
    "BAL": 110,
    "LAD": 143,
    "BOS": 111,
    "DET": 116,
    "CIN": 113,
    "NYM": 121,
    "MIN": 142,
    "WSH": 120,
}

team_dates = (
    pl.read_parquet("./data/all_data.parquet")
    .unique(["home_team", "away_team", "game_date"])
    .unpivot(on=["home_team", "away_team"], index="game_date", value_name="team")
    .select(
        pl.col("game_date"),
        pl.col("game_date").dt.to_string("%m/%d/%Y").alias("date_string"),
        pl.col("team"),
        pl.col("team").replace(team_mappings).alias("teamId"),
    )
    .sort("game_date")
)

rosters = []


for i, row in tqdm(enumerate(team_dates.iter_rows(named=True))):
    try:
        result = statsapi.get(
            "team_roster",
            {
                "teamId": row["teamId"],
                "rosterType": "active",
                "date": row["date_string"],
            },
        )["roster"]
    except:
        continue

    ids = [x["person"]["id"] for x in result]
    names = [x["person"]["fullName"] for x in result]
    positions = ["P" if x["position"]["code"] == "1" else "B" for x in result]
    result_df = pl.DataFrame(
        {
            "team": row["team"],
            "date": row["game_date"],
            "id": ids,
            "name": names,
            "position": positions,
        }
    )
    rosters.append(result_df)


rosters = pl.concat(rosters, how="vertical")
rosters.write_parquet("./data/all_rosters.parquet")
