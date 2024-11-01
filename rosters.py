from io import StringIO
import re

import statsapi
import polars as pl

all_data = pl.read_parquet("./data/all_data.parquet").with_columns(
    batting_team=pl.when(pl.col("inning_topbot") == "Bot")
    .then(pl.col("home_team"))
    .otherwise(pl.col("away_team")),
    pitching_team=pl.when(pl.col("inning_topbot") == "Top")
    .then(pl.col("home_team"))
    .otherwise(pl.col("away_team")),
)

pp_rosters = all_data.select(
    team=pl.col("batting_team"),
    season=pl.col("game_year"),
    player=pl.col("batter_name"),
    player_type=pl.lit("position_player"),
).unique()
pitcher_rosters = all_data.select(
    team=pl.col("pitching_team"),
    season=pl.col("game_year"),
    player=pl.col("pitcher_name"),
    player_type=pl.lit("pitcher"),
).unique()

total_rosters = pl.concat([pp_rosters, pitcher_rosters])

batting_orders = (
    all_data.sort(["at_bat_number"], descending=False)
    .unique(["game_pk", "batting_team", "batter_name"], keep="first")
    .with_columns(
        batting_order=pl.cum_count("at_bat_number").over(
            partition_by=["game_pk", "batting_team"], order_by="at_bat_number"
        )
    )
    .filter(pl.col("batting_order") <= 9, pl.col("game_pk") == 661965)
    .sort(["batting_team", "batting_order"])
)

print(batting_orders)
