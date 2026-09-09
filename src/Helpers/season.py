import pandas as pd

# The harvest season starts on September 1st: dates from September to
# December belong to the season named after that year, dates from January
# to August belong to the season that started the year before.
SEASON_START_MONTH = 9
SEASON_START_DAY = 1


def season_start_of(date) -> pd.Timestamp:
    """
    Return September 1st of the season the given date belongs to.
    """
    date = pd.Timestamp(date)
    year = date.year if date.month >= SEASON_START_MONTH else date.year - 1

    return pd.Timestamp(year=year, month=SEASON_START_MONTH, day=SEASON_START_DAY)


def season_of(date) -> str:
    """
    Return the season label for a date, e.g. "2025/2026".
    """
    start = season_start_of(date)

    return f"{start.year}/{start.year + 1}"


def day_of_season(date) -> int:
    """
    Return how many days into the season the given date falls, counting
    September 1st as day 1.
    """
    date = pd.Timestamp(date)

    return (date - season_start_of(date)).days + 1
