import pickle
from typing import Tuple, TypeVar
from tilebox.workflows.cache import JobCache
from datetime import datetime, timedelta
import xarray as xr
from config import (
    VIIRS_COLLECTION,
    VIIRS_START_DATE,
    MODIS_COLLECTION,
    DATA_SEARCH_PRE_DAYS,
    DATA_SEARCH_POST_DAYS,
)

T = TypeVar("T")


def from_param_or_cache(
    param: T | None,
    cache: JobCache,
    cache_key: str,
    default: T | None = None,
) -> T:
    """
    Gets a value from a parameter, the cache, or a default factory.
    The value is stored in the cache if it's not already there.
    Assumes string values that need to be encoded/decoded for the cache.
    Raises ValueError if param, cache value, and default are all None.
    """
    if param is not None:
        cache[cache_key] = pickle.dumps(param)
        return param

    if cache_key in cache:
        return pickle.loads(cache[cache_key])

    if default is None:
        raise ValueError(
            f"No value found for {cache_key} in parameters, cache, or default"
        )

    cache[cache_key] = pickle.dumps(default)
    return default


def _query_fpar_metadata(dataset, time_range: tuple[datetime, datetime]) -> xr.Dataset:
    """
    Queries for FPAR metadata within a given time range.
    """
    modis = dataset.collection(MODIS_COLLECTION).query(
        temporal_extent=(time_range[0], min(time_range[1], VIIRS_START_DATE))
    )
    viirs = dataset.collection(VIIRS_COLLECTION).query(
        temporal_extent=(max(time_range[0], VIIRS_START_DATE), time_range[1])
    )

    non_empty = [ds for ds in [modis, viirs] if ds]
    if len(non_empty) == 0:
        return xr.Dataset()
    if len(non_empty) == 1:
        return non_empty[0]
    return xr.concat(non_empty, dim="time")
