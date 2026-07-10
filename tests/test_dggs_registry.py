import pytest

from greenpy.dggs import RESOLUTION_RANGES, SYSTEM_NAMES, get_system


@pytest.mark.parametrize("name", SYSTEM_NAMES)
def test_get_system_roundtrip(name):
    system = get_system(name)
    assert system.name == name
    assert (system.min_resolution, system.max_resolution) == RESOLUTION_RANGES[name]


def test_get_system_unknown():
    with pytest.raises(ValueError, match="Unknown DGGS"):
        get_system("hexbin")


@pytest.mark.parametrize("name", SYSTEM_NAMES)
def test_column_name(name):
    lo, _ = RESOLUTION_RANGES[name]
    assert get_system(name).column_name(lo) == f"{name}_{lo}"


@pytest.mark.parametrize("name", SYSTEM_NAMES)
def test_validate_resolution_bounds(name):
    system = get_system(name)
    system.validate_resolution(system.min_resolution)
    system.validate_resolution(system.max_resolution)
    for bad in (system.min_resolution - 1, system.max_resolution + 1, None, 1.5):
        with pytest.raises(ValueError):
            system.validate_resolution(bad)
