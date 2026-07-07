from propagator.io.actions import ActionType, WaterlineAction, parse_actions


def test_parse_actions_accepts_canonical_waterline():
    data = {"waterline": ["LINE:[0 1];[0 1]"]}

    actions = parse_actions(data, epsg=4326)

    assert data == {}
    assert len(actions) == 1
    assert isinstance(actions[0], WaterlineAction)
    assert actions[0].action_type == ActionType.WATERLINE


def test_parse_actions_accepts_legacy_waterline_action():
    data = {"waterline_action": ["LINE:[0 1];[0 1]"]}

    actions = parse_actions(data, epsg=4326)

    assert data == {}
    assert len(actions) == 1
    assert isinstance(actions[0], WaterlineAction)
    assert actions[0].action_type == ActionType.WATERLINE


def test_parse_actions_combines_waterline_aliases():
    data = {
        "waterline": ["LINE:[0 1];[0 1]"],
        "waterline_action": ["LINE:[2 3];[2 3]"],
    }

    actions = parse_actions(data, epsg=4326)

    assert data == {}
    assert len(actions) == 1
    assert isinstance(actions[0], WaterlineAction)
    assert len(actions[0].geometries) == 2
