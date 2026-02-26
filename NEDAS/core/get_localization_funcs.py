from NEDAS.core import Context

registry = {
    'gaspari_cohn': ('distance_based', 'gaspari_cohn_func'),
    'step': ('distance_based', 'step_func'),
    'exponential': ('distance_based', 'exponential_func'),
}

def get_localization_funcs(c: Context) -> dict:
    local_funcs = {}
    assert c.localization_def is not None, "c.localization_def needs to be defined in the config file"
    for key in ['horizontal', 'vertical', 'temporal']:
        assert key in c.localization_def, f"{key} needs to be defined in c.localization_def"
        if c.localization_def[key]:
            if 'type' not in c.localization_def[key]:
                raise KeyError(f"'type' needs to be specified for c.localization['{key}']")
            local_funcs[key] = get_localization_func_component(c.localization_def[key]['type'])
        else:
            local_funcs[key] = None
    return local_funcs

def get_localization_func_component(localization_types):
    localization_types = localization_types.lower().split(',')

    ##distance-based localization schemes
    if 'gaspari_cohn' in localization_types:
        from .distance_based import gaspari_cohn_func as local_func
    elif 'step' in localization_types:
        from .distance_based import step_func as local_func
    elif 'exponential' in localization_types:
        from .distance_based import exponential_func as local_func

    # ##correlation based localization schemes
    # elif 'SER' in localization_types:
    #     from NEDAS.assim_tools.localization.SER import CorrelationBasedLocalization as Localization
    # elif 'NICE' in localization_types:
    #     from NEDAS.assim_tools.localization.NICE import 
    else:
        raise ValueError(f"Unknown localization type {type}")
    return local_func
