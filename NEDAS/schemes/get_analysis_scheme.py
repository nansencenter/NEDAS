
def get_analysis_scheme(c):
    if c.analysis_type == 'offline_filter':
        from NEDAS.schemes.offline_filter import OfflineFilterAnalysisScheme as Scheme
    # elif c.analysis_type == 'online_filter':
    #     from NEDAS.schemes.online_filter import OnlineFilterAnalysisScheme as Scheme
    else:
        raise NotImplementedError(f"Unknown analysis type: '{c.analysis_type}'")
    return Scheme()

