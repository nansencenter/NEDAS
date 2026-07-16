from NEDAS.core import Context, Preconditioner

class Identity(Preconditioner):
    """
    No-op preconditioner (default). Matches Transform's Identity in spirit.
    """
    is_identity = True

    def pre_assimilate(self, c: Context) -> None:
        pass
