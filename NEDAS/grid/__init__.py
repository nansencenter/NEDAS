from NEDAS.grid.grid import Grid
from NEDAS.grid.grid_regular import RegularGrid
from NEDAS.grid.grid_irregular import IrregularGrid
from NEDAS.grid.grid_1d import Grid1D

from typing import Union, TypeVar
GridType = TypeVar('GridType', bound=Union[RegularGrid, IrregularGrid, Grid1D])

__all__ = ['Grid', 'RegularGrid', 'IrregularGrid', 'Grid1D', 'GridType']