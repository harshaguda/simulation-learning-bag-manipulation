
from pxr import PhysxSchema, Sdf, Semantics, Usd, UsdGeom, UsdPhysics, UsdShade
import omni.isaac.core.utils.stage as stage_utils
from typing import TYPE_CHECKING, Any, Callable
import re 

def find_first_matching_prim(prim_path_regex: str, stage: Usd.Stage | None = None) -> Usd.Prim | None:
    """Find the first matching prim in the stage based on input regex expression.

    Args:
        prim_path_regex: The regex expression for prim path.
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        The first prim that matches input expression. If no prim matches, returns None.
    """
    # get current stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # need to wrap the token patterns in '^' and '$' to prevent matching anywhere in the string
    pattern = f"^{prim_path_regex}$"
    compiled_pattern = re.compile(pattern)
    # obtain matching prim (depth-first search)
    for prim in stage.Traverse():
        # check if prim passes predicate
        if compiled_pattern.match(prim.GetPath().pathString) is not None:
            return prim
    return None

def get_all_matching_child_prims(
    prim_path: str,
    predicate: Callable[[Usd.Prim], bool] = lambda _: True,
    depth: int | None = None,
    stage: Usd.Stage | None = None,
) -> list[Usd.Prim]:
    """Performs a search starting from the root and returns all the prims matching the predicate.

    Args:
        prim_path: The root prim path to start the search from.
        predicate: The predicate that checks if the prim matches the desired criteria. It takes a prim as input
            and returns a boolean. Defaults to a function that always returns True.
        depth: The maximum depth for traversal, should be bigger than zero if specified.
            Defaults to None (i.e: traversal happens till the end of the tree).
        stage: The stage where the prim exists. Defaults to None, in which case the current stage is used.

    Returns:
        A list containing all the prims matching the predicate.
    """
    # get current stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim at path '{prim_path}' is not valid.")
    # check if depth is valid
    if depth is not None and depth <= 0:
        raise ValueError(f"Depth must be bigger than zero, got {depth}.")

    # iterate over all prims under prim-path
    # list of tuples (prim, current_depth)
    all_prims_queue = [(prim, 0)]
    output_prims = []
    while len(all_prims_queue) > 0:
        # get current prim
        child_prim, current_depth = all_prims_queue.pop(0)
        # check if prim passes predicate
        if predicate(child_prim):
            output_prims.append(child_prim)
        # add children to list
        if depth is None or current_depth < depth:
            all_prims_queue += [(child, current_depth + 1) for child in child_prim.GetChildren()]

    return output_prims