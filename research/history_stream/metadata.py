"""Fresh-frame history feasibility, without opening images or importing Torch."""
from bisect import bisect_left


def history_candidates(visible, excluded, max_age=64):
    """Count history/search pairs with two distinct earlier template frames.

    Both templates can be shared by the current and historical contexts.
    The previous-search frame supplies the shared crop geometry and must also
    be fresh and visible. It may be the historical image itself.
    """
    assert max_age > 0
    excluded = set(excluded)
    assert all(0 <= frame < len(visible) for frame in excluded)
    fresh = [frame for frame, valid in enumerate(visible)
             if valid and frame not in excluded]
    fresh_set = set(fresh)
    searches, pairs, first = 0, 0, None
    for index, search in enumerate(fresh):
        if search - 1 not in fresh_set:
            continue
        # Historical images need two distinct earlier fresh templates.
        left = max(2, bisect_left(fresh, search - max_age))
        count = index - left
        if count <= 0:
            continue
        searches += 1
        pairs += count
        if first is None:
            first = dict(template_frames=fresh[:2],
                history_template_frames=fresh[:2], search_frame=search,
                history_frame=fresh[left], crop_reference_frame=search - 1,
                history_visible=True)
    return dict(fresh_visible_frames=len(fresh), eligible_searches=searches,
                history_search_pairs=pairs, first_candidate=first)
