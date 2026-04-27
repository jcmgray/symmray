import pytest

from symmray.array_common import calc_reshape_args


def test_calc_reshape_args_edgecase():
    axs_unfuse, axs_fuse_groupings, axs_expand = calc_reshape_args(
        shape=(4, 4, 4),
        newshape=(16, 4),
        subshapes=(None, None, (4, 4)),
    )
    assert axs_unfuse == ()
    assert axs_expand == ()
    assert axs_fuse_groupings == (((0, 1),),)


@pytest.mark.parametrize(
    "shape,newshape,subshapes,expected",
    [
        # fuse middle
        (
            (3, 4, 5, 6),
            (3, -1, 6),
            (None,) * 4,
            ((), (((1, 2),),), ()),
        ),
        # fuse all
        (
            (3, 4, 5, 6),
            (-1,),
            (None,) * 4,
            ((), (((0, 1, 2, 3),),), ()),
        ),
        # trailing -1 with no remaining axes -> singleton expand
        (
            (3, 4, 5, 6),
            (3, 4, 5, 6, -1),
            (None,) * 4,
            ((), (), (4,)),
        ),
        # single middle axis -> direct match
        ((3, 4, 5, 6), (3, -1, 5, 6), (None,) * 4, ((), (), ())),
        # left boundary
        (
            (3, 4, 5, 6),
            (-1, 6),
            (None,) * 4,
            ((), (((0, 1, 2),),), ()),
        ),
        # right boundary
        (
            (3, 4, 5, 6),
            (3, -1),
            (None,) * 4,
            ((), (((1, 2, 3),),), ()),
        ),
        # division -1 inference would fail, so this must match structurally
        (
            (3, 12, 6),
            (3, 4, 5, -1),
            (None, (4, 5), None),
            ((1,), (), ()),
        ),
        # internal singleton swept up by -1
        (
            (2, 1, 3, 4),
            (2, -1),
            (None,) * 4,
            ((), (((1, 2, 3),),), ()),
        ),
        # leading singleton swept up by -1
        (
            (1, 3, 4),
            (-1, 4),
            (None,) * 3,
            ((), (((0, 1),),), ()),
        ),
    ],
)
def test_calc_reshape_args_negative_one(shape, newshape, subshapes, expected):
    assert calc_reshape_args(shape, newshape, subshapes) == expected


@pytest.mark.parametrize(
    "shape,newshape,subshapes,expected",
    [
        # expand before fuse
        (
            (2, 3),
            (1, 6),
            (None, None),
            ((), (((0, 1),),), (0,)),
        ),
        # expand between fuse and retained axis
        (
            (2, 3, 4),
            (6, 1, 4),
            (None, None, None),
            ((), (((0, 1),),), (1,)),
        ),
        # expand after fuse
        (
            (2, 3),
            (6, 1),
            (None, None),
            ((), (((0, 1),),), (1,)),
        ),
        # leading singleton squeezed into fuse group
        (
            (1, 2, 3),
            (6,),
            (None, None, None),
            ((), (((0, 1, 2),),), ()),
        ),
        # internal singleton squeezed into fuse group
        (
            (2, 1, 3),
            (6,),
            (None, None, None),
            ((), (((0, 1, 2),),), ()),
        ),
        # trailing singleton squeezed into fuse group
        (
            (2, 3, 1),
            (6,),
            (None, None, None),
            ((), (((0, 1, 2),),), ()),
        ),
        # multiple unfuses with cumulative axis shifts
        (
            (6, 20),
            (2, 3, 4, 5),
            ((2, 3), (4, 5)),
            ((0, 2), (), ()),
        ),
        # suffix-side unfuse matched after -1
        (
            (3, 20),
            (-1, 4, 5),
            (None, (4, 5)),
            ((1,), (), ()),
        ),
        # explicit singleton expansion before -1
        (
            (2, 3),
            (1, -1),
            (None, None),
            ((), (((0, 1),),), (0,)),
        ),
        # explicit singleton expansion after -1
        (
            (2, 3),
            (-1, 1),
            (None, None),
            ((), (((0, 1),),), (1,)),
        ),
        # singleton expansion on both sides of -1
        (
            (2, 3),
            (1, -1, 1),
            (None, None),
            ((), (((0, 1),),), (1, 0)),
        ),
        # prefix and suffix unfuses around -1
        (
            (6, 3, 20),
            (2, 3, -1, 4, 5),
            ((2, 3), None, (4, 5)),
            ((0, 3), (), ()),
        ),
        # prefix and suffix fuses around -1
        (
            (2, 3, 4, 5, 6),
            (6, -1, 30),
            (None, None, None, None, None),
            ((), (((0, 1),), ((2, 3),)), ()),
        ),
        # adjacent fuse groups handled in one fuse call
        (
            (2, 3, 4, 5),
            (6, 20),
            (None, None, None, None),
            ((), (((0, 1), (2, 3)),), ()),
        ),
        # separated fuse groups handled in separate fuse calls
        (
            (2, 3, 4, 5, 6),
            (6, 4, 30),
            (None, None, None, None, None),
            ((), (((0, 1),), ((2, 3),)), ()),
        ),
    ],
)
def test_calc_reshape_args_combinations(shape, newshape, subshapes, expected):
    assert calc_reshape_args(shape, newshape, subshapes) == expected


def test_calc_reshape_args_multiple_negative_one():
    with pytest.raises(ValueError):
        calc_reshape_args((3, 4), (-1, -1), (None, None))


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="partial unfuse followed by refuse is not supported yet",
)
def test_calc_reshape_args_unfuse_then_refuse():
    assert calc_reshape_args(
        shape=(6, 4, 5),
        newshape=(2, 12, 5),
        subshapes=((2, 3), None, None),
    ) == ((0,), (((1, 2),),), ())


@pytest.mark.parametrize(
    "shape,newshape,subshapes",
    [
        ((2, 3), (2,), (None, None)),
        ((1,), (), (None,)),
        ((2,), (2, -1, 2), (None,)),
    ],
)
def test_calc_reshape_args_invalid(shape, newshape, subshapes):
    with pytest.raises(ValueError):
        calc_reshape_args(shape, newshape, subshapes)
