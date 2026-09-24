def beam_splitter(a, d, r, t_coeff):
    """Beam splitter: ``a`` is the waveguide input, ``d`` the resonator pulse.

    Works for both unbatched ``(Nt,)`` and batched ``(B, Nt)`` tensors thanks
    to broadcasting (``a`` may be ``(Nt,)`` while ``d`` is ``(B, Nt)``).
    """
    b = t_coeff * a + r * d
    c = r * a + t_coeff * d
    return b, c
