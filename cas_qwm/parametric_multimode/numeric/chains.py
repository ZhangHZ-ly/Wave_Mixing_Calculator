"""Weak-field ladders of the exact numerical evolution."""
from dataclasses import dataclass

__all__ = ['Chain', 'collective', 'pair', 'degenerate']


@dataclass(frozen=True)
class Chain:
    """
    A group of weak modes with a common phase mismatch, evolving on su(1,1) ladders |k, m> of
    pair excitations m, raised by K_+ = Σ a_1† a_2† (a_1†² / 2 for a degenerate mode), with
    K_0 = Σ(1 + N_1 + N_2) / 2 = k + m; the groups are coupled only through the drive.

    kind: 'collective' (n vacuum pairs summed over, k = n / 2, n any positive real), 'pair' (one
    signal/idler pair) or 'degenerate' (one degenerate mode); occ: default Fock input of an
    explicit chain, (n_1, n_2) or n.
    """
    kind: str
    mismatch: float = 0.0
    n: float = 1
    occ: tuple = ()

    @property
    def explicit(self):
        return self.kind != 'collective'

    def sector(self, occ=None):
        """(k, m, d) of a Fock input: N_1 = m + d_1, N_2 = m + d_2 (N = 2 m + d_1 if degenerate)."""
        if self.kind == 'collective':
            return self.n / 2, 0, (0, 0)
        occ = self.occ if occ is None else occ
        if self.kind == 'pair':
            m = min(occ)
            return (1 + abs(occ[0] - occ[1])) / 2, m, (occ[0] - m, occ[1] - m)
        return 0.25 + 0.5 * (occ % 2), occ // 2, (occ % 2, occ % 2)


def collective(n_modes, mismatch=0.0) -> Chain:
    """n_modes pairs in vacuum, summed over (k = n_modes / 2)."""
    if n_modes <= 0:
        raise ValueError('The mode number of a collective chain must be positive')
    return Chain('collective', mismatch, n_modes)


def pair(n1=0, n2=0, mismatch=0.0) -> Chain:
    """One signal/idler pair, by default in the Fock state |n1, n2>."""
    return Chain('pair', mismatch, 1, (n1, n2))


def degenerate(n=0, mismatch=0.0) -> Chain:
    """A degenerate mode (signal = idler), by default in the Fock state |n>."""
    return Chain('degenerate', mismatch, 1, n)
