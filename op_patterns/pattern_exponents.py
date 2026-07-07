from sympy.core.expr import Expr
from sympy.core.sympify import sympify
from sympy.core.singleton import S
from collections.abc import Sequence

is_nat_n = lambda *xs: all(x.is_integer and x.is_nonnegative for x in xs)

def diff_neighbor(x, y):
    d = y - x 
    if d.is_Integer:
        return d
    if d.is_integer == False:
        raise ValueError("Fermionic operators can only be raised to a"
            " positive integer power")
    if x == 0 or (d >= 0) == True:
        return 'g'
    if (d <= 0) == True or y == 0:
        return 'l'
    return None

def _pe_flatten(args):
    for arg in args:
        if isinstance (arg, Sequence):
                yield from _pe_flatten(arg)
        elif isinstance(arg, PatExps):
                yield from _pe_flatten(arg.args)
        else:
            yield sympify(arg)

class PatExps(Expr):
    """
    Fermion/Boson pattern exponents container
    """
    def __new__(cls, *args, **hints):
        if hints.get('flatten', False):
            return Expr.__new__(cls, *_pe_flatten(args))
        elif len(args) == 1:
            if isinstance(args[0], cls):
                return args[0]
            if isinstance(args[0], Sequence):
                return Expr.__new__(cls, *(sympify(a) for a in args[0]))
        return Expr.__new__(cls, *(sympify(a) for a in args))
    
    @property
    def len(self):
        return len(self.args)
    
    def parity_diff(self, la=True):
        total = S.Zero
        for n in self.args:
            if la:
                total -= n
            else:
                total += n
            la = not la
        return total
    
    def drop_zeros(self):
        from itertools import takewhile
        zero_count1 = sum(1 for _ in takewhile(lambda x: x == 0, self.args))
        flip = (zero_count1) & 1
        zero_count2 = sum(1 for _ in takewhile(lambda x: x == 0, reversed(self.args)))
            
        if zero_count1 + zero_count2 > 0:
            return flip, PatExps(*self.args[(zero_count1):-zero_count2])
        return (flip, self)
    
    def in_new(self, f=False):
        """
        Low-cost simplification algorithm for the exponents
        which both keeps the equivalent essential arguments and 
        guarantees the simpliest form if all the exponents are given
        
        If both 2 and illegal exponent are in the exponents,
        whether to force zero or raise error will depend on
        their left-to-right sequence

        There are more to simplify if the possible values of exponents
        can be specified, but this is usually expensive
        """
        flip, zd = self.drop_zeros()
        new = []
        zero_end = False
        if f:
            equal_end = False
            for exponent in zd.args:
                if (exponent < 0) == True or exponent.is_integer == False:
                    raise ValueError("Fermionic operators can only be raised to a"
                        " positive integer power")
                if (exponent > 1) == True and exponent.is_integer:
                    return 0, None
                if zero_end:
                    if exponent == 0:
                        new.pop()
                    elif is_nat_n(new[-2], exponent):
                        new.pop()
                        new[-1] += exponent
                        if (new[-1] > 1) == True:
                            return 0, None
                        if len(new) > 1:
                            equal_end = (new[-2] == new[-1])
                    else:
                        new.append(exponent)
                        equal_end = False
                    zero_end = False
                elif new and exponent == new[-1]:
                    if equal_end:
                        new.pop()
                        equal_end = False
                    else:
                        new.append(exponent)
                        equal_end = True
                else:
                    new.append(exponent)
                    if exponent == 0:
                        zero_end = True
                    else:
                        equal_end = False
        else:
            for exponent in zd.args:
                if not isinstance(exponent, Expr):
                    raise TypeError('Expect Expr as exponents, got %s' % type(exponent))
                if exponent == 0:
                    zero_end = True
                    continue
                if zero_end:
                    new[-1] += exponent
                    zero_end = False
                elif exponent == 0:
                    zero_end = True
                else:
                    new.append(exponent)
        return flip, PatExps(*new)
    
    def f_simp(self, **hints):
        cond = (lambda _: True) if hints.get('force', False) else is_nat_n
        if hints.get('deep', True):
            flip, exponents = self.doit(**hints).drop_zeros()
        else:
            flip, exponents = self.drop_zeros()
        new_exponents, zero_at, diff_mark = [], [-2], dict()
        for exponent in exponents.args:
            if (exponent < 0) == True or exponent.is_integer == False:
                raise ValueError("Fermionic operators can only be raised to a"
                    " positive integer power")
            c1 = cond(exponent)
            if (exponent > 1) == True and c1:
                return 0, None
            if zero_at[-1] == len(new_exponents):
                if exponent == 0:
                    del new_exponents[-1]
                    del zero_at[-1]
                    continue
                elif c1 and cond(new_exponents[-2]):
                    del new_exponents[-1]
                    del zero_at[-1]
                    exponent += new_exponents.pop()
                    if (exponent > 1) == True:
                        return 0, None
                else:
                    new_exponents.append(exponent)
                    continue
            end_diff = diff_mark.get(len(new_exponents)-1, None)
            if end_diff is None:
                d = diff_neighbor(new_exponents[-1], exponent) if new_exponents else None
            while end_diff is not None:
                d = diff_neighbor(new_exponents[-1], exponent)
                if isinstance(d, Expr) and d.is_integer == False:
                    raise ValueError("Fermionic operators can only be raised to a"
                        " positive integer power")
                c2 = c1 and cond(new_exponents[-1])
                if isinstance(d, Expr) and d.is_integer and (abs(d) > 1) == True:
                    return 0, None
                c3 = c2 and cond(new_exponents[-2])
                if end_diff == 0:
                    if d == 0:
                        del new_exponents[-1]
                        del diff_mark[len(new_exponents)]
                        break
                    elif c2 and (d == 'g' or d == 1):
                        del new_exponents[-2:]
                        end_diff = diff_mark.get(len(new_exponents)-1, None)
                        continue
                elif c3:
                    if end_diff == -1:
                        if d == 1 or d == -1:
                            return 0, None
                        if d == 0 or d == 'l' or exponent == 0:
                            del new_exponents[-1]
                            del diff_mark[len(new_exponents)]
                            break
                    if end_diff == 'l' and d == 0:
                        del new_exponents[-1]
                        del diff_mark[len(new_exponents)]
                        break
                    if end_diff == 1:
                        if d == 1:
                            return 0, None
                        if d == -1:
                            exponent = 0
                            # If exponent = 1, then new_exponents[-1] = 2
                            # will null the whole expression
                    elif d == 1:
                        del new_exponents[-1]
                        end_diff = diff_mark.get(len(new_exponents), None)
                        exponent += new_exponents.pop()
                        if (exponent > 1) == True:
                            return 0, None
                        continue
                end_diff = None
            else:
                if exponent == 0:
                    zero_at.append(len(new_exponents))
                elif d is not None:
                    diff_mark[len(new_exponents)] = d
                new_exponents.append(exponent)
        if diff_mark.get(1, 0) == 1:
            del new_exponents[0]
            flip = not flip
        if diff_mark.get(len(new_exponents)-1, 0) == -1:
            del new_exponents[-1]
        return flip, PatExps(*new_exponents)

    def cat(self, tail, merge_ends, f=False):
        if isinstance(tail, PatExps):
            tail = tail.args
        if not tail:
            return self
        head = self.args
        if f:
            if merge_ends:
                if is_nat_n(head[-1]) and is_nat_n(tail[0]):
                    m = head[-1] + tail[0]
                    if (m > 1) == True:
                        return None
                    if len(head) > 1 and m == head[-2]:
                        if (len(head) > 2 and m == head[-3]
                            or len(tail) > 1 and m == tail[1]):
                            return PatExps(*head[:-2], *tail[1:])
                    elif len(tail) > 2 and m == tail[1] == tail[2]:
                        return PatExps(*head[:-1], *tail[2:])
                    else:
                        return PatExps(*head[:-1], m, *tail[1:])
                else:
                    return PatExps(*head, 0, *tail)
            elif head[-1] == tail[0]:
                if(len(head) > 1 and head[-2] == head[-1]
                    or len(tail) > 1 and tail[0] == tail[1]):
                    return PatExps(*head[:-1], *tail[1:])
            return PatExps(*head, *tail)
        elif merge_ends:
            return PatExps(*head[:-1], head[-1] + tail[0], *tail[1:])
        else:
            return PatExps(*head, *tail)

    def _sympystr(self, printer):
        return printer._print(self.args)
