import sys, os
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tilt_force_balance.py')
os.chdir(os.path.dirname(src))
t = open(src, encoding='utf-8').read()
extra = '''
    Hq = Q @ H @ Q
    bare = e @ Hq @ e
    print(f"   bare tilt stiffness e.H.e = {bare:.2f} eV/A^2 ; relaxed {1/comp:.2f} ; softening by other coordinates {bare-1/comp:.2f}")
    # species-pair decomposition of bare stiffness
    E = e.reshape(N, 3)
    Hb = Hq.reshape(N, 3, N, 3)
    contrib = {}
    for i in range(N):
        for j in range(N):
            c = E[i] @ Hb[i, :, j, :] @ E[j]
            key = '-'.join(sorted([sp[i], sp[j]])) + ('(self)' if i == j else '')
            contrib[key] = contrib.get(key, 0.0) + c
    print('   pair contributions:', {k: round(v, 2) for k, v in sorted(contrib.items())})
    w_sp = {s: float(sum((E[i] @ E[i]) for i in range(N) if sp[i] == s)) for s in set(sp)}
    print('   tilt weight by species:', {k: round(v, 3) for k, v in w_sp.items()})
'''
t = t.replace('    print(f"   |f| = {np.linalg.norm(f):.4f}', extra + '    print(f"   |f| = {np.linalg.norm(f):.4f}')
exec(compile(t, src, 'exec'), {'__file__': src, '__name__': '__main__'})
