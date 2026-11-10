"""Orthographic polyhedral renders for the figures: depth-sorted translucent polyhedra, shaded atoms and displacement
arrows, drawn with matplotlib patches so the output stays vector and in the figure style."""
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from scipy.spatial import ConvexHull


def view_matrix(view, up=(0, 0, 1)):
    """Rows: screen x, screen y, depth (towards the viewer) for a view along -view."""
    d = np.asarray(view, float); d /= np.linalg.norm(d)
    u = np.asarray(up, float); u = u - (u @ d) * d; u /= np.linalg.norm(u)
    return np.array([np.cross(u, d), u, d])


def _mix(color, other, t):
    import matplotlib.colors as mc
    a, b = np.array(mc.to_rgb(color)), np.array(mc.to_rgb(other))
    return tuple((1 - t) * a + t * b)


class Scene:
    def __init__(self, M):
        self.M = M; self.items = []   # (depth, kind, payload)

    def p(self, r):
        return np.asarray(r) @ self.M.T

    def polyhedron(self, pts, face, edge, alpha=0.35, lw=0.4):
        pts = np.asarray(pts); hull = ConvexHull(pts); q = self.p(pts)
        for s in hull.simplices:
            n = np.cross(q[s[1]] - q[s[0]], q[s[2]] - q[s[0]])
            shade = 0.25 * abs(n[2]) / (np.linalg.norm(n) + 1e-12)
            self.items.append((q[s, 2].mean(), "face", (q[s, :2], _mix(face, "white", shade), edge, alpha, lw)))

    def atom(self, r, radius, color, lw=0.5, z_bias=0.0):
        q = self.p(r); self.items.append((q[2] + z_bias, "atom", (q[:2], radius, color, lw)))

    def bond(self, r0, r1, color, lw=0.8):
        q0, q1 = self.p(r0), self.p(r1)
        self.items.append(((q0[2] + q1[2]) / 2, "bond", (q0[:2], q1[:2], color, lw)))

    def arrow(self, r, d, color, lw=1.0, head=4.0, min_len=0.05):
        q = self.p(r); e = (np.asarray(d) @ self.M.T)[:2]
        if np.linalg.norm(e) < min_len:
            return
        self.items.append((q[2] + 0.5, "arrow", (q[:2], q[:2] + e, color, lw, head)))

    def draw(self, ax):
        for _, kind, pl in sorted(self.items, key=lambda t: t[0]):
            if kind == "face":
                xy, fc, ec, a, lw = pl
                ax.add_patch(Polygon(xy, closed=True, fc=fc, ec="none", alpha=a, zorder=2))
                ax.add_patch(Polygon(xy, closed=True, fc="none", ec=ec, lw=lw, alpha=min(1, a + 0.35), zorder=2))
            elif kind == "atom":
                xy, rad, c, lw = pl
                ax.add_patch(Circle(xy, rad, fc=c, ec="black", lw=lw, zorder=2))
                ax.add_patch(Circle(xy + [-0.3 * rad, 0.3 * rad], 0.35 * rad, fc=_mix(c, "white", 0.65), ec="none", zorder=2))
            elif kind == "bond":
                a, b, c, lw = pl
                ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, solid_capstyle="round", zorder=2)
            else:
                a, b, c, lw, h = pl
                ax.add_patch(FancyArrowPatch(a, b, arrowstyle=f"-|>,head_length={h * 0.1},head_width={h * 0.06}",
                                             color=c, lw=lw, mutation_scale=10, shrinkA=0, shrinkB=0, zorder=3))
        ax.set_aspect("equal"); ax.axis("off"); ax.margins(0.01); ax.autoscale_view()
