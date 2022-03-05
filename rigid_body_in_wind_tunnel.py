"""
Flow past cylinder
"""
import logging
from time import time
import os
import numpy
import numpy as np
from numpy import pi, cos, sin, exp

from pysph.base.kernels import QuinticSpline
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation, Group
from pysph.solver.application import Application
from pysph.tools import geometry as G
from pysph.sph.bc.inlet_outlet_manager import (
    InletInfo, OutletInfo)
from pysph.sph.scheme import add_bool_argument

from edac_rigid_bodies import WCSPHRigidBodyScheme, SolidWallNoSlipBCReverse, setup_properties


logger = logging.getLogger()

# Fluid mechanical/numerical parameters
rho = 1000
umax = 1.0
u_freestream = 1.0
c0 = 10 * umax
p0 = rho * c0 * c0


def potential_flow(x, y, u_infty=1.0, center=(0, 0), diameter=1.0):
    x = x - center[0]
    y = y - center[1]
    z = x + 1j*y
    a2 = (diameter * 0.5)**2
    vel = (u_infty - a2/(z*z)).conjugate()
    u = vel.real
    v = vel.imag
    p = 500 - 0.5 * rho * np.abs(vel)**2
    return u, v, p


def compute_forces(fname, sname, nu):
    from edac import (
        MomentumEquationPressureGradient2, SummationDensityGather,
        SetWallVelocity, EvaluateNumberDensity, ComputeBeta
    )

    all_sources = [fname, sname]
    equations = []
    g1 = []
    g1.extend([
        SummationDensityGather(dest=sname, sources=all_sources),
        ComputeBeta(dest=sname, sources=all_sources, dim=2),
        EvaluateNumberDensity(dest=sname, sources=[fname]),
        SetWallVelocity(dest=sname, sources=[fname]),
    ])
    equations.append(Group(equations=g1, real=False))

    g2 = []
    g2.extend([
        MomentumEquationPressureGradient2(dest=sname, sources=[fname]),
        SolidWallNoSlipBCReverse(dest=sname, sources=[fname], nu=nu),
    ])
    equations.append(Group(equations=g2, real=True))
    return equations


class ShepardInterpolateCharacteristics(Equation):
    def initialize(self, d_idx, d_J1, d_J2u, d_J3u, d_J2v, d_J3v, d_v):
        d_J1[d_idx] = 0.0
        d_J2u[d_idx] = 0.0
        d_J3u[d_idx] = 0.0

        d_J2v[d_idx] = 0.0
        d_J3v[d_idx] = 0.0
        d_v[d_idx] = 0.0

    def loop(self, d_idx, d_J1, d_J2u, s_J1, s_J2u, d_J3u, s_J3u, s_idx, s_J2v,
             s_J3v, d_J2v, d_J3v, WIJ, s_v, d_v):
        d_J1[d_idx] += s_J1[s_idx] * WIJ
        d_J2u[d_idx] += s_J2u[s_idx] * WIJ
        d_J3u[d_idx] += s_J3u[s_idx] * WIJ

        d_J2v[d_idx] += s_J2v[s_idx] * WIJ
        d_J3v[d_idx] += s_J3v[s_idx] * WIJ

        d_v[d_idx] += s_v[s_idx] * WIJ

    def post_loop(self, d_idx, d_J1, d_J2u, d_wij, d_avg_j2u, d_avg_j1, d_J3u,
                  d_avg_j3u, d_J2v, d_J3v, d_avg_j2v, d_avg_j3v, d_v):
        if d_wij[d_idx] > 1e-14:
            d_J1[d_idx] /= d_wij[d_idx]
            d_J2u[d_idx] /= d_wij[d_idx]
            d_J3u[d_idx] /= d_wij[d_idx]

            d_J2v[d_idx] /= d_wij[d_idx]
            d_J3v[d_idx] /= d_wij[d_idx]

            d_v[d_idx] /= d_wij[d_idx]
        else:
            d_J1[d_idx] = d_avg_j1[0]
            d_J2u[d_idx] = d_avg_j2u[0]
            d_J3u[d_idx] = d_avg_j3u[0]

            d_J2v[d_idx] = d_avg_j2v[0]
            d_J3v[d_idx] = d_avg_j3v[0]

    def reduce(self, dst, t, dt):
        dst.avg_j1[0] = numpy.average(dst.J1[dst.wij > 0.0001])
        dst.avg_j2u[0] = numpy.average(dst.J2u[dst.wij > 0.0001])
        dst.avg_j3u[0] = numpy.average(dst.J3u[dst.wij > 0.0001])

        dst.avg_j2v[0] = numpy.average(dst.J2v[dst.wij > 0.0001])
        dst.avg_j3v[0] = numpy.average(dst.J3v[dst.wij > 0.0001])


class EvaluateCharacterisctics(Equation):
    def __init__(self, dest, sources, c_ref, rho_ref, u_ref, p_ref, v_ref):
        self.c_ref = c_ref
        self.rho_ref = rho_ref
        self.p_ref = p_ref
        self.u_ref = u_ref
        self.v_ref = v_ref
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_p, d_rho, d_J1, d_J2u, d_J3u, d_J2v,
                   d_J3v):
        a = self.c_ref
        rho_ref = self.rho_ref

        rho = d_rho[d_idx]
        pdiff = d_p[d_idx] - self.p_ref
        udiff = d_u[d_idx] - self.u_ref
        vdiff = d_v[d_idx] - self.v_ref

        d_J1[d_idx] = -a * a * (rho - rho_ref) + pdiff
        d_J2u[d_idx] =  rho * a * udiff + pdiff
        d_J3u[d_idx] = -rho * a * udiff + pdiff

        d_J2v[d_idx] =  rho * a * vdiff + pdiff
        d_J3v[d_idx] = -rho * a * vdiff + pdiff


class EvaluatePropertyfromCharacteristics(Equation):
    def __init__(self, dest, sources, c_ref, rho_ref, u_ref, v_ref, p_ref):
        self.c_ref = c_ref
        self.rho_ref = rho_ref
        self.p_ref = p_ref
        self.u_ref = u_ref
        self.v_ref = v_ref
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_J1, d_J2u, d_J2v, d_J3u, d_J3v, d_rho,
                   d_u, d_v, d_xn, d_yn):
        a = self.c_ref
        a2_1 = 1.0/(a*a)
        rho = d_rho[d_idx]
        xn = d_xn[d_idx]
        yn = d_yn[d_idx]

        # Characteristic in the downstream direction.
        if xn > 0.5 or yn > 0.5:
            J1 = d_J1[d_idx]
            J3 = 0.0
            if xn > 0.5:
                J2 = d_J2u[d_idx]
                d_u[d_idx] = self.u_ref + (J2 - J3) / (2 * rho * a)
            else:
                J2 = d_J2v[d_idx]
                d_v[d_idx] = self.v_ref + (J2 - J3) / (2 * rho * a)
        # Characteristic in the upstream direction.
        else:
            J1 = 0.0
            J2 = 0.0
            if xn < -0.5:
                J3 = d_J3u[d_idx]
                d_u[d_idx] = self.u_ref + (J2 - J3) / (2 * rho * a)
            else:
                J3 = d_J3v[d_idx]
                d_v[d_idx] = self.v_ref + (J2 - J3) / (2 * rho * a)

        d_rho[d_idx] = self.rho_ref + a2_1 * (-J1 + 0.5 * (J2 + J3))
        d_p[d_idx] = self.p_ref + 0.5 * (J2 + J3)


class ResetInletVelocity(Equation):
    def __init__(self, dest, sources, U, V, W):
        self.U = U
        self.V = V
        self.W = W

        super().__init__(dest, sources)

    def loop(self, d_idx, d_u, d_v, d_w, d_uref):
        if d_idx == 0:
            d_uref[0] = self.U
        d_u[d_idx] = self.U
        d_v[d_idx] = self.V
        d_w[d_idx] = self.W


class WindTunnel(Application):
    def initialize(self):
        # Geometric parameters
        self.Lt = 50.0  # length of tunnel
        self.Wt = 15.0  # half width of tunnel
        self.dc = 1.2  # diameter of cylinder
        self.nl = 10  # Number of layers for wall/inlet/outlet

        # rigid body properties
        self.rigid_body_rho = 2000

    def add_user_options(self, group):
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=200,
            help="Reynolds number."
        )
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.2,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--dx", action="store", type=float, dest="dx",
            default=0.5,
            help="Spacing resolution"
        )
        group.add_argument(
            "--lt", action="store", type=float, dest="Lt", default=50,
            help="Length of the WindTunnel."
        )
        group.add_argument(
            "--wt", action="store", type=float, dest="Wt", default=25,
            help="Half width of the WindTunnel."
        )
        group.add_argument(
            "--dc", action="store", type=float, dest="dc", default=2.0,
            help="Diameter of the cylinder."
        )
        group.add_argument(
            "--cfl-factor", action="store", type=float, dest="cfl_factor",
            default=0.25,
            help="CFL number, useful when using different Integrator."
        )

    def consume_user_options(self):
        if self.options.n_damp is None:
            self.options.n_damp = 20
        self.Lt = self.options.Lt
        self.Wt = self.options.Wt
        self.dc = self.options.dc
        self.dx = self.options.dx
        re = self.options.re

        self.nu = nu = umax * self.dc / re
        self.cxy = self.Lt / 2.5, 0.0

        self.hdx = hdx = self.options.hdx

        self.h = h = hdx * self.dx
        self.cfl = cfl = self.options.cfl_factor
        dt_cfl = cfl * h / (c0 + umax)
        dt_viscous = 0.125 * h**2 / nu

        self.dt = min(dt_cfl, dt_viscous)
        self.tf = 6.0

    def _create_solid(self):
        dx = self.dx
        h0 = self.hdx*dx

        r = np.arange(dx/2, self.dc/2, dx)
        x, y = np.array([]), np.array([])
        for i in r:
            spacing = dx
            theta = np.linspace(0, 2*pi, int(2*pi*i/spacing), endpoint=False)
            x = np.append(x,  i * cos(theta))
            y = np.append(y,  i * sin(theta))

        x += self.cxy[0]
        volume = dx*dx
        solid = get_particle_array(
            name='solid', x=x, y=y, m=volume*rho, rho=rho, h=h0
        )
        solid.add_constant('ds_min', dx)

        # add rigid body collision properties
        solid.add_property('rad_s')
        solid.rad_s[:] = self.dx / 2.
        return solid

    def _create_rigid_body(self):
        from pysph.base.utils import (
            get_particle_array_rigid_body)
        dx = self.dx
        h0 = self.hdx*dx

        r = np.arange(dx/2, self.dc/2, dx)
        x, y = np.array([]), np.array([])
        for i in r:
            spacing = dx
            theta = np.linspace(0, 2*pi, int(2*pi*i/spacing), endpoint=False)
            x = np.append(x,  i * cos(theta))
            y = np.append(y,  i * sin(theta))

        x += self.cxy[0]
        volume = dx*dx
        m = self.rigid_body_rho * volume
        b_id = 0

        body = get_particle_array_rigid_body(x=x, y=y, h=h0, m=m, rho=self.rigid_body_rho,
                                             rad_s=self.dx/2., V=volume, cs=0.,
                                             body_id=b_id, name="body")

        # properties for rigid fluid coupling
        for name in ['m_fsi', 'rho_fsi', 'p_fsi', 'wij', 'wij2']:
            body.add_property(name)
        body.m_fsi[:] = rho * volume
        body.rho_fsi[:] = rho
        body.p_fsi[:] = 0.

        return body

    def _create_box(self):
        dx = self.dx
        m = rho * dx * dx
        h0 = self.hdx*dx
        layers = self.nl * dx
        w = self.Wt
        l = self.Lt
        x, y = np.mgrid[-layers:l+layers:dx, -w-layers:w+layers:dx]

        # First form walls, then inlet and outlet, and finally fluid.
        wall_cond = (y > w - dx/2) | (y < -w + dx/2)
        xw, yw = x[wall_cond], y[wall_cond]
        x, y = x[~wall_cond], y[~wall_cond]
        wall = get_particle_array(
            name='wall', x=xw, y=yw, m=m, h=h0, rho=rho
        )

        # Used in the Non-reflection boundary conditions. See create_equations
        # below.
        props = [
            'xn', 'yn', 'zn', 'J2v', 'J3v', 'J2u', 'J3u', 'J1', 'wij2', 'disp',
            'ioid'
        ]
        for prop in props:
            wall.add_property(prop)
        consts = [
            'avg_j2u', 'avg_j3u', 'avg_j2v', 'avg_j3v', 'avg_j1', 'uref'
        ]
        for const in consts:
            wall.add_constant(const, 0.0)
        wall.yn[wall.y > 0.0] = 1.0
        wall.yn[wall.y <= 0.0] = -1.0
        # add rigid body collision properties
        wall.add_property('rad_s')
        wall.rad_s[:] = self.dx / 2.

        # Create Inlet.
        inlet_cond = (x < dx/2)
        xi, yi = x[inlet_cond], y[inlet_cond]
        x, y = x[~inlet_cond], y[~inlet_cond]
        inlet = get_particle_array(
            name='inlet', x=xi, y=yi, m=m, h=h0, u=u_freestream, rho=rho,
            p=0.0, uhat=u_freestream, xn=-1.0, yn=0.0
        )

        # Create Outlet.
        outlet_cond = (x > l - dx/2)
        xo, yo = x[outlet_cond], y[outlet_cond]
        # Use uhat=umax. So that the particles are moving out, if 0.0 is used
        # instead, the outlet particles will not move.
        outlet = get_particle_array(
            name='outlet', x=xo, y=yo, m=m, h=h0, u=u_freestream, rho=rho,
            p=0.0, uhat=u_freestream, vhat=0.0, xn=1.0, yn=0.0
        )

        # Create Fluid.
        xf, yf = x[~outlet_cond], y[~outlet_cond]
        fluid = get_particle_array(
            name='fluid', x=xf, y=yf, m=m, h=h0, u=u_freestream, rho=rho,
            p=0.0, vmag=0.0
        )
        setup_properties([fluid, inlet, outlet])
        return fluid, wall, inlet, outlet

    def create_particles(self):
        fluid, wall, inlet, outlet = self._create_box()
        solid = self._create_solid()
        body = self._create_rigid_body()

        particles = [fluid, inlet, outlet, wall, body]

        # Do not use clean=True here. The properties not used in EDAC equations
        # but used in the create_equations below will be erased.
        self.scheme.setup_properties(particles, clean=False)

        # Remove the fluid particles
        G.remove_overlap_particles(
            fluid, body, self.dx, dim=2
        )

        return particles

    def create_scheme(self):
        nu = None
        s = WCSPHRigidBodyScheme(
            ['fluid'], None, ['body'], dim=2, rho0=rho, c0=c0,
            nu=nu, h=None, inlet_outlet_manager=None,
            inviscid_solids=['wall'], cfl=None
        )
        return s

    def create_equations(self):
        from pysph.sph.equation import Group
        from edac_rigid_bodies import EvaluateNumberDensity
        from pysph.sph.bc.inlet_outlet_manager import (
            UpdateNormalsAndDisplacements
        )

        equations = self.scheme.get_equations()
        eq = []
        eq.append(
            Group(equations=[
                EvaluateCharacterisctics(
                    dest='fluid', sources=None, c_ref=c0, rho_ref=rho,
                    u_ref=u_freestream, v_ref=0.0, p_ref=0.0
                )
            ])
        )
        eq.append(
            Group(equations=[
                UpdateNormalsAndDisplacements(
                    'inlet', None, xn=-1, yn=0, zn=0, xo=0, yo=0, zo=0
                ),
                UpdateNormalsAndDisplacements(
                    'outlet', None, xn=1, yn=0, zn=0, xo=0, yo=0, zo=0
                ),
                EvaluateNumberDensity(dest='inlet', sources=['fluid']),
                ShepardInterpolateCharacteristics(dest='inlet', sources=['fluid']),
                EvaluateNumberDensity(dest='wall', sources=['fluid']),
                ShepardInterpolateCharacteristics(dest='wall', sources=['fluid']),
                EvaluateNumberDensity(dest='outlet', sources=['fluid']),
                ShepardInterpolateCharacteristics(dest='outlet', sources=['fluid']),
            ])
        )
        eq.append(Group(equations=[
            EvaluatePropertyfromCharacteristics(
                dest='wall', sources=None, c_ref=c0, rho_ref=rho,
                u_ref=u_freestream, v_ref=0.0, p_ref=0.0
            ),
            EvaluatePropertyfromCharacteristics(
                dest='inlet', sources=None, c_ref=c0, rho_ref=rho,
                u_ref=u_freestream, v_ref=0.0, p_ref=0.0
            ),
            EvaluatePropertyfromCharacteristics(
                dest='outlet', sources=None, c_ref=c0, rho_ref=rho,
                u_ref=u_freestream, v_ref=0.0, p_ref=0.0
            )])
        )
        # Remove "SolidWallPressureBC" for "wall" particle only, as it is set
        # "EvaluatePropertyfromCharacteristics" equation.
        # import pudb
        # pudb.set_trace()
        # equations[1].equations.pop()
        equations = eq + equations
        return equations

    def configure_scheme(self):
        scheme = self.scheme
        self.iom = self._create_inlet_outlet_manager()
        scheme.inlet_outlet_manager = self.iom
        pfreq = 100
        kernel = QuinticSpline(dim=2)
        self.iom.update_dx(self.dx)
        scheme.configure(h=self.h, nu=self.nu, cfl=self.cfl)

        scheme.configure_solver(
            kernel=kernel, tf=self.tf, dt=self.dt, pfreq=pfreq, n_damp=0,
            output_at_times=list(range(1, 7))
        )

    def _get_io_info(self):
        from pysph.sph.bc.hybrid.outlet import Outlet
        from hybrid_simple_inlet_outlet import Inlet, SimpleInletOutlet

        i_has_ghost = False
        o_has_ghost = False
        i_update_cls = Inlet
        o_update_cls = Outlet
        manager = SimpleInletOutlet

        props_to_copy = [
            'x0', 'y0', 'z0', 'uhat', 'vhat', 'what', 'x', 'y', 'z',
            'u', 'v', 'w', 'm', 'h', 'rho', 'p', 'ioid'
        ]
        props_to_copy += ['u0', 'v0', 'w0', 'p0']

        inlet_info = InletInfo(
            pa_name='inlet', normal=[-1.0, 0.0, 0.0],
            refpoint=[self.dx/2, 0.0, 0.0], equations=None,
            has_ghost=i_has_ghost, update_cls=i_update_cls,
            umax=u_freestream
        )

        outlet_info = OutletInfo(
            pa_name='outlet', normal=[1.0, 0.0, 0.0],
            refpoint=[self.Lt - self.dx/2, 0.0, 0.0], equations=None,
            has_ghost=o_has_ghost, update_cls=o_update_cls,
            props_to_copy=props_to_copy
        )

        return inlet_info, outlet_info, manager

    def _create_inlet_outlet_manager(self):
        inlet_info, outlet_info, manager = self._get_io_info()
        iom = manager(
            fluid_arrays=['fluid'], inletinfo=[inlet_info],
            outletinfo=[outlet_info]
        )
        return iom

    def create_inlet_outlet(self, particle_arrays):
        iom = self.iom
        io = iom.get_inlet_outlet(particle_arrays)
        return io

    def customize_output(self):
        self._mayavi_config('''
        if 'wake' in particle_arrays:
            particle_arrays['wake'].visible = False
        if 'ghost_inlet' in particle_arrays:
            particle_arrays['ghost_inlet'].visible = False
        for name in ['fluid', 'inlet', 'outlet']:
            b = particle_arrays[name]
            b.scalar = 'p'
            b.range = '-1000, 1000'
            b.plot.module_manager.scalar_lut_manager.lut_mode = 'seismic'
        for name in ['fluid', 'solid']:
            b = particle_arrays[name]
            b.point_size = 2.0
        ''')

    def post_step(self, solver):
        freq = 500
        if solver.count % freq == 0:
            self.nnps.update()
            for i, pa in enumerate(self.particles):
                if pa.name == 'fluid':
                    self.nnps.spatially_order_particles(i)
            self.nnps.update()


if __name__ == '__main__':
    app = WindTunnel()
    app.run()
    # app.post_process(app.info_filename)
