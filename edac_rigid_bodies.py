"""
EDAC SPH formulation
#####################

Equations for the Entropically Damped Artificial Compressibility SPH scheme.

Please note that this scheme is still under development and this module may
change at some point in the future.

References
----------

    .. [PRKP2017] Prabhu Ramachandran and Kunal Puri, Entropically damped
       artificial compressibility for SPH, under review, 2017.
       http://arxiv.org/pdf/1311.2167v2.pdf

"""

from math import sin, sqrt
from math import pi as M_PI
from compyle.api import declare
import numpy as np

from pysph.base.utils import get_particle_array
from pysph.base.utils import DEFAULT_PROPS
from pysph.sph.equation import Equation, Group
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme, add_bool_argument
from pysph.sph.wc.linalg import mat_vec_mult

# Rigid body imports
from pysph.sph.rigid_body import (
    BodyForce, SummationDensityBoundary, RigidBodyCollision, RigidBodyMoments,
    RigidBodyMotion, AkinciRigidFluidCoupling, RK2StepRigidBody)


EDAC_PROPS = (
    'ap', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'x0', 'y0', 'z0', 'u0',
    'v0', 'w0', 'p0', 'V', 'vmag'
)


def setup_properties(particle_arrays):
    for pa in particle_arrays:
        pa.add_property('htmp')
        pa.add_property('ds')
        pa.add_property('n_nbrs', type='int')
        pa.add_output_arrays(['n_nbrs', 'h'])
        pa.add_property('m_min')
        pa.add_property('m_max')
        pa.add_property('closest_idx', type='int')
        pa.add_property('split', type='int')
        pa.add_property('gradv', stride=9)
        pa.add_property('gradp', stride=3)
        pa.add_property('shift_x')
        pa.add_property('shift_y')
        pa.add_property('shift_z')




def get_particle_array_edac(constants=None, **props):
    "Get the fluid array for the transport velocity formulation"

    pa = get_particle_array(
        constants=constants, additional_props=EDAC_PROPS, **props
    )
    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p',
                          'au', 'av', 'aw', 'ap', 'm', 'h', 'vmag'])

    return pa


EDAC_SOLID_PROPS = ('ap', 'p0', 'wij', 'uf', 'vf', 'wf', 'ug', 'vg', 'wg',
                    'ax', 'ay', 'az', 'V')


def get_particle_array_edac_solid(constants=None, **props):
    "Get the fluid array for the transport velocity formulation"

    pa = get_particle_array(
        constants=constants, additional_props=EDAC_SOLID_PROPS, **props
    )
    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p',
                          'h'])

    return pa


class SummationDensity(Equation):
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_m, WIJ):
        d_rho[d_idx] += s_m[s_idx]*WIJ


class SummationDensityOnFluidDueToRigidBody(Equation):
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_m_fsi, WIJ):
        d_rho[d_idx] += s_m_fsi[s_idx]*WIJ


class SummationDensityOnRigidBodyDueToRigidBody(Equation):
    def initialize(self, d_idx, d_rho_fsi):
        d_rho_fsi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho_fsi, s_m_fsi, WIJ):
        d_rho_fsi[d_idx] += s_m_fsi[s_idx]*WIJ


class SummationDensityOnRigidBodyDueToFluid(Equation):
    def initialize(self, d_idx, d_rho_fsi):
        d_rho_fsi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho_fsi, s_m, WIJ):
        d_rho_fsi[d_idx] += s_m[s_idx]*WIJ


class EDACStep(IntegratorStep):
    """Standard Predictor Corrector integrator for the WCSPH formulation

    Use this integrator for WCSPH formulations. In the predictor step,
    the particles are advanced to `t + dt/2`. The particles are then
    advanced with the new force computed at this position.

    This integrator can be used in PEC or EPEC mode.

    The same integrator can be used for other problems. Like for
    example solid mechanics (see SolidMechStep)

    """
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p, d_vmag):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_p0[d_idx] = d_p[d_idx]

        d_vmag[d_idx] = sqrt(d_u[d_idx] * d_u[d_idx] + d_v[d_idx] *
                             d_v[d_idx] + d_w[d_idx] * d_w[d_idx])

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p, d_au, d_av,
               d_aw, d_ap, dt, d_vmag):
        dtb2 = 0.5*dt
        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_vmag[d_idx] = sqrt(d_u[d_idx] * d_u[d_idx] + d_v[d_idx] *
                             d_v[d_idx] + d_w[d_idx] * d_w[d_idx])

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dtb2 * d_ap[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p, d_au, d_av,
               d_aw, d_ap, d_vmag, dt):

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_vmag[d_idx] = sqrt(d_u[d_idx] * d_u[d_idx] + d_v[d_idx] *
                             d_v[d_idx] + d_w[d_idx] * d_w[d_idx])

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dt * d_ap[d_idx]


class SolidWallNoSlipBCReverse(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_auf, d_avf, d_awf):
        d_auf[d_idx] = 0.0
        d_avf[d_idx] = 0.0
        d_awf[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho,
             d_ug, d_vg, d_wg,
             d_auf, d_avf, d_awf,
             s_u, s_v, s_w,
             DWI, DWJ, R2IJ, EPS, XIJ):
        mj = s_m[s_idx]
        rhoij = (d_rho[d_idx] + s_rho[s_idx])
        dwij = declare('matrix(3)')
        dwij[0] = 0.5 * (DWJ[0] + DWI[0])
        dwij[1] = 0.5 * (DWJ[1] + DWI[1])
        dwij[2] = 0.5 * (DWJ[2] + DWI[2])
        Fij = dwij[0]*XIJ[0] + dwij[1]*XIJ[1] + dwij[2]*XIJ[2]

        tmp = 4 * mj * self.nu * Fij / (rhoij * (R2IJ + EPS))

        d_auf[d_idx] += tmp * (d_ug[d_idx] - s_u[s_idx])
        d_avf[d_idx] += tmp * (d_vg[d_idx] - s_v[s_idx])
        d_awf[d_idx] += tmp * (d_wg[d_idx] - s_w[s_idx])


class SolidWallNoSlipBC(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_rho, s_rho, d_u, d_v, d_w, d_au, d_av,
             d_aw, s_ug, s_vg, s_wg, DWI, DWJ, R2IJ, EPS, XIJ):
        mj = s_m[s_idx]
        rhoij = (d_rho[d_idx] + s_rho[s_idx])
        dwij = declare('matrix(3)')
        dwij[0] = 0.5 * (DWJ[0] + DWI[0])
        dwij[1] = 0.5 * (DWJ[1] + DWI[1])
        dwij[2] = 0.5 * (DWJ[2] + DWI[2])
        Fij = dwij[0]*XIJ[0] + dwij[1]*XIJ[1] + dwij[2]*XIJ[2]

        tmp = 4 * mj * self.nu * Fij / (rhoij * (R2IJ + EPS))

        d_au[d_idx] += tmp * (d_u[d_idx] - s_ug[s_idx])
        d_av[d_idx] += tmp * (d_v[d_idx] - s_vg[s_idx])
        d_aw[d_idx] += tmp * (d_w[d_idx] - s_wg[s_idx])


class SolidWallPressureBC(Equation):
    r"""Solid wall pressure boundary condition from Adami and Hu (transport
    velocity formulation).

    """
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx]*gdotxij*WIJ

    def post_loop(self, d_idx, d_wij, d_p):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]


class SolidWallPressureBCRigidBody(Equation):
    r"""Solid wall pressure boundary condition from Adami and Hu (transport
    velocity formulation).

    """
    def __init__(self, dest, sources, c0, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.c0 = c0

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_p_fsi):
        d_p_fsi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p_fsi, s_p, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        d_p_fsi[d_idx] += s_p[s_idx]*WIJ + s_rho[s_idx]*gdotxij*WIJ

    def post_loop(self, d_idx, d_wij, d_p_fsi, d_rho_fsi):
        # extrapolated pressure at the ghost particle
        if d_wij[d_idx] > 1e-14:
            d_p_fsi[d_idx] /= d_wij[d_idx]
            d_rho_fsi[d_idx] = d_p_fsi[d_idx] / self.c0**2. + 1


class ClampWallPressure(Equation):
    r"""Clamp the wall pressure to non-negative values.
    """
    def post_loop(self, d_idx, d_p):
        if d_p[d_idx] < 0.0:
            d_p[d_idx] = 0.0


class SourceNumberDensity(Equation):
    r"""Evaluates the number density due to the source particles"""
    def initialize(self, d_idx, d_wij):
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, d_wij, WIJ):
        d_wij[d_idx] += WIJ


class SetWallVelocity(Equation):
    r"""Extrapolating the fluid velocity on to the wall Eq. (22) in REF1:

    .. math::

        \tilde{\boldsymbol{v}}_a = \frac{\sum_b\boldsymbol{v}_b W_{ab}}
        {\sum_b W_{ab}}

    Notes:

    This should be used only after (or in the same group) as the
    SolidWallPressureBC equation.

    The destination particle array for this equation should define the
    *filtered* velocity variables :math:`uf, vf, wf`.

    """
    def initialize(self, d_idx, d_uf, d_vf, d_wf):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf,
             s_u, s_v, s_w, WIJ):
        # sum in Eq. (22)
        # this will be normalized in post loop
        d_uf[d_idx] += s_u[s_idx] * WIJ
        d_vf[d_idx] += s_v[s_idx] * WIJ
        d_wf[d_idx] += s_w[s_idx] * WIJ

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx,
                  d_ug, d_vg, d_wg, d_u, d_v, d_w):

        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface

        # Note that d_wij is already computed for the pressure BC.
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ug[d_idx] = 2*d_u[d_idx] - d_uf[d_idx]
        d_vg[d_idx] = 2*d_v[d_idx] - d_vf[d_idx]
        d_wg[d_idx] = 2*d_w[d_idx] - d_wf[d_idx]


class EvaluateNumberDensity(Equation):
    def initialize(self, d_idx, d_wij, d_wij2):
        d_wij[d_idx] = 0.0
        d_wij2[d_idx] = 0.0

    def loop(self, d_idx, d_wij, d_wij2, XIJ, HIJ, RIJ, SPH_KERNEL):
        wij = SPH_KERNEL.kernel(XIJ, RIJ, HIJ)
        wij2 = SPH_KERNEL.kernel(XIJ, RIJ, 0.5*HIJ)
        d_wij[d_idx] += wij
        d_wij2[d_idx] += wij2


class SlipVelocityExtrapolation(Equation):
    '''Slip boundary condition on the wall

    The velocity of the fluid is extrapolated over to the wall using
    shepard extrapolation. The velocity normal to the wall is reflected back
    to impose no penetration.
    '''
    def initialize(self, d_idx, d_ug_star, d_vg_star, d_wg_star):
        d_ug_star[d_idx] = 0.0
        d_vg_star[d_idx] = 0.0
        d_wg_star[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_ug_star, d_vg_star, d_wg_star, s_u,
             s_v, s_w, d_wij2, XIJ, RIJ, HIJ, SPH_KERNEL):
        wij = SPH_KERNEL.kernel(XIJ, RIJ, HIJ)
        wij2 = SPH_KERNEL.kernel(XIJ, RIJ, 0.5*HIJ)

        if d_wij2[d_idx] > 1e-14:
            d_ug_star[d_idx] += s_u[s_idx]*wij2
            d_vg_star[d_idx] += s_v[s_idx]*wij2
            d_wg_star[d_idx] += s_w[s_idx]*wij2
        else:
            d_ug_star[d_idx] += s_u[s_idx]*wij
            d_vg_star[d_idx] += s_v[s_idx]*wij
            d_wg_star[d_idx] += s_w[s_idx]*wij

    def post_loop(self, d_idx, d_wij, d_wij2, d_ug_star, d_vg_star,
                  d_wg_star, d_normal, d_u, d_v, d_w):
        idx = declare('int')
        idx = 3*d_idx
        if d_wij2[d_idx] > 1e-14:
            d_ug_star[d_idx] /= d_wij2[d_idx]
            d_vg_star[d_idx] /= d_wij2[d_idx]
            d_wg_star[d_idx] /= d_wij2[d_idx]
        elif d_wij[d_idx] > 1e-14:
            d_ug_star[d_idx] /= d_wij[d_idx]
            d_vg_star[d_idx] /= d_wij[d_idx]
            d_wg_star[d_idx] /= d_wij[d_idx]

        # u_g \cdot n = 2*(u_wall \cdot n ) - (u_f \cdot n)
        # u_g \cdot t = (u_f \cdot t) = u_f - (u_f \cdot n)
        tmp1 = d_u[d_idx] - d_ug_star[d_idx]
        tmp2 = d_v[d_idx] - d_vg_star[d_idx]
        tmp3 = d_w[d_idx] - d_wg_star[d_idx]

        projection = (tmp1*d_normal[idx] +
                      tmp2*d_normal[idx+1] +
                      tmp3*d_normal[idx+2])

        d_ug_star[d_idx] += 2*projection * d_normal[idx]
        d_vg_star[d_idx] += 2*projection * d_normal[idx+1]
        d_wg_star[d_idx] += 2*projection * d_normal[idx+2]


class NoSlipVelocityExtrapolation(Equation):
    '''No Slip boundary condition on the wall

    The velocity of the fluid is extrapolated over to the wall using
    shepard extrapolation. The velocity normal to the wall is reflected back
    to impose no penetration.
    '''
    def initialize(self, d_idx, d_u, d_v, d_w):
        d_u[d_idx] = 0.0
        d_v[d_idx] = 0.0
        d_w[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_u, d_v, d_w, s_u, s_v, s_w, WIJ):
        d_u[d_idx] += s_u[s_idx]*WIJ
        d_v[d_idx] += s_v[s_idx]*WIJ
        d_w[d_idx] += s_w[s_idx]*WIJ

    def post_loop(self, d_idx, d_wij, d_u, d_v, d_w, d_xn, d_yn, d_zn):
        if d_wij[d_idx] > 1e-14:
            d_u[d_idx] /= d_wij[d_idx]
            d_v[d_idx] /= d_wij[d_idx]
            d_w[d_idx] /= d_wij[d_idx]

        projection = d_u[d_idx]*d_xn[d_idx] +\
            d_v[d_idx]*d_yn[d_idx] + d_w[d_idx]*d_zn[d_idx]

        d_u[d_idx] = d_u[d_idx] - 2 * projection * d_xn[d_idx]
        d_v[d_idx] = d_v[d_idx] - 2 * projection * d_yn[d_idx]
        d_w[d_idx] = d_w[d_idx] - 2 * projection * d_zn[d_idx]


class NoSlipAdvVelocityExtrapolation(Equation):
    '''No Slip boundary condition on the wall

    The advection velocity of the fluid is extrapolated over to the wall
    using shepard extrapolation. The advection velocity normal to the wall
    is reflected back to impose no penetration.
    '''
    def initialize(self, d_idx, d_uhat, d_vhat, d_what):
        d_uhat[d_idx] = 0.0
        d_vhat[d_idx] = 0.0
        d_what[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uhat, d_vhat, d_what, s_uhat, s_vhat,
             s_what, WIJ):
        d_uhat[d_idx] += s_uhat[s_idx]*WIJ
        d_vhat[d_idx] += s_vhat[s_idx]*WIJ
        d_what[d_idx] += s_what[s_idx]*WIJ

    def post_loop(self, d_idx, d_wij, d_uhat, d_vhat, d_what, d_xn, d_yn,
                  d_zn):
        if d_wij[d_idx] > 1e-14:
            d_uhat[d_idx] /= d_wij[d_idx]
            d_vhat[d_idx] /= d_wij[d_idx]
            d_what[d_idx] /= d_wij[d_idx]

        projection = d_uhat[d_idx]*d_xn[d_idx] +\
            d_vhat[d_idx]*d_yn[d_idx] + d_what[d_idx]*d_zn[d_idx]

        d_uhat[d_idx] = d_uhat[d_idx] - 2 * projection * d_xn[d_idx]
        d_vhat[d_idx] = d_vhat[d_idx] - 2 * projection * d_yn[d_idx]
        d_what[d_idx] = d_what[d_idx] - 2 * projection * d_zn[d_idx]


class MomentumEquationArtificialViscosity(Equation):
    r"""**Artificial viscosity for the momentum equation**

    Eq. (11) in [Adami2012]:

    .. math::

        \frac{d \boldsymbol{v}_a}{dt} = -\sum_b m_b \alpha h_{ab}
        c_{ab} \frac{\boldsymbol{v}_{ab}\cdot
        \boldsymbol{r}_{ab}}{\rho_{ab}\left(|r_{ab}|^2 + \epsilon
        \right)}\nabla_a W_{ab}

    where

    .. math::

        \rho_{ab} = \frac{\rho_a + \rho_b}{2}\\

        c_{ab} = \frac{c_a + c_b}{2}\\

        h_{ab} = \frac{h_a + h_b}{2}
    """
    def __init__(self, dest, sources, c0, alpha=0.1):
        r"""
        Parameters
        ----------
        alpha : float
            constant
        c0 : float
            speed of sound
        """

        self.alpha = alpha
        self.c0 = c0
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_au, d_av, d_aw,
             RHOIJ1, R2IJ, EPS, DWI, DWJ, VIJ, XIJ, HIJ):

        # v_{ab} \cdot r_{ab}
        vijdotrij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        # scalar part of the accelerations Eq. (11)
        piij = 0.0
        if vijdotrij < 0:
            muij = (HIJ * vijdotrij)/(R2IJ + EPS)

            piij = -self.alpha*self.c0*muij
            piij = s_m[s_idx] * piij*RHOIJ1

        d_au[d_idx] += -piij * 0.5 * (DWJ[0] + DWI[0])
        d_av[d_idx] += -piij * 0.5 * (DWJ[1] + DWI[1])
        d_aw[d_idx] += -piij * 0.5 * (DWJ[2] + DWI[2])


class EDACTVFStep(IntegratorStep):

    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p, d_vmag,
                   d_rho, d_rho0):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_p0[d_idx] = d_p[d_idx]
        d_rho0[d_idx] = d_rho[d_idx]

        d_vmag[d_idx] = sqrt(d_u[d_idx] * d_u[d_idx] + d_v[d_idx] *
                             d_v[d_idx] + d_w[d_idx] * d_w[d_idx])

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p, d_au,
               d_av, d_auhat, d_avhat, d_awhat, d_uhat, d_vhat, d_what,
               d_aw, d_ap, dt, d_vmag):
        dtb2 = 0.5*dt
        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2*d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2*d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2*d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_uhat[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_vhat[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_what[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dtb2 * d_ap[d_idx]

        d_vmag[d_idx] = sqrt(d_u[d_idx] * d_u[d_idx] + d_v[d_idx] *
                             d_v[d_idx] + d_w[d_idx] * d_w[d_idx])

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_u0, d_v0, d_w0, d_u, d_v, d_w, d_p0, d_p, d_au, d_av,
               d_aw, d_auhat, d_avhat, d_awhat, d_uhat, d_vhat, d_what,
               d_ap, d_vmag, dt):
        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_vmag[d_idx] = sqrt(d_u[d_idx] * d_u[d_idx] + d_v[d_idx] *
                             d_v[d_idx] + d_w[d_idx] * d_w[d_idx])

        d_uhat[d_idx] = d_u[d_idx] + dt*d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dt*d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dt*d_awhat[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_uhat[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_vhat[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_what[d_idx]

        d_p[d_idx] = d_p0[d_idx] + dt * d_ap[d_idx]


def update_rho(dests, sources, method):
    equations = []
    if method == 'scatter':
        for dest in dests:
            equations.append(Group(
                equations=[SummationDensityScatter(dest, sources)]
            ))
    elif method == 'gather':
        for dest in dests:
            equations.append(Group(
                equations=[SummationDensityGatherCorrected(dest, sources)]
            ))
    elif method == 'standard':
        for dest in dests:
            equations.append(Group(
                equations=[SummationDensity(dest, sources)]
            ))
    return equations


class InitializeUhat(Equation):
    def initialize(self, d_idx, d_dummy_uhat, d_dummy_vhat, d_dummy_what,
                   d_dummy_x, d_dummy_y, d_dummy_z):
        d_dummy_uhat[d_idx] = 0.0
        d_dummy_vhat[d_idx] = 0.0
        d_dummy_what[d_idx] = 0.0

        d_dummy_x[d_idx] = 0.0
        d_dummy_y[d_idx] = 0.0
        d_dummy_z[d_idx] = 0.0


class ShiftUpdatePosition(Equation):
    def __init__(self, dest, sources, dt_fac):
        self.dt_fac = dt_fac
        super().__init__(dest, sources)

    def post_loop(self, d_idx, d_x, d_y, d_z, d_auhat, d_avhat, d_awhat,
                  d_dummy_uhat, d_dummy_vhat, d_dummy_what, dt,
                  d_dummy_x, d_dummy_y, d_dummy_z):
        dt = dt/self.dt_fac
        dt2b2 = 0.5*dt*dt

        d_x[d_idx] += dt*d_dummy_uhat[d_idx] + dt2b2*d_auhat[d_idx]
        d_y[d_idx] += dt*d_dummy_vhat[d_idx] + dt2b2*d_avhat[d_idx]
        d_z[d_idx] += dt*d_dummy_what[d_idx] + dt2b2*d_awhat[d_idx]

        d_dummy_x[d_idx] += dt*d_dummy_uhat[d_idx] + dt2b2*d_auhat[d_idx]
        d_dummy_y[d_idx] += dt*d_dummy_vhat[d_idx] + dt2b2*d_avhat[d_idx]
        d_dummy_z[d_idx] += dt*d_dummy_what[d_idx] + dt2b2*d_awhat[d_idx]

        d_dummy_uhat[d_idx] += dt*d_auhat[d_idx]
        d_dummy_vhat[d_idx] += dt*d_avhat[d_idx]
        d_dummy_what[d_idx] += dt*d_awhat[d_idx]


class ShiftPositionLind(Equation):
    def __init__(self, dest, sources, dim):
        self.dim = dim
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_auhat, d_avhat, d_awhat, s_rho, d_h,
             HIJ, DWIJ, WIJ, SPH_KERNEL):
        mj = s_m[s_idx]
        rhoj = s_rho[s_idx]
        # Don't update this rho i.e., don't correct this rho by using summation
        # density inside the shifting loop.
        Vj = mj/rhoj
        hi = d_h[d_idx]

        dp = SPH_KERNEL.get_deltap() * HIJ
        WDX = SPH_KERNEL.kernel([0.0, 0.0, 0.0], dp, HIJ)
        fac = 0.5 * hi * hi
        tmp = fac * Vj * (1 + 0.24*(WIJ/WDX)**4)
        d_auhat[d_idx] -= tmp * DWIJ[0]
        d_avhat[d_idx] -= tmp * DWIJ[1]
        d_awhat[d_idx] -= tmp * DWIJ[2]

    def post_loop(self, d_idx, d_auhat, d_avhat, d_awhat, d_x, d_y, d_z):
        d_x[d_idx] += d_auhat[d_idx]
        d_y[d_idx] += d_avhat[d_idx]
        d_z[d_idx] += d_awhat[d_idx]


class ShiftForceLind(Equation):
    def __init__(self, dest, sources, cs, cfl, rho0):
        self.cs = cs
        self.cfl = cfl
        self.rho0 = rho0
        self.fac = 0.5 * (self.cs/self.cfl)**2
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_auhat, d_avhat, d_awhat,
             DWIJ, WIJ, HIJ, SPH_KERNEL):
        mj = s_m[s_idx]
        # Don't update this rho i.e., don't correct this rho by using summation
        # density inside the shifting loop.
        Vj = mj/self.rho0

        dp = SPH_KERNEL.get_deltap() * HIJ
        WDX = SPH_KERNEL.kernel([0.0, 0.0, 0.0], dp, HIJ)
        tmp = self.fac * Vj * (1 + 0.24*(WIJ/WDX)**4)
        d_auhat[d_idx] -= tmp * DWIJ[0]
        d_avhat[d_idx] -= tmp * DWIJ[1]
        d_awhat[d_idx] -= tmp * DWIJ[2]

    def post_loop(self, d_idx, d_auhat, d_avhat, d_awhat, dt, d_h):
        shift_disp = dt * dt * sqrt(d_auhat[d_idx] * d_auhat[d_idx] +
                                    d_avhat[d_idx] * d_avhat[d_idx] +
                                    d_awhat[d_idx] * d_awhat[d_idx])
        max_disp = 0.025 * d_h[d_idx]
        disp_allowed = 0.25 * max_disp

        if shift_disp > disp_allowed:
            fac = disp_allowed / shift_disp
        else:
            fac = 1.0

        d_auhat[d_idx] *= fac
        d_avhat[d_idx] *= fac
        d_awhat[d_idx] *= fac


def shift_force(fluids, sources, cs, cfl, rho0, iters=1):
    eqns = []
    for fluid in fluids:
        eqns.append(Group(
            equations=[InitializeUhat(dest=fluid, sources=None)]
        ))
    shift = []
    for fluid in fluids:
        shift.append(
            ShiftForceLind(dest=fluid, sources=sources, cs=cs, cfl=cfl,
                            rho0=rho0)
        )
        shift.append(
            ShiftUpdatePosition(dest=fluid, sources=None, dt_fac=float(iters))
        )
    eqns.append(Group(shift, iterate=True, min_iterations=iters, max_iterations=iters))
    return eqns


def shift_positions(fluids, sources, dim, iters=0):
    shift = []

    iterate = False
    if iters > 0:
        iterate = True

    for fluid in fluids:
        shift.append(Group(equations=[
            ShiftPositionLind(
                dest=fluid, sources=sources, dim=dim
            ),
        ], iterate=iterate, min_iterations=iters, max_iterations=iters))
    return shift



#######################################################################
# Equations using m/rho formulation
#######################################################################
class MomentumEquationPressureGradient(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0, tdamp=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.tdamp = tdamp
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_p, d_au, d_av, d_aw,
             s_m, s_rho, s_p, DWI, DWJ):
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        pi = d_p[d_idx]
        pj = s_p[s_idx]

        pi = pi / (rhoi * rhoi)
        pj = pj / (rhoj * rhoj)

        mj = s_m[s_idx]

        d_au[d_idx] += -mj * (pi * DWI[0] + pj * DWJ[0])
        d_av[d_idx] += -mj * (pi * DWI[1] + pj * DWJ[1])
        d_aw[d_idx] += -mj * (pi * DWI[2] + pj * DWJ[2])


class MomentumEquationViscosity(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_au, d_av, d_aw,
             R2IJ, EPS, DWI, DWJ, VIJ, XIJ):
        dwij = declare('matrix(3)')
        dwij[0] = 0.5 * (DWJ[0] + DWI[0])
        dwij[1] = 0.5 * (DWJ[1] + DWI[1])
        dwij[2] = 0.5 * (DWJ[2] + DWI[2])
        Fij = dwij[0]*XIJ[0] + dwij[1]*XIJ[1] + dwij[2]*XIJ[2]

        Vj = s_m[s_idx] / (d_rho[d_idx] + s_rho[s_idx])
        tmp = Vj * 4 * self.nu * Fij/(R2IJ + EPS)

        d_au[d_idx] += tmp * VIJ[0]
        d_av[d_idx] += tmp * VIJ[1]
        d_aw[d_idx] += tmp * VIJ[2]


class EDACEquation(Equation):
    def __init__(self, dest, sources, cs, nu, rho0, edac_alpha):
        self.cs = cs
        self.nu = nu
        self.rho0 = rho0
        self.edac_alpha = edac_alpha

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_ap, d_div, d_div_hat):
        d_ap[d_idx] = 0.0
        d_div[d_idx] = 0.0
        d_div_hat[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_ap, d_p, s_idx, s_m, s_rho, s_p, d_div,
             d_u, d_v, d_w, d_uhat, d_vhat, d_h, s_h, d_what,
             DWJ, DWI, VIJ, XIJ, R2IJ, EPS):
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        cs2 = self.cs * self.cs
        mj = s_m[s_idx]
        Vj = s_m[s_idx]/s_rho[s_idx]

        ki = self.edac_alpha * d_h[d_idx] * self.cs/8
        kj = self.edac_alpha * s_h[s_idx] * self.cs/8
        kij = 4 * (ki * kj)/(ki + kj)

        # Since we are using gather formulation, DWI is used instead of DWIJ.
        # divergence of velocity term.
        vijdotdwij = DWI[0]*VIJ[0] + DWI[1]*VIJ[1] + DWI[2]*VIJ[2]

        d_ap[d_idx] += self.rho0 * Vj * cs2 * vijdotdwij
        d_div[d_idx] += Vj * -vijdotdwij

        # Viscous damping of pressure.
        dwij = declare('matrix(3)')
        dwij[0] = 0.5 * (DWI[0] + DWJ[0])
        dwij[1] = 0.5 * (DWI[1] + DWJ[1])
        dwij[2] = 0.5 * (DWI[2] + DWJ[2])

        xijdotdwij = dwij[0]*XIJ[0] + dwij[1]*XIJ[1] + dwij[2]*XIJ[2]

        tmp = Vj * kij * xijdotdwij/(R2IJ + EPS)
        # Diffusion of pressure term.
        d_ap[d_idx] += tmp*(d_p[d_idx] - s_p[s_idx])

        # Apply corrections due to shifting.
        udiff = declare('matrix(3)')
        udiff[0] = d_uhat[d_idx] - d_u[d_idx]
        udiff[1] = d_vhat[d_idx] - d_v[d_idx]
        udiff[2] = d_what[d_idx] - d_w[d_idx]

        gradp = declare('matrix(3)')
        pi = d_p[d_idx]
        pj = s_p[s_idx]

        pi = (pi) / (rhoi * rhoi)
        pj = (pj) / (rhoj * rhoj)

        gradp[0] = mj * (pi * DWI[0] + pj * DWJ[0])
        gradp[1] = mj * (pi * DWI[1] + pj * DWJ[1])
        gradp[2] = mj * (pi * DWI[2] + pj * DWJ[2])

        d_ap[d_idx] += udiff[0]*gradp[0] + udiff[1]*gradp[1] + udiff[2]*gradp[2]


class EDACEquationSolids(Equation):
    def __init__(self, dest, sources, cs, nu, rho0, edac_alpha):
        self.cs = cs
        self.nu = nu
        self.rho0 = rho0
        self.edac_alpha = edac_alpha
        super().__init__(dest, sources)

    def loop(self, d_idx, d_m, d_rho, d_ap, d_p, s_idx, s_m, s_rho, s_p,
             d_div, d_beta, d_u, d_v, d_w, d_uhat, d_vhat, d_h, s_h,
             d_what, d_pavg, s_ug, s_vg, s_wg, DWJ, DWI, XIJ, R2IJ, EPS,):
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        Vj = s_m[s_idx]/s_rho[s_idx]
        betai = d_beta[d_idx]
        cs2 = self.cs * self.cs

        ki = self.edac_alpha * d_h[d_idx] * self.cs/8
        kj = self.edac_alpha * s_h[s_idx] * self.cs/8
        kij = 4 * (ki * kj)/(ki + kj)

        # This is the same as continuity acceleration times cs^2
        # Since we are using gather formulation, DWI is used instead of DWIJ.
        vij = declare('matrix(3)')
        vij[0] = d_u[d_idx] - s_ug[s_idx]
        vij[1] = d_v[d_idx] - s_vg[s_idx]
        vij[2] = d_w[d_idx] - s_wg[s_idx]
        vijdotdwij = DWI[0]*vij[0] + DWI[1]*vij[1] + DWI[2]*vij[2]

        d_ap[d_idx] += self.rho0 * Vj * cs2 * vijdotdwij/betai
        d_div[d_idx] += Vj * -vijdotdwij

        # Viscous damping of pressure.
        xijdotdwij = DWI[0]*XIJ[0] + DWI[1]*XIJ[1] + DWI[2]*XIJ[2]

        tmp = Vj * kij * xijdotdwij/(R2IJ + EPS)
        d_ap[d_idx] += tmp*(d_p[d_idx] - s_p[s_idx])

        # Apply corrections due to shifting.
        udiff = declare('matrix(3)')
        udiff[0] = d_uhat[d_idx] - d_u[d_idx]
        udiff[1] = d_vhat[d_idx] - d_v[d_idx]
        udiff[2] = d_what[d_idx] - d_w[d_idx]

        gradp = declare('matrix(3)')
        pavg = d_pavg[d_idx]
        pi = d_p[d_idx]
        pj = s_p[s_idx]

        pi = (pi - pavg) / (rhoi * rhoi)
        pj = (pj - pavg) / (rhoj * rhoj)

        mj = s_m[s_idx]

        gradp[0] = mj * (pi * DWI[0] + pj * DWJ[0])
        gradp[1] = mj * (pi * DWI[1] + pj * DWJ[1])
        gradp[2] = mj * (pi * DWI[2] + pj * DWJ[2])

        d_ap[d_idx] += udiff[0]*gradp[0] + udiff[1]*gradp[1] + udiff[2]*gradp[2]


class EDACEquationRigidBodies(Equation):
    def __init__(self, dest, sources, cs, nu, rho0, edac_alpha):
        self.cs = cs
        self.nu = nu
        self.rho0 = rho0
        self.edac_alpha = edac_alpha

        super().__init__(dest, sources)

    def initialize(self, d_idx, d_ap, d_div, d_div_hat):
        d_ap[d_idx] = 0.0
        d_div[d_idx] = 0.0
        d_div_hat[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_ap, d_p, s_idx, s_m_fsi, s_rho_fsi, s_p_fsi, d_div,
             d_u, d_v, d_w, d_uhat, d_vhat, d_h, s_h, d_what,
             DWJ, DWI, VIJ, XIJ, R2IJ, EPS):
        rhoi = d_rho[d_idx]
        rhoj = s_rho_fsi[s_idx]
        cs2 = self.cs * self.cs
        mj = s_m_fsi[s_idx]
        Vj = s_m_fsi[s_idx]/s_rho_fsi[s_idx]

        ki = self.edac_alpha * d_h[d_idx] * self.cs/8
        kj = self.edac_alpha * s_h[s_idx] * self.cs/8
        kij = 4 * (ki * kj)/(ki + kj)

        # Since we are using gather formulation, DWI is used instead of DWIJ.
        # divergence of velocity term.
        vijdotdwij = DWI[0]*VIJ[0] + DWI[1]*VIJ[1] + DWI[2]*VIJ[2]

        d_ap[d_idx] += self.rho0 * Vj * cs2 * vijdotdwij
        d_div[d_idx] += Vj * -vijdotdwij

        # Viscous damping of pressure.
        dwij = declare('matrix(3)')
        dwij[0] = 0.5 * (DWI[0] + DWJ[0])
        dwij[1] = 0.5 * (DWI[1] + DWJ[1])
        dwij[2] = 0.5 * (DWI[2] + DWJ[2])

        xijdotdwij = dwij[0]*XIJ[0] + dwij[1]*XIJ[1] + dwij[2]*XIJ[2]

        tmp = Vj * kij * xijdotdwij/(R2IJ + EPS)
        # Diffusion of pressure term.
        d_ap[d_idx] += tmp*(d_p[d_idx] - s_p_fsi[s_idx])

        # Apply corrections due to shifting.
        udiff = declare('matrix(3)')
        udiff[0] = d_uhat[d_idx] - d_u[d_idx]
        udiff[1] = d_vhat[d_idx] - d_v[d_idx]
        udiff[2] = d_what[d_idx] - d_w[d_idx]

        gradp = declare('matrix(3)')
        pi = d_p[d_idx]
        pj = s_p_fsi[s_idx]

        pi = (pi) / (rhoi * rhoi)
        pj = (pj) / (rhoj * rhoj)

        gradp[0] = mj * (pi * DWI[0] + pj * DWJ[0])
        gradp[1] = mj * (pi * DWI[1] + pj * DWJ[1])
        gradp[2] = mj * (pi * DWI[2] + pj * DWJ[2])

        d_ap[d_idx] += udiff[0]*gradp[0] + udiff[1]*gradp[1] + udiff[2]*gradp[2]

#######################################################################
#######################################################################


class UpdateGhostProps(Equation):
    def initialize(self, d_idx, d_tag, d_gid, d_rho, d_p):
        idx = declare('int')
        if d_tag[d_idx] == 2:
            idx = d_gid[d_idx]
            d_rho[d_idx] = d_rho[idx]
            d_p[d_idx] = d_p[idx]


class ForceOnFluidDueToRigidBody(Equation):
    def loop(self, d_idx, d_m, d_rho, d_au, d_av, d_aw,  d_p,
             s_idx, s_V, DWIJ, s_m_fsi):

        psi = s_m_fsi[s_idx]

        _t1 = 2 * d_p[d_idx] / (d_rho[d_idx]**2)
        if d_p[d_idx] < 0.:
            _t1 = 0.

        d_au[d_idx] -= psi * _t1 * DWIJ[0]
        d_av[d_idx] -= psi * _t1 * DWIJ[1]
        d_aw[d_idx] -= psi * _t1 * DWIJ[2]


class ForceOnRigidBodyDueToFluid(Equation):
    def loop(self, d_idx, d_rho,
             s_idx, d_fx, d_fy, d_fz, DWIJ, d_m, d_m_fsi,
             s_m, s_p, s_rho):

        psi = s_m[s_idx]

        _t1 = 2 * s_p[s_idx] / (s_rho[s_idx]**2)
        if s_p[s_idx] < 0.:
            _t1 = 0.

        scale = d_m_fsi[d_idx] / d_m[d_idx]

        d_fx[d_idx] -= scale * psi * _t1 * DWIJ[0]
        d_fy[d_idx] -= scale * psi * _t1 * DWIJ[1]
        d_fz[d_idx] -= scale * psi * _t1 * DWIJ[2]


class WCSPHRigidBodyScheme(Scheme):
    def __init__(self, fluids, solids, rigid_bodies, dim, c0, nu, rho0, cfl,
                 gx=0.0, gy=0.0, gz=0.0, eps=0.0, h=0.0,
                 edac_alpha=1.5, alpha=0.0, clamp_p=False,
                 inlet_outlet_manager=None, inviscid_solids=None,
                 hdx=1.0, domain=None, has_ghosts=False):
        """The EDAC scheme.

        Parameters
        ----------

        fluids : list(str)
            List of names of fluid particle arrays.
        solids : list(str)
            List of names of solid particle arrays.
        rigid_bodies : list(str)
            List of names of rigid_bodies particle arrays.
        dim: int
            Dimensionality of the problem.
        c0 : float
            Speed of sound.
        nu : float
            Kinematic viscosity.
        rho0 : float
            Density of fluid.
        gx, gy, gz : float
            Componenents of body acceleration (gravity, external forcing etc.)
        tdamp: float
            Time for which the acceleration should be damped.
        eps : float
            XSPH smoothing factor, defaults to zero.
        h : float
            Parameter h used for the particles -- used to calculate viscosity.
        edac_alpha : float
            Factor to use for viscosity.
        alpha : float
            Factor to use for artificial viscosity.
        bql : bool
            Use the Basa Quinlan Lastiwka correction.
        clamp_p : bool
            Clamp the boundary pressure to positive values.  This is only used
            for external flows.
        inlet_outlet_manager : InletOutletManager Instance
            Pass the manager if inlet outlet boundaries are present
        inviscid_solids : list
            list of inviscid solid array names
        """
        self.c0 = c0
        self.cfl = cfl
        self.nu = nu
        self.rho0 = rho0
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.dim = dim
        self.eps = eps
        self.fluids = fluids
        if solids is None:
            self.solids = []
        else:
            self.solids = solids

        if rigid_bodies is None:
            self.rigid_bodies = []
        else:
            self.rigid_bodies = rigid_bodies

        self.solver = None
        self.clamp_p = clamp_p
        self.edac_alpha = edac_alpha
        self.alpha = alpha
        self.h = h
        self.inlet_outlet_manager = inlet_outlet_manager
        self.inviscid_solids = [] if inviscid_solids is None else\
            inviscid_solids
        self.attributes_changed()
        self.hdx = hdx
        self.domain = domain
        self.has_ghosts = has_ghosts

    # Public protocol ###################################################
    def add_user_options(self, group):
        group.add_argument(
            "--alpha", action="store", type=float, dest="alpha",
            default=None,
            help="Alpha for the artificial viscosity."
        )
        group.add_argument(
            "--edac-alpha", action="store", type=float, dest="edac_alpha",
            default=None,
            help="Alpha for the EDAC scheme viscosity."
        )
        add_bool_argument(
            group, 'clamp-pressure', dest='clamp_p',
            help="Clamp pressure on boundaries to be non-negative.",
            default=None
        )

    def consume_user_options(self, options):
        var = ['alpha', 'edac_alpha', 'clamp_p']
        data = dict((v, self._smart_getattr(options, v))
                    for v in var)
        self.configure(**data)

    def attributes_changed(self):
        self.use_tvf = True
        if self.h is not None and self.c0 is not None:
            self.art_nu = self.edac_alpha*self.h*self.c0/8

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        """Configure the solver to be generated.

        This is to be called before `get_solver` is called.

        Parameters
        ----------

        dim : int
            Number of dimensions.
        kernel : Kernel instance.
            Kernel to use, if none is passed a default one is used.
        integrator_cls : pysph.sph.integrator.Integrator
            Integrator class to use, use sensible default if none is
            passed.
        extra_steppers : dict
            Additional integration stepper instances as a dict.
        **kw : extra arguments
            Any additional keyword args are passed to the solver instance.
        """
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.integrator import PECIntegrator, EPECIntegrator

        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = EDACTVFStep
        default_int_cls = EPECIntegrator

        cls = integrator_cls if integrator_cls is not None else default_int_cls

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        iom = self.inlet_outlet_manager
        if iom is not None:
            iom_stepper = iom.get_stepper(self, cls, self.use_tvf)
            for name in iom_stepper:
                steppers[name] = iom_stepper[name]

        step_cls = RK2StepRigidBody
        for body in self.rigid_bodies:
            if body not in steppers:
                steppers[body] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

        if iom is not None:
            iom.setup_iom(dim=self.dim, kernel=kernel)

    def get_equations(self):
        return self._get_internal_flow_equations()

    def get_solver(self):
        return self.solver

    def setup_properties(self, particles, clean=True):
        """Setup the particle arrays so they have the right set of properties
        for this scheme.

        Parameters
        ----------

        particles : list
            List of particle arrays.

        clean : bool
            If True, removes any unnecessary properties.
        """
        particle_arrays = dict([(p.name, p) for p in particles])
        TVF_FLUID_PROPS = set([
            'uhat', 'vhat', 'what', 'ap',
            'auhat', 'avhat', 'awhat', 'V',
            'p0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
            'pavg', 'nnbr'
        ])
        extra_props = TVF_FLUID_PROPS if self.use_tvf else EDAC_PROPS

        all_fluid_props = DEFAULT_PROPS.union(extra_props)
        iom = self.inlet_outlet_manager
        fluids_with_io = self.fluids
        if iom is not None:
            io_particles = iom.get_io_names(ghost=True)
            fluids_with_io = self.fluids + io_particles
        for fluid in fluids_with_io:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, all_fluid_props, clean)
            pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p',
                                  'm', 'h', 'V'])
            if 'pavg' in pa.properties:
                pa.add_output_arrays(['pavg'])
            if iom is not None:
                iom.add_io_properties(pa, self)

        TVF_SOLID_PROPS = ['V', 'wij', 'ax', 'ay', 'az', 'uf', 'vf', 'wf',
                           'ug', 'vg', 'wg', 'ug_star', 'vg_star', 'wg_star']
        if self.inviscid_solids:
            TVF_SOLID_PROPS += ['xn', 'yn', 'zn', 'uhat', 'vhat', 'what']
        extra_props = TVF_SOLID_PROPS if self.use_tvf else EDAC_SOLID_PROPS
        all_solid_props = DEFAULT_PROPS.union(extra_props)
        for solid in self.solids + self.inviscid_solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, all_solid_props, clean)
            pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p',
                                  'm', 'h', 'V'])

        for fluid in fluids_with_io:
            pa = particle_arrays[fluid]
            pa.add_property('closest_idx', type='int')
            pa.add_property('split', type='int')
            pa.add_property('n_nbrs', type='int')

        props = [
            'htmp', 'uhat', 'vhat', 'what', 'vmag', 'm_min', 'm_max',
            'beta', 'div', 'div_hat', 'wij', 'm_nbr_max', 'arho',
            'rho0', 'vor', 'shift_x', 'shift_y', 'shift_z'
        ]
        for fluid in fluids_with_io:
            pa = particle_arrays[fluid]
            for prop in props:
                pa.add_property(prop)
            pa.add_output_arrays([
                'm_min', 'm_max', 'closest_idx', 'split', 'n_nbrs', 'h', 'vmag',
                'beta', 'wij', 'vor', 'div', 'shift_x', 'shift_y', 'auhat',
                'avhat',
            ])
            if 'vmax' not in pa.constants:
                pa.add_constant('vmax', [0.0])
            if 'iters' not in pa.constants:
                pa.add_constant('iters', [0.0])
            pa.add_property('gradv', stride=9)
            pa.add_property('gradp', stride=3)

        for solid in self.solids + self.inviscid_solids:
            pa = particle_arrays[solid]
            pa.add_property('wij2')
            pa.add_property('n_nbrs', type='int')
            self._get_normals(pa)

    # Private protocol ###################################################
    def _get_edac_nu(self):
        if self.art_nu > 0:
            nu = self.art_nu
            print("Using artificial viscosity for EDAC with nu = %s" % nu)
        else:
            nu = self.nu
            print("Using real viscosity for EDAC with nu = %s" % self.nu)
        return nu

    def _get_normals(self, pa):
        from pysph.tools.sph_evaluator import SPHEvaluator
        from wall_normal import ComputeNormals, SmoothNormals

        pa.add_property('normal', stride=3)
        pa.add_property('normal_tmp', stride=3)

        name = pa.name

        props = ['m', 'rho', 'h']
        for p in props:
            x = pa.get(p)
            if np.all(x < 1e-12):
                msg = f'WARNING: cannot compute normals "{p}" is zero'
                print(msg)

        seval = SPHEvaluator(
            arrays=[pa], equations=[
                Group(equations=[
                    ComputeNormals(dest=name, sources=[name])
                ]),
                Group(equations=[
                    SmoothNormals(dest=name, sources=[name])
                ]),
            ],
            dim=self.dim, domain_manager=self.domain
        )
        seval.evaluate()

    def _get_internal_flow_equations(self):
        edac_nu = self._get_edac_nu()

        iom = self.inlet_outlet_manager
        fluids_with_io = self.fluids
        all_solids = self.solids + self.inviscid_solids
        if iom is not None:
            fluids_with_io = self.fluids + iom.get_io_names()
        all = fluids_with_io + all_solids

        equations = []
        # inlet-outlet
        if iom is not None:
            io_eqns = iom.get_equations(self, self.use_tvf)
            for grp in io_eqns:
                equations.append(grp)

        group1 = []
        for fluid in fluids_with_io:
            group1.append(
                SummationDensity(dest=fluid, sources=all),
            )
            # Add rigid body influence equation
            if len(self.rigid_bodies) > 0.:
                group1.append(
                    SummationDensityOnFluidDueToRigidBody(
                        dest=fluid, sources=self.rigid_bodies),
                )

        if len(all_solids) > 0:
            for solid in all_solids:
                group1.extend([
                    SummationDensity(dest=solid, sources=fluids_with_io),
                    EvaluateNumberDensity(dest=solid, sources=fluids_with_io),
                ])
            for solid in self.solids:
                group1.extend([
                    SetWallVelocity(dest=solid, sources=fluids_with_io),
                    # FIXME: Make this an option. This is useful for kepleian problem.
                    # SlipVelocityExtrapolation(dest=solid, sources=fluids_with_io),
                ])
            for solid in self.inviscid_solids:
                group1.extend([
                    NoSlipVelocityExtrapolation(
                        dest=solid, sources=fluids_with_io),
                    NoSlipAdvVelocityExtrapolation(
                        dest=solid, sources=fluids_with_io)
                ])

        # Compute density of Rigid body as fluid particles
        if len(self.rigid_bodies) > 0:
            for body in self.rigid_bodies:
                group1.extend([
                    # SummationDensityOnRigidBodyDueToFluid(dest=body, sources=fluids_with_io),
                    # SummationDensityOnRigidBodyDueToRigidBody(dest=body,
                    #                                           sources=self.rigid_bodies),
                    EvaluateNumberDensity(dest=body, sources=fluids_with_io+self.rigid_bodies),
                ])
        equations.append(Group(equations=group1))

        if len(all_solids) > 0:
            group_bc = []
            for solid in all_solids:
                group_bc.append(
                    SolidWallPressureBC(
                        dest=solid, sources=fluids_with_io,
                        gx=self.gx, gy=self.gy, gz=self.gz
                    )
                )
            equations.append(Group(equations=group_bc))

        # also set the pressure of the rigid body when acting as fluid particle
        if len(self.rigid_bodies) > 0:
            group_bc = []
            for body in self.rigid_bodies:
                group_bc.append(
                    SolidWallPressureBCRigidBody(
                        dest=body, sources=fluids_with_io,
                        c0=self.c0,
                        gx=self.gx, gy=self.gy, gz=self.gz
                    )
                )
            equations.append(Group(equations=group_bc))

        if self.has_ghosts:
            eq = []
            for fluid in self.fluids:
                eq.append(UpdateGhostProps(dest=fluid, sources=None))
            equations.append(Group(equations=eq, real=False))

        group2 = []
        for fluid in self.fluids:
            group2.append(
                MomentumEquationPressureGradient(
                    dest=fluid, sources=all, gx=self.gx, gy=self.gy,
                    gz=self.gz
                )
            )
            if self.alpha > 0.0:
                sources = fluids_with_io + self.solids
                group2.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=sources, alpha=self.alpha,
                        c0=self.c0
                    )
                )
            if self.nu > 0.0:
                group2.append(
                    MomentumEquationViscosity(
                        dest=fluid, sources=fluids_with_io, nu=self.nu
                    )
                )

            # Add force due to rigid body on fluid
            if len(self.rigid_bodies) > 0:
                group2.append(
                    ForceOnFluidDueToRigidBody(dest=fluid, sources=self.rigid_bodies),
                )

            if len(self.solids) > 0 and self.nu > 0.0:
                group2.append(
                    SolidWallNoSlipBC(
                        dest=fluid, sources=self.solids, nu=self.nu
                    )
                )
            # group2.append(
            #     MomentumEquationArtificialStress(
            #         dest=fluid, sources=fluids_with_io
            #     )
            # )
            group2.append(
                EDACEquation(
                    dest=fluid, sources=fluids_with_io, nu=edac_nu,
                    cs=self.c0, rho0=self.rho0, edac_alpha=self.edac_alpha
                ))
            if len(self.solids) > 0 and self.edac_nu > 0.0:
                group2.append(
                    EDACEquationSolids(
                        dest=fluid, sources=all_solids, nu=edac_nu,
                        cs=self.c0, rho0=self.rho0,
                        edac_alpha=self.edac_alpha
                ))

            if len(self.rigid_bodies) > 0 and edac_nu > 0.0:
                group2.append(
                    EDACEquationRigidBodies(
                        dest=fluid, sources=self.rigid_bodies, nu=edac_nu,
                        cs=self.c0, rho0=self.rho0,
                        edac_alpha=self.edac_alpha
                ))

        equations.append(Group(equations=group2))

        # inlet-outlet
        if iom is not None:
            io_eqns = iom.get_equations_post_compute_acceleration()
            for grp in io_eqns:
                equations.append(grp)

        # Add rigid body equations (Accelerations)
        if len(self.rigid_bodies) > 0:
            g5 = []
            g6 = []
            g7 = []
            g8 = []

            # Force due to fluid on rigid body
            for body in self.rigid_bodies:
                g5.append(BodyForce(
                    dest=body, sources=None,
                    gx=self.gx,
                    gy=self.gy,
                    gz=self.gz))

                g5.append(ForceOnRigidBodyDueToFluid(
                    dest=body, sources=fluids_with_io))

                g5.append(RigidBodyCollision(
                    dest=body, sources=all_solids+self.rigid_bodies))

            equations.append(Group(g5))

            for body in self.rigid_bodies:
                g6.append(RigidBodyMoments(dest=body, sources=None))
                g7.append(RigidBodyMotion(dest=body, sources=None))

            equations.append(Group(g6))
            equations.append(Group(g7))

        return equations
