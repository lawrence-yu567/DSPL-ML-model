import numpy as np
import matplotlib
#matplotlib.use("Agg")  # remove/comment out if running interactively and want plt.show()
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c as light_speed

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Util import util, image_util
import lenstronomy.Util.simulation_util as sim_util


rng = np.random.default_rng(seed=42)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

background_rms = .005  #  background noise per pixel
exp_time = 1000.  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
numPix = 120  #  cutout pixel size per axis
deltaPix = 0.05  #  pixel size in arcsec (area per pixel = pixel_scale**2)
fwhm = 0.05  # full width at half maximum of PSF
psf_type = 'GAUSSIAN'


#define redshifts
zl = 0.222
zs1 = 0.609
zs2 = 2.025


#parameterisation functions 
ARCSEC_PER_RAD = 206264.80624709636
c = light_speed.to("km/s").value

def velocity_dispersion_to_theta_e(sig_v, zl, zs):
    d_s = cosmo.angular_diameter_distance(zs).value
    d_ls = cosmo.angular_diameter_distance_z1z2(zl, zs).value
    return 4 * np.pi * (sig_v/c)**2 * (d_ls/d_s) * ARCSEC_PER_RAD

def axis_ratio_to_e1e2(q, phi):
    e = (1.0 - q) / (1.0 + q)
    e1 = e * np.cos(2 * phi)
    e2 = e * np.sin(2 * phi)
    return e1, e2


def build_sie_kwargs(sig_v, q, phi, zl, zs, center_x=0.0, center_y=0.0):
    """Return an SIE kwargs dict for lenstronomy from CMU DeepLens params."""
    theta_E = velocity_dispersion_to_theta_e(sig_v, zl, zs)
    e1, e2 = axis_ratio_to_e1e2(q, phi)
    return {
        "theta_E": theta_E,
        "e1": e1,
        "e2": e2,
        "center_x": center_x,
        "center_y": center_y,
    }

def beta_scaling(zl, zs_from, zs_to):
    d_s_from = cosmo.angular_diameter_distance(zs_from).value
    d_ls_from = cosmo.angular_diameter_distance_z1z2(zl, zs_from).value
    d_s_to = cosmo.angular_diameter_distance(zs_to).value
    d_ls_to = cosmo.angular_diameter_distance_z1z2(zl, zs_to).value
    return (d_ls_to / d_s_to) / (d_ls_from / d_s_from)

def generate(rng):
    sig_v = rng.uniform(150, 300)
    q = rng.uniform(0.5, 1.0)
    phi = rng.uniform(0, 2*np.pi)
    return sig_v, q, phi

def direct_ratio_check(zl, zs_from, zs_to):
    d_s_from = cosmo.angular_diameter_distance(zs_from).value
    d_ls_from = cosmo.angular_diameter_distance_z1z2(zl, zs_from).value
    d_s_to = cosmo.angular_diameter_distance(zs_to).value
    d_ls_to = cosmo.angular_diameter_distance_z1z2(zl, zs_to).value
    return (d_ls_to * d_s_from) / (d_s_to * d_ls_from)


l_sie_kwargs_s1 = {'theta_E': 1.397, 'e1': -0.01333, 'e2': -0.00488, 'center_x': 0.018, 'center_y': 0.059,}
beta_l = beta_scaling(zl, zs1, zs2) #rescaling second einstein ring
l_sie_kwargs_s2 = dict(l_sie_kwargs_s1)
l_sie_kwargs_s2["theta_E"] = l_sie_kwargs_s1["theta_E"] * beta_l

kwargs_shear = {
    'gamma1': 0.05924,
    'gamma2': -0.06908,
}

#instead of sie, its an sis lens
s1_sis_kwargs = {
    'theta_E': 0.133,
    'center_x': 0.020,
    'center_y': -0.017,
}


#leave sheer for now
#keep light profile fixed, can think about varying this as well
l_sersic_kwargs = [{'amp': 16, 'R_sersic': 0.6, 'n_sersic': 2, 'e1': -0.1, 'e2': 0.1,
                     'center_x': 0.05, 'center_y': 0}]
s1_sersic_kwargs = [{'amp': 16, 'R_sersic': 0.1, 'n_sersic': 1, 'e1': -0.1, 'e2': 0.1,
                      'center_x': 0.1, 'center_y': 0}]
s2_sersic_kwargs = [{'amp': 12, 'R_sersic': 0.08, 'n_sersic': 1, 'e1': 0.15, 'e2': -0.05,
                      'center_x': -0.10, 'center_y': 0.05}]



#light model class
l_light_model = LightModel(
    light_model_list=['SERSIC_ELLIPSE']
)

s1_light_model  = LightModel(
    light_model_list=['SERSIC_ELLIPSE']
)

s2_light_model = LightModel(
    light_model_list=['SERSIC_ELLIPSE']
)
#lens model 

#this is to model s1 light getting lensed by the lens
s1_lens_model = LensModel(
    lens_model_list=['SIE', 'SHEAR'],
    lens_redshift_list=[zl, zl],
    z_source=zs1,
    multi_plane=True,
    cosmo=cosmo
)

l_s1_kwargs = [l_sie_kwargs_s1, kwargs_shear]


#this is to model how s2 light getting lensed twice
s2_lens_model = LensModel(
    lens_model_list=['SIE','SHEAR', 'SIS'], #can add shear of s1
    lens_redshift_list=[zl, zl, zs1],
    z_source=zs2,
    multi_plane=True,
    cosmo=cosmo
)

l_s1_s2_kwargs = [l_sie_kwargs_s2, kwargs_shear, s1_sis_kwargs]
kwargs_data = sim_util.data_configure_simple(
    numPix,
    deltaPix,
    exposure_time=exp_time,
    background_rms=background_rms,
)
kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 3}
kwargs_numerics = {"supersampling_factor": 2}

data_class = ImageData(**kwargs_data)
psf_class = PSF(**kwargs_psf)
#2 image models
#s1
s1_image_model = ImageModel(
    data_class=data_class,
    psf_class=psf_class,
    lens_model_class=s1_lens_model,
    lens_light_model_class=l_light_model,
    source_model_class=s1_light_model,
    kwargs_numerics=kwargs_numerics
)

#s2
s2_image_model = ImageModel(
    data_class=data_class,
    psf_class=psf_class,
    lens_model_class=s2_lens_model,
    lens_light_model_class=None,
    source_model_class=s2_light_model,
    kwargs_numerics=kwargs_numerics
)

s1_image = s1_image_model.image(
    kwargs_lens=l_s1_kwargs,
    kwargs_source=s1_sersic_kwargs,
    kwargs_lens_light=l_sersic_kwargs
)

s2_image = s2_image_model.image(
    kwargs_lens=l_s1_s2_kwargs,
    kwargs_source=s2_sersic_kwargs,
    kwargs_lens_light=None
)

image = s1_image + s2_image
#test cell

x_grid, y_grid = util.make_grid(
    numPix=numPix,
    deltapix=deltaPix,
)

beta_x1, beta_y1 = s1_lens_model.ray_shooting(
    x_grid,
    y_grid,
    l_s1_kwargs
)

beta_x2, beta_y2 = s2_lens_model.ray_shooting(
    x_grid,
    y_grid,
    l_s1_s2_kwargs
)

alpha_x1, alpha_y1 = s1_lens_model.alpha(
    x_grid,
    y_grid,
    l_s1_kwargs
)

alpha_x2, alpha_y2 = s2_lens_model.alpha(
    x_grid,
    y_grid,
    l_s1_s2_kwargs
)

x_grid_2d = x_grid.reshape((numPix, numPix))
y_grid_2d = y_grid.reshape((numPix, numPix))

beta_x1_2d = beta_x1.reshape((numPix, numPix))
beta_y1_2d = beta_y1.reshape((numPix, numPix))

beta_x2_2d = beta_x2.reshape((numPix, numPix))
beta_y2_2d = beta_y2.reshape((numPix, numPix))

alpha_x1_2d = alpha_x1.reshape((numPix, numPix))
alpha_y1_2d = alpha_y1.reshape((numPix, numPix))

alpha_x2_2d = alpha_x2.reshape((numPix, numPix))
alpha_y2_2d = alpha_y2.reshape((numPix, numPix))


fig, ax = plt.subplots(
    figsize=(8, 8)
)

im = ax.imshow(
    np.log10(
        np.clip(image, 1e-6, None)
    ),
    origin="lower",
    cmap="viridis",
)

ax.set_title(
    "Image — ideal"
)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()

plt.show()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(
    beta_x2_2d,
    origin="lower",
    cmap="coolwarm"
)
axes[0].set_title(r"$\beta_x$")

im2 = axes[1].imshow(
    beta_y2_2d,
    origin="lower",
    cmap="coolwarm"
)
axes[1].set_title(r"$\beta_y$")

plt.tight_layout()
plt.show()
# S2 lensed only by the main foreground lens


fig, axes = plt.subplots(1, 2, figsize=(20, 5))

imgs = [
    s1_image,
    s2_image
]

titles = [
    "S1 source",
    "S2"
]

for ax, img, title in zip(axes, imgs, titles):
    ax.imshow(
        np.log10(np.clip(img, 1e-6, None)),
        origin="lower",
        cmap="viridis"
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

poisson_noise = image_util.add_poisson(
    image,
    exp_time=exp_time,
)

background_noise = image_util.add_background(
    image,
    sigma_bkd=background_rms,
)

image_noisy = (image + poisson_noise + background_noise)
x_grid, y_grid = util.make_grid(
    numPix=numPix,
    deltapix=deltaPix,
)

beta_x1, beta_y1 = s1_lens_model.ray_shooting(
    x_grid,
    y_grid,
    l_s1_kwargs
)

beta_x2, beta_y2 = s2_lens_model.ray_shooting(
    x_grid,
    y_grid,
    l_s1_s2_kwargs
)

alpha_x1, alpha_y1 = s1_lens_model.alpha(
    x_grid,
    y_grid,
    l_s1_kwargs
)

alpha_x2, alpha_y2 = s2_lens_model.alpha(
    x_grid,
    y_grid,
    l_s1_s2_kwargs
)
x_grid_2d = x_grid.reshape((numPix, numPix))
y_grid_2d = y_grid.reshape((numPix, numPix))

beta_x1_2d = beta_x1.reshape((numPix, numPix))
beta_y1_2d = beta_y1.reshape((numPix, numPix))

beta_x2_2d = beta_x2.reshape((numPix, numPix))
beta_y2_2d = beta_y2.reshape((numPix, numPix))

alpha_x1_2d = alpha_x1.reshape((numPix, numPix))
alpha_y1_2d = alpha_y1.reshape((numPix, numPix))

alpha_x2_2d = alpha_x2.reshape((numPix, numPix))
alpha_y2_2d = alpha_y2.reshape((numPix, numPix))

fig, ax = plt.subplots(
    figsize=(8, 8)
)

im = ax.imshow(
    np.log10(
        np.clip(image_noisy, 1e-6, None)
    ),
    origin="lower",
    cmap="viridis",
)

ax.set_title(
    "Combined DSPL Image — noisy"
)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()

plt.show()
