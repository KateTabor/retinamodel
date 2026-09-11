"""Core mRGC model calculations from Cells 1-10 of JNeitz_mRGC_v5.ipynb"""

import numpy as np

from retinamodel.reporting import format_mrgc_report

# Parameters
LUT = {
    "wavelength_nm": np.arange(390, 701, 5),
    "mac_od": np.array([0.04533,0.06489,0.0868,0.112,0.1365,0.1631,0.1981,0.2345,0.2618,0.2772,0.2884,0.308,0.3332,0.3486,0.35,0.3269,0.2996,0.2842,0.2786,0.2772,0.2688,0.2485,0.2093,0.1652,0.1211,0.0812,0.0525,0.0329,0.0175,0.00929,0.00459,0.00166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    "lens_od": np.array([2.5122,2.1306,1.7649,1.4257,1.13738,0.9063,0.72398,0.59572,0.4876,0.4081,0.34132,0.29998,0.26288,0.2438,0.2279,0.21306,0.20458,0.19292,0.18338,0.1749,0.16748,0.16006,0.1537,0.14628,0.1378,0.12932,0.12296,0.1166,0.11024,0.10494,0.09858,0.09222,0.08586,0.0795,0.0742,0.06784,0.06148,0.05512,0.04876,0.04346,0.03812,0.03286,0.02968,0.02544,0.02226,0.01908,0.01696,0.01484,0.01166,0.00848,0.0053,0.00424,0.00318,0.00106,0,0,0,0,0,0,0,0,0]),
    "cone_ext_parameters": np.array([0.417050601,0.002072146099,0.0001638878333,-1.922880605,-16.05774461,0.001575425685,0.0000511375521,0.00157980964,0.00006584275242,0.00006684015221,0.002310441807,0.00007313131766,0.00001862685404,0.002008124081,0.00005407174451,0.00000514735556,0.001455412882,0.00004217635152,4.80E-06,0.001809022392,0.00003866766734,2.99E-05,0.001757314922,0.00001473437751,1.51E-05,0.35000001,0.1248091545,0.02274730785,0.0009731002994], dtype=float),
    "input_photopigment_peak_s": 419,
    "input_photopigment_peak_m": 530,
    "input_photopigment_peak_l": 557.25,
    "input_ODmax_s": 0.35,
    "input_ODmax_m": 0.3,
    "input_ODmax_l": 0.3,
    "primary_r": np.array([0.15,0.00,0.00,0.00,0.16,0.67,2.27,4.65,9.27,14.71,20.17,28.36,32.79,26.87,15.82,8.33,5.12,3.28,2.11,1.45,1.00,0.95,0.93,0.89,1.11,1.33,1.43,1.65,1.54,1.37,1.14,1.07,0.77,0.82,0.72,1.13,3.86,20.89,83.73,217.09,406.93,617.30,767.04,841.61,880.56,874.64,833.56,786.86,735.62,683.48,626.69,553.89,502.62,469.58,427.00,386.31,342.35,305.56,262.91,227.58,199.75,179.59,161.84]),
    "primary_g": np.array([0.00,0.05,0.00,0.00,0.00,0.00,0.12,0.55,1.63,4.60,10.72,23.07,37.52,47.36,55.72,71.12,99.87,123.43,131.23,127.34,142.58,174.86,227.38,285.70,388.61,508.82,621.44,644.47,712.30,760.29,789.32,789.25,751.83,742.14,687.43,630.10,573.19,490.28,411.33,330.16,246.74,173.14,111.25,64.28,34.18,16.73,8.20,4.39,2.64,1.83,1.34,1.16,0.95,0.78,0.69,0.72,0.80,1.00,1.15,1.54,2.07,2.67,3.44]),
    "primary_b": np.array([0.21,0.13,0.00,0.00,0.89,4.87,16.55,46.70,118.50,249.45,450.46,819.32,1234.87,1291.28,944.04,599.84,417.24,281.30,181.82,121.72,97.11,91.72,89.71,87.65,86.12,79.72,66.47,48.76,40.28,33.82,29.70,23.65,17.67,12.50,8.21,5.86,4.69,4.07,3.65,3.21,2.82,2.50,2.20,1.87,1.82,1.72,1.90,2.13,2.45,2.96,3.63,4.32,5.13,5.98,6.58,7.01,6.84,6.51,5.91,5.18,4.75,4.65,4.22]),
    "sigma_center": 0.025,
    "sigma_surround": 0.165,
    "surround_ratio_lm": np.array([1/2, 1/2]),
    "surround_ratio_lms": np.array([1/3, 1/3, 1/3]),
    "center_inhib_ratio_lm": np.array([0.5, 0.5]),
}

def validate_lut():
    """Check that wavelength-dependent LUT arrays have matching lengths."""
    n = len(LUT["wavelength_nm"])
    keys = ["mac_od", "lens_od", "primary_r", "primary_g", "primary_b"]
    
    for key in keys:
        if len(LUT[key]) != n:
            raise ValueError(
                f"{key} has {len(LUT[key])} entries, expected {n}"
            )

def _gauss(x, mu, sigma):
    """Return a unit-area Gaussian evaluated at x."""
    return (
        1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * ((x - mu) / sigma) **2)
    )

def cone_log_extinction(wavelength_nm, lambda_max_nm, p, ref_nm=558.5):
    """Calculate cone log extinction using the Carroll equation."""
    shift = np.log10(1.0 / lambda_max_nm) - np.log10(1.0 / ref_nm)
    x = 10.0 ** (np.log10(1.0 / wavelength_nm) - shift)

    part1_inside = -p["E"] + p["E"] * np.tanh(-((x - p["F"]) / p["G"]))
    part1 = (
        np.log10(part1_inside)
        + p["D"]
        + p["A"] * np.tanh(-((x - p["B"]) / p["C"]))
    )

    part2 = (
        -(p["J"] / p["I"]) * _gauss(x, p["H"], p["I"])
        -(p["M"] / p["L"]) * _gauss(x, p["K"], p["L"])
        -(p["P"] / p["O"]) * _gauss(x, p["N"], p["O"])
        +(p["S"] / p["R"]) * _gauss(x, p["Q"], p["R"])
        +(p["V"] / p["U"]) * _gauss(x, p["T"], p["U"]) / 10.0
        +(p["Y"] / p["X"]) * _gauss(x, p["W"], p["X"]) / 100.0
    )

    return part1 + part2

def apply_optical_density(cone_log_extinction, input_od_max, mac_od, lens_od):
    """Apply photopigment, macular, and lens optical density."""
    cone_linear_extinction = 10.0 ** cone_log_extinction

    numerator = 1.0 - 10.0 ** (-(cone_linear_extinction * input_od_max))
    denominator = 1.0 - 10.0 ** (-input_od_max)

    step1 = np.log10(numerator / denominator) - lens_od - mac_od
    final = 10.0 ** (step1 + 2)

    return final

def gaussian_2d(x, y, amplitude, sigma):
    """Return a two-dimensional Gaussian evaluated at x and y."""
    return amplitude * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

class MRGCModel:
    """The fixed mRGC model used in JNeitz_mRGC_v5."""

    def __init__(self):
        validate_lut()
        self.lut = LUT

        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXY")
        self.cone_params = dict(zip(letters, self.lut["cone_ext_parameters"][:25]))

        self._compute_log_extinctions()
        self._compute_effective_cone_sensitivities()
        self._compute_surround_sensitivities()
        self._compute_reference_light_spectra()
        self._compute_reference_responses()
        self._compute_scalars()
        self._build_spatial_receptive_fields()
        self._compute_white_reference_outputs()
        self._validate_white_reference_outputs()
        self._compute_yellow_reference_outputs()

    def _compute_log_extinctions(self):
        """Compute S-, M-, and L-cone log extinction spectra."""
        self.log_ext_s = cone_log_extinction(
            self.lut["wavelength_nm"],
            self.lut["input_photopigment_peak_s"],
            self.cone_params,
        )

        self.log_ext_m = cone_log_extinction(
            self.lut["wavelength_nm"],
            self.lut["input_photopigment_peak_m"],
            self.cone_params,
        )

        self.log_ext_l = cone_log_extinction(
            self.lut["wavelength_nm"],
            self.lut["input_photopigment_peak_l"],
            self.cone_params,
        )

    def _compute_effective_cone_sensitivities(self):
        """Compute optical-density-adjusted S-, M-, and L-cone sensitivities."""
        self.ext_s = apply_optical_density(
            self.log_ext_s,
            self.lut["input_ODmax_s"],
            self.lut["mac_od"],
            self.lut["lens_od"],
        )

        self.ext_m = apply_optical_density(
            self.log_ext_m,
            self.lut["input_ODmax_m"],
            self.lut["mac_od"],
            self.lut["lens_od"],
        )

        self.ext_l = apply_optical_density(
            self.log_ext_l,
            self.lut["input_ODmax_l"],
            self.lut["mac_od"],
            self.lut["lens_od"],
        )

    def _compute_surround_sensitivities(self):
        """Compute the cone mixtures used for receptive-field surrounds."""
        self.ext_surround_lm = (
            self.lut["surround_ratio_lm"][0] * self.ext_m
            + self.lut["surround_ratio_lm"][1] * self.ext_l
        )

        self.ext_surround_lms = (
            self.lut["surround_ratio_lms"][0] * self.ext_s
            + self.lut["surround_ratio_lms"][1] * self.ext_m
            + self.lut["surround_ratio_lms"][2] * self.ext_l
        )

        self.ext_center_inhib_lm = (
            self.lut["center_inhib_ratio_lm"][0] * self.ext_m
            + self.lut["center_inhib_ratio_lm"][1] * self.ext_l
        )

    def _compute_reference_light_spectra(self):
        """Compute the fixed white- and yellow-light primary spectra."""
        self.white = np.array([1.0, 1.0, 1.0])
        self.yellow = np.array([1.0, 1.0, 0.0])

        self.primaries = np.stack(
            [
                self.lut["primary_r"],
                self.lut["primary_g"],
                self.lut["primary_b"],
            ],
            axis=1,
        )

        self.primary_white = (self.white * self.primaries).sum(axis=1)
        self.primary_yellow = (self.yellow * self.primaries).sum(axis=1)

        self.total_quanta_white = self.primary_white.sum()
        self.total_quanta_yellow = self.primary_yellow.sum()

    def _compute_reference_responses(self):
        """Compute cone and surround responses to the reference lights."""
        self.response_white_lm = np.sum(
            self.ext_surround_lm * self.primary_white
        )
        self.response_white_l = np.sum(
            self.ext_l * self.primary_white
        )
        self.response_white_m = np.sum(
            self.ext_m * self.primary_white
        )
        self.response_white_s = np.sum(
            self.ext_s * self.primary_white
        )
        self.response_white_lms = np.sum(
            self.ext_surround_lms * self.primary_white
        )
        self.response_white_inhib_lm = np.sum(
            self.ext_center_inhib_lm * self.primary_white
        )

        self.response_yellow_lm = np.sum(
            self.ext_surround_lm * self.primary_yellow
        )
        self.response_yellow_l = np.sum(
            self.ext_l * self.primary_yellow
        )
        self.response_yellow_m = np.sum(
            self.ext_m * self.primary_yellow
        )

    def _compute_scalars(self):
        """Compute scaling factors that balance center and surround responses."""
        self.scalar_l = self.response_white_lm / self.response_white_l
        self.scalar_m = self.response_white_lm / self.response_white_m
        self.scalar_s = self.response_white_lm / self.response_white_s
        self.scalar_lms = self.response_white_lm / self.response_white_lms
        self.scalar_s_inhib_lm = (
            self.response_white_inhib_lm / self.response_white_s
        )

    def _build_spatial_receptive_fields(self):
        """Build the fixed 2D center-surround receptive fields."""
        self.radius = 1.05 # in degrees
        self.dx = 0.005 # degrees per pixel

        self.x = np.arange(
            -self.radius,
            self.radius + self.dx,
            self.dx,
        )
        self.X, self.Y = np.meshgrid(self.x, self.x)

        self.amp_center = 1.0
        self.amp_surround = (
            self.amp_center
            * self.lut["sigma_center"] ** 2
            / self.lut["sigma_surround"] ** 2
        )

        self.gaussian_center = gaussian_2d(
            self.X,
            self.Y,
            self.amp_center,
            self.lut["sigma_center"],
        )

        self.gaussian_surround = gaussian_2d(
            self.X,
            self.Y,
            self.amp_surround,
            self.lut["sigma_surround"],
        )

        self.rf_on = self.gaussian_center - self.gaussian_surround
        self.rf_off = -self.rf_on

    def _compute_white_reference_outputs(self):
        """Compute uniform-field mRGC responses under white light."""
        center_integral = self.gaussian_center.sum() * self.dx * self.dx
        surround_integral = self.gaussian_surround.sum() * self.dx * self.dx

        self.l_center_white = (
            self.scalar_l * self.response_white_l * center_integral
        )

        self.m_center_white = (
            self.scalar_m * self.response_white_m * center_integral
        )

        self.s_center_white = (
            self.scalar_s * self.response_white_s * center_integral
        )

        self.surround_white = (
            self.response_white_lm * surround_integral
        )

        self.lm_center_white = (
            self.response_white_inhib_lm * center_integral
        )

        self.dog_l_white = self.l_center_white - self.surround_white
        self.dog_m_white = self.m_center_white - self.surround_white

        self.dog_h2_white = (
            self.scalar_s * self.response_white_s * center_integral
            - self.scalar_lms * self.response_white_lms * surround_integral
        )

        self.dog_s_white = (
            self.scalar_s_inhib_lm * self.response_white_s * center_integral
            - self.response_white_inhib_lm * center_integral
        )

    def _validate_white_reference_outputs(self):
        """Assert that white-light DoG responses are close to zero."""
        dog_values = [
            self.dog_l_white,
            self.dog_m_white,
            self.dog_h2_white,
            self.dog_s_white,
        ]

        assert max(abs(value) for value in dog_values) <= 1.0e-04, (
            "White-light DoG validation failed: "
            f"one or more magnitudes exceed 1.0e-04. Values: {dog_values}"
        )

    def _compute_yellow_reference_outputs(self):
        """Compute uniform-field mRGC responses under yellow light."""
        center_integral = self.gaussian_center.sum() * self.dx * self.dx
        surround_integral = self.gaussian_surround.sum() * self.dx * self.dx

        self.l_center_yellow = (
            self.scalar_l * self.response_yellow_l * center_integral
        )

        self.m_center_yellow = (
            self.scalar_m * self.response_yellow_m * center_integral
        )

        self.surround_yellow = (
            self.response_yellow_lm * surround_integral
        )

        self.dog_l_yellow = self.l_center_yellow - self.surround_yellow
        self.dog_m_yellow = self.m_center_yellow - self.surround_yellow

        self.percent_dog_l_yellow = (
            self.dog_l_yellow / self.l_center_yellow
        )

        self.percent_dog_m_yellow = (
            self.dog_m_yellow / self.m_center_yellow
        )

    def print_reference_outputs(self):
        """Print reference values for checking that the mRGC model is working."""
        print("L center under white light: ", self.l_center_white)
        print("M center under white light: ", self.m_center_white)
        print("S center under white light: ", self.s_center_white)
        print("surround under white light: ", self.surround_white)
        print("LM center under white light: ", self.lm_center_white)

        print(
            "DoG under white light (L-center, M-center, H2, S-center): ",
            self.dog_l_white,
            self.dog_m_white,
            self.dog_h2_white,
            self.dog_s_white,
        )

        print()

        print("L center under yellow light: ", self.l_center_yellow)
        print("M center under yellow light: ", self.m_center_yellow)
        print("surround under yellow light: ", self.surround_yellow)

        print(
            "DoG under yellow light (L-center, M-center): ",
            self.dog_l_yellow,
            self.dog_m_yellow,
        )

        print(
            "Percentage of center, DoG under yellow light (L-center, M-center): ",
            self.percent_dog_l_yellow,
            self.percent_dog_m_yellow,
        )

def main():
    """Build the mRGC model and print reference outputs."""
    model = MRGCModel()
    print(format_mrgc_report(model))