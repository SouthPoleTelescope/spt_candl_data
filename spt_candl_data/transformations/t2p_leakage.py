# --------------------------------------#
# IMPORTS
# --------------------------------------#

from candl.lib import *
import candl.transformations.abstract_base
import candl.constants
from candl.lib import jax_optional_set_element, jit, jnp
import candl.io
import candl.likelihood


# --------------------------------------#
# LEAKAGE MODULE
# --------------------------------------#


class Leakage(candl.transformations.abstract_base.Transformation):
    def __init__(
        self,
        sigmas,
        param_dict,
        ells,
        long_ells,
        spec_freqs,
        spec_order,
        spec_types,
        operation_hint="additive",
        descriptor="T2P leakage",
    ):
        self.sigmas = sigmas
        self.spec_order = spec_order
        self.spec_freqs = spec_freqs
        self.freqs = np.unique(spec_freqs)
        self.nfreqs = len(self.freqs)
        self.amp_params = []
        for fr in self.freqs:
            self.amp_params += [par for par in param_dict[fr].values()]
        super().__init__(
            ells, descriptor, param_names=self.amp_params, operation_hint=operation_hint
        )

        self.spec_types = [list(k) for k in spec_types]
        self.spec_tuples = []
        for i in range(len(spec_freqs)):
            self.spec_tuples.append([])
            for j in range(len(spec_freqs[i])):
                self.spec_tuples[i].append(f"{spec_types[i][j]}{spec_freqs[i][j]}")
        self.spec_tuples_unique = list(np.unique(self.spec_tuples))

        self.ells = ells
        self.long_ells = long_ells
        self.total_ells = self.long_ells.size

        self.mat_spec_to_map = jnp.zeros(
            (
                len(self.spec_tuples_unique),
                len(self.spec_tuples_unique),
                len(self.spec_types),
            )
        )
        for i, spec_tuple in enumerate(self.spec_tuples):
            left, right = self.spec_tuples_unique.index(
                spec_tuple[0]
            ), self.spec_tuples_unique.index(spec_tuple[1])
            self.mat_spec_to_map = jax_optional_set_element(
                self.mat_spec_to_map, (left, right, i), 1
            )
            self.mat_spec_to_map = jax_optional_set_element(
                self.mat_spec_to_map, (right, left, i), 1
            )

        self.nonzero = (self.mat_spec_to_map @ np.arange(1, len(self.spec_order) + 1))[
            np.triu_indices(len(self.spec_tuples_unique))
        ].nonzero()
        self.argsort = (self.mat_spec_to_map @ np.arange(1, len(self.spec_order) + 1))[
            np.triu_indices(len(self.spec_tuples_unique))
        ][self.nonzero].argsort()

        self.leakage_matrices = {}
        self.leakage_order = {}
        for fr in self.freqs:
            assert isinstance(param_dict[fr], dict)
            for order, par in param_dict[fr].items():
                self.sigmas[par] = sigmas[fr]
                self.leakage_order[par] = order
                self.leakage_matrices[par] = jnp.zeros(
                    (len(self.spec_tuples_unique), len(self.spec_tuples_unique))
                )
                if (
                    f"T{fr}" in self.spec_tuples_unique
                    and f"E{fr}" in self.spec_tuples_unique
                ):
                    self.leakage_matrices[par] = jax_optional_set_element(
                        self.leakage_matrices[par],
                        (
                            self.spec_tuples_unique.index(f"T{fr}"),
                            self.spec_tuples_unique.index(f"E{fr}"),
                        ),
                        1,
                    )
        # to extend to higher order leakages, result = np.einsum('ijk,jlk->ilk', A, B)  # Matrix multiplication on the first two dims

    def spec_to_map(self, Dls):
        return self.mat_spec_to_map @ Dls.reshape(-1, self.ells.size)

    def map_to_spec(self, dl3d):
        return dl3d[np.triu_indices(len(self.spec_tuples_unique))][self.nonzero][
            self.argsort
        ].reshape(self.long_ells.size)

    @partial(jit, static_argnums=(0,))
    def output(self, Dls, sample_params):
        leakmat = jnp.eye(len(self.spec_tuples_unique))[:, :, jnp.newaxis] * jnp.ones(
            self.ells.size
        ) + jnp.sum(
            jnp.array(
                [
                    sample_params[par]
                    * self.sigmas[par] ** self.leakage_order[par]
                    * self.leakage_matrices[par][:, :, jnp.newaxis]
                    * (self.ells ** self.leakage_order[par])
                    for par in self.amp_params
                ]
            ),
            axis=0,
        )
        return (
            self.map_to_spec(
                jnp.einsum(
                    "ijm,jkm,klm->ilm",
                    jnp.transpose(leakmat, axes=(1, 0, 2)),
                    self.spec_to_map(Dls),
                    leakmat,
                )
            )
            - Dls
        )

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        return Dls + self.output(Dls, sample_params)


class LeakageFromData(candl.transformations.abstract_base.Transformation):
    def __init__(
        self,
        sigmas,
        ells,
        spec_order,
        param_dict,
        spec_freqs,
        data_set_dict,
        data_file,
        # external_calibration="Tcal_ext150",
        operation_hint="additive",
        descriptor="T2P leakage",
    ):

        # First, I need to read the (full) data band powers, and use candl to provide me
        # with a few tools to build a data Dls array from it.
        self.full_data_set_dict, self.data_set_file = candl.io.load_info_yaml(
            os.path.join(data_set_dict["data_set_path"], data_file)
        )
        self.full_data_set_dict["data_set_path"] = data_set_dict["data_set_path"]

        (
            self.full_spec_order,
            self.full_spec_types,
            self.full_spec_freqs,
            self.full_N_spectra_total,
            self.full_N_bins,
        ) = candl.io.read_spectrum_info_from_yaml(self.full_data_set_dict)

        self.bins_stop_ix = np.cumsum(self.full_N_bins)
        self.bins_start_ix = np.insert(self.bins_stop_ix[:-1], 0, 0)

        self.data_band_powers = candl.io.read_file_from_yaml(
            self.full_data_set_dict, "band_power_file"
        )

        # I need the windows function to build the data Dls from the band powers
        # I will use the window functions to build the pseudo-inverse of the window functions
        self.window_functions = candl.io.read_window_functions_from_yaml(
            self.full_data_set_dict, self.full_spec_order, self.full_N_bins
        )
        self.wf_pinv = [np.linalg.pinv(wf) for wf in self.window_functions]

        # get ell helpers
        N_ell_bins_theory = int(jnp.shape(self.window_functions[0])[0])
        self.full_ells = jnp.arange(2, N_ell_bins_theory + 2)
        tiled_ells = jnp.tile(self.full_ells, (self.full_N_spectra_total, 1))
        self.long_ells = tiled_ells.flatten()

        self.data_Dls = jnp.block(
            [
                jnp.dot(
                    wf_pinv.T,
                    self.data_band_powers[
                        self.bins_start_ix[ix] : self.bins_stop_ix[ix]
                    ],
                )
                for ix, wf_pinv in enumerate(self.wf_pinv)
            ]
        )

        # the crop mask is used to crop the leakage Dls to the desired ells
        self.crop_mask = np.array([sp in spec_order for sp in self.full_spec_order])
        self.crop_mask = np.repeat(self.crop_mask, ells.size)

        # Now I can proceed with the rest of the initialization
        self.sigmas = sigmas
        self.spec_freqs = spec_freqs
        self.freqs = np.unique(spec_freqs)
        self.nfreqs = len(self.freqs)
        self.amp_params = []
        for fr in self.freqs:
            self.amp_params += [par for par in param_dict[fr].values()]
        # self.external_calibration = external_calibration
        super().__init__(
            ells,
            descriptor,
            param_names=self.amp_params, # + [self.external_calibration],
            operation_hint=operation_hint,
        )

        # All of the rest can be built for the full data
        self.full_spec_tuples = []
        for i in range(len(self.full_spec_freqs)):
            self.full_spec_tuples.append([])
            for j in range(len(self.full_spec_freqs[i])):
                self.full_spec_tuples[i].append(
                    f"{self.full_spec_types[i][j]}{self.full_spec_freqs[i][j]}"
                )
        self.full_spec_tuples_unique = list(np.unique(self.full_spec_tuples))

        self.mat_spec_to_map = jnp.zeros(
            (
                len(self.full_spec_tuples_unique),
                len(self.full_spec_tuples_unique),
                len(self.full_spec_types),
            )
        )
        for i, spec_tuple in enumerate(self.full_spec_tuples):
            left, right = self.full_spec_tuples_unique.index(
                spec_tuple[0]
            ), self.full_spec_tuples_unique.index(spec_tuple[1])
            self.mat_spec_to_map = jax_optional_set_element(
                self.mat_spec_to_map, (left, right, i), 1
            )
            self.mat_spec_to_map = jax_optional_set_element(
                self.mat_spec_to_map, (right, left, i), 1
            )

        self.nonzero = (
            self.mat_spec_to_map @ np.arange(1, len(self.full_spec_order) + 1)
        )[np.triu_indices(len(self.full_spec_tuples_unique))].nonzero()
        self.argsort = (
            self.mat_spec_to_map @ np.arange(1, len(self.full_spec_order) + 1)
        )[np.triu_indices(len(self.full_spec_tuples_unique))][self.nonzero].argsort()

        self.leakage_matrices = {}
        self.leakage_order = {}
        for fr in self.freqs:
            assert isinstance(param_dict[fr], dict)
            for order, par in param_dict[fr].items():
                self.sigmas[par] = sigmas[fr]
                self.leakage_order[par] = order
                self.leakage_matrices[par] = jnp.zeros(
                    (
                        len(self.full_spec_tuples_unique),
                        len(self.full_spec_tuples_unique),
                    )
                )
                if (
                    f"T{fr}" in self.full_spec_tuples_unique
                    and f"E{fr}" in self.full_spec_tuples_unique
                ):
                    self.leakage_matrices[par] = jax_optional_set_element(
                        self.leakage_matrices[par],
                        (
                            self.full_spec_tuples_unique.index(f"T{fr}"),
                            self.full_spec_tuples_unique.index(f"E{fr}"),
                        ),
                        1,
                    )
        # to extend to higher order leakages, result = np.einsum('ijk,jlk->ilk', A, B)  # Matrix multiplication on the first two dims

    def spec_to_map(self):
        return self.mat_spec_to_map @ self.data_Dls.reshape(-1, self.full_ells.size)

    def map_to_spec(self, dl3d):
        return dl3d[np.triu_indices(len(self.full_spec_tuples_unique))][self.nonzero][
            self.argsort
        ].reshape(self.long_ells.size)

    def output(self, sample_params):
        leakmat = jnp.eye(len(self.full_spec_tuples_unique))[
            :, :, jnp.newaxis
        ] * jnp.ones(self.full_ells.size) + jnp.sum(
            jnp.array(
                [
                    sample_params[par]
                    * self.sigmas[par] ** self.leakage_order[par]
                    * self.leakage_matrices[par][:, :, jnp.newaxis]
                    * (self.ells ** self.leakage_order[par])
                    for par in self.amp_params
                ]
            ),
            axis=0,
        )
        return (
            self.map_to_spec(
                jnp.einsum(
                    "ijm,jkm,klm->ilm",
                    jnp.transpose(leakmat, axes=(1, 0, 2)),
                    self.spec_to_map(),# / sample_params[self.external_calibration] ** 2,
                    leakmat,
                )
            )
            - self.data_Dls
        )[self.crop_mask]

    @partial(jit, static_argnums=(0,))
    def transform(self, Dls, sample_params):
        return Dls + self.output(sample_params)
