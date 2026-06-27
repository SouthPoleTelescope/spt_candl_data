# SPT3G 2019-2020 lensing foreground/systematics emulator definition
from jax import config
config.update("jax_enable_x64", True)

from sklearn.preprocessing import StandardScaler
import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx
from gpjax.parameters import PositiveReal

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class LensingSysEmulator:
    """
    Predict Clkk_i/Clkk_fid through emulation, where Clkk_fid is Agora
    lensing reconstruction with fiducial parameters but foregrounds
    set to 0 (which also matches with theory to a few %).

    Expects input array of length 14:

    [Tcal, Pcal, beam1, beam2, beam3, beam4, betapol090, betapol150,
    betapol220, Atsz, Acib150, Acib220, Arad090, Arad150]

    Technically this can predict more than 1 output in one go.

    Example
    -------
    >>>  Lemu = LensingSysEmulator.load(file_emul)
    >>>  Cle = Lemu.predict(params)
    """

    def __init__(self, jitter=1e-6, restarts=5):
        self.models = []
        self.x_mean = None
        self.x_scale = None
        self.y_means = None
        self.y_scales = None

    # ---------- Load ----------
    @classmethod
    def load(emulator, file_emul):
        """
        Load an emulator from disk.

        Parameters
        ----------
        file_emul : str
            Path to a `.npz` file containing saved GP parameters and scalers.

        Returns
        -------
        LensingSysEmulator
            An instance with models and scalers restored.
        """
        raw = jnp.load(file_emul, allow_pickle=True)

        emu = emulator() #just calls parent class

        emu.x_mean = jnp.asarray(raw["x_scaler_mean"])
        emu.x_scale = jnp.asarray(raw["x_scaler_scale"])

        n_models = int(raw["n_models"])
        y_means_list, y_scales_list = [], []

        # Loop through each ell bin
        for i in range(n_models):
            # Collect Y-scaler parameters
            y_means_list.append(jnp.asarray(raw[f"y{i}_scaler_mean"]))
            y_scales_list.append(jnp.asarray(raw[f"y{i}_scaler_scale"]))

            # Load training Dataset
            D = gpx.Dataset(
                X=jnp.asarray(raw[f"X{i}"]), 
                y=jnp.asarray(raw[f"y{i}"])
            )

            # Rebuild kernel, likelihood, and posterior
            kernel = gpx.kernels.RBF(
                lengthscale=jnp.asarray(raw[f"len{i}"]),
                variance=jnp.asarray(raw[f"var{i}"]),
            )

            prior = gpx.gps.Prior(kernel, mean_function=gpx.mean_functions.Zero())
            
            like = gpx.likelihoods.Gaussian(
                num_datapoints=D.n,
                obs_stddev=jnp.asarray(raw[f"noise{i}"]),
            )

            post = prior * like
            emu.models.append((post, D))

        # Stack y-scaler parameters into single JAX arrays for vectorized operations
        emu.y_means = jnp.concatenate(y_means_list)
        emu.y_scales = jnp.concatenate(y_scales_list)

        print(f"Emulator loaded from: {file_emul}")
        return emu



    # ---------- predict ----------

    def predict(self, Xnew):
        """
        Predict the Clkk_i/Clkk_fid ratio for new parameter sets.

        Parameters
        ----------
        Xnew : array_like, shape (14,) or (n_samples, 14)
            Input parameters for which to predict the ratio. Must have 14 columns.

        Returns
        -------
        jnp.ndarray, shape (n_samples, n_models)
            Predicted ratios Clkk_i(Xnew)/Clkk_fid
            for each output dimension.

        Raises
        ------
        AssertionError
            If Xnew does not have exactly 14 features per sample.
        RuntimeError
            If the emulator has not been trained or loaded.
        """
        if self.x_mean is None:
            raise RuntimeError("Emulator has not been loaded. Call .load() first.")

        X = jnp.atleast_2d(jnp.asarray(Xnew, dtype=jnp.float64))
        
        if X.shape[-1] != self.x_mean.shape[0]:
            raise ValueError(f"Input must have {self.x_mean.shape[0]} features, but got {X.shape[-1]}")

        Xnew_scaled = (X - self.x_mean) / self.x_scale

        preds_scaled_list = [
            post.predict(Xnew_scaled, train_data=D).mean
            for post, D in self.models
        ]

        preds_scaled = jnp.stack(preds_scaled_list, axis=1)

        # 3. Differentiable inverse-scaling of the output
        #    This vectorized operation is efficient and traceable.
        preds = preds_scaled * self.y_scales + self.y_means

        return preds





class LensingSysEmulator_scikit:
    def __init__(self):
        # Define the kernel for the GP. Here, we use a constant kernel times an RBF (squared exponential).
        # You may need to tune the length_scale or other parameters.
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*14, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)

    def train(self, param_samples, Cls_samples):
        """
        Trains the GP emulator on a set of cosmological parameters and corresponding Cls values.

        Parameters:
        - param_samples: np.ndarray, shape (n_samples, 5)
          Array of cosmological parameters used to generate the Cls samples.
        - Cls_samples: np.ndarray, shape (n_samples, n_ell)
          Array of Cls for each parameter set. Each row should be the Cls for one set of parameters.
        """
        self.param_samples = param_samples
        self.Cls_samples = Cls_samples

        # Fit the GP model on the training data
        self.gp.fit(param_samples, Cls_samples)
        print("Gaussian Process Emulator trained successfully.")

    def predict(self, params):
        """
        Predicts Cls for new sets of cosmological parameters.

        Parameters:
        - params: np.ndarray, shape (m, 5)
          New parameter values for which to predict Cls.

        Returns:
        - Cls_pred: np.ndarray, shape (m, n_ell)
          Predicted Cls values for each set of input parameters.
        """
        return self.gp.predict(params)


def _fix_scale(scale, floor=1e-12):
    # sklearn sets scale_=1 for constant features; we also clip tiny values
    scale = np.asarray(scale, dtype=np.float64)
    scale = np.where(scale <= 0.0, 1.0, scale)
    scale = np.where(scale < floor, floor, scale)
    return scale


def _fix_sigma_host(sigma, fallback, floor):
    # host-side; used only in save/load/train (not jitted)
    sigma = float(np.asarray(sigma).reshape(-1)[0])
    if (not np.isfinite(sigma)) or (sigma <= 0.0):
        sigma = max(float(fallback), float(floor))
    if sigma < floor:
        sigma = float(floor)
    return sigma


class LensingSysEmulator_gpjax:
    """
    - train(): can use sklearn + scipy optimizer (host-side).
    - predict(): JAX-only so it can be called from inside jit (Candl).
    - save/load: host-side numpy serialization.
    """

    def __init__(
        self,
        jitter=5e-3,
        restarts=20,
        objective="loocv", 
        kernel="matern52", # "rbf" or "matern52"
        max_iters=2000,
        seed=0,
        sigma_floor=1e-12,
        debug=False,
    ):
        self.jitter = float(jitter)
        self.restarts = int(restarts)
        self.objective = str(objective).lower()
        self.kernel = str(kernel).lower()
        self.max_iters = int(max_iters)
        self.seed = int(seed)
        self.sigma_floor = float(sigma_floor)
        self.debug = bool(debug)

        if self.objective not in ["mll", "loocv"]:
            raise ValueError("objective must be 'mll' or 'loocv'")
        if self.kernel not in ["rbf", "matern52"]:
            raise ValueError("kernel must be 'rbf' or 'matern52'")

        # trained objects
        self.models = []
        self.x_scaler = None
        self.y_scalers = []

        # cached JAX scaler params for jittable predict
        self.x_mean = None
        self.x_scale = None
        self.y_mean = None
        self.y_scale = None

    def _cache_scalers_for_jax(self):
        if self.x_scaler is None or len(self.y_scalers) == 0:
            raise RuntimeError("Scalers not initialized (did you train or load?).")

        x_mean = np.asarray(self.x_scaler.mean_, dtype=np.float64)
        x_scale = _fix_scale(self.x_scaler.scale_, floor=1e-12)

        y_mean = np.array([sc.mean_.reshape(-1)[0] for sc in self.y_scalers], dtype=np.float64)
        y_scale = _fix_scale([sc.scale_.reshape(-1)[0] for sc in self.y_scalers], floor=1e-12)

        self.x_mean = jnp.asarray(x_mean)
        self.x_scale = jnp.asarray(x_scale)
        self.y_mean = jnp.asarray(y_mean)
        self.y_scale = jnp.asarray(y_scale)

    def train(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)

        self.x_scaler = StandardScaler().fit(X)
        Xs = jnp.asarray(self.x_scaler.transform(X))
        n_dims = int(Xs.shape[1])

        if self.objective == "loocv":
            obj = lambda p, d: -gpx.objectives.conjugate_loocv(p, d)
        else:
            obj = lambda p, d: -gpx.objectives.conjugate_mll(p, d)

        self.models = []
        self.y_scalers = []

        key = jr.PRNGKey(self.seed)

        for out_i, col in enumerate(Y.T):
            sc_y = StandardScaler().fit(col[:, None])
            self.y_scalers.append(sc_y)
            ys = jnp.asarray(sc_y.transform(col[:, None]).squeeze())

            D = gpx.Dataset(X=Xs, y=ys[:, None])

            dX = Xs[:, None, :] - Xs[None, :, :]
            dist = jnp.linalg.norm(dX, axis=-1)
            dist = dist[~jnp.eye(dist.shape[0], dtype=bool)]
            med_dist = jnp.median(dist)

            init_L = jnp.full((n_dims,), med_dist)
            init_var = jnp.var(ys) + 1e-4

            best_loss = np.inf
            best_post = None

            for _ in range(self.restarts):
                key, sub1, sub2 = jr.split(key, 3)
                L0 = jnp.exp(jr.normal(sub1, (n_dims,))) * init_L
                var0 = jnp.exp(jr.normal(sub2)) * init_var

                if self.kernel == "matern52":
                    k = gpx.kernels.Matern52(lengthscale=PositiveReal(L0), variance=PositiveReal(var0))
                else:
                    k = gpx.kernels.RBF(lengthscale=PositiveReal(L0), variance=PositiveReal(var0))

                prior = gpx.gps.Prior(k, gpx.mean_functions.Zero())
                like = gpx.likelihoods.Gaussian(
                    num_datapoints=D.n,
                    obs_stddev=PositiveReal(max(self.jitter, self.sigma_floor)),
                )
                post = prior * like

                try:
                    post_opt, hist = gpx.fit_scipy(
                        model=post,
                        objective=obj,
                        train_data=D,
                        max_iters=self.max_iters,
                    )
                    loss = float(np.asarray(hist)[-1])
                    if np.isfinite(loss) and (loss < best_loss):
                        best_loss, best_post = loss, post_opt
                except Exception:
                    continue

            if best_post is None:
                raise RuntimeError(f"All restarts failed for output {out_i:02d}.")

            self.models.append((best_post, D))

        self._cache_scalers_for_jax()
        return self

    def predict(self, Xnew):
        if self.x_mean is None or self.y_mean is None:
            raise RuntimeError("Call train() or load() first.")

        Xnew = jnp.asarray(Xnew)
        Xnew = jnp.atleast_2d(Xnew)

        Xs = (Xnew - self.x_mean[None, :]) / self.x_scale[None, :]

        if self.debug:
            jax.debug.print("[emu] Xs finite? {ok}", ok=jnp.all(jnp.isfinite(Xs)))

        preds_scaled = []
        for post, D in self.models:
            latent = post.predict(Xs, train_data=D)
            yhat = post.likelihood(latent)
            mu = jnp.squeeze(yhat.mean)
            mu = jnp.atleast_1d(mu)
            preds_scaled.append(mu)

        preds_scaled = jnp.stack(preds_scaled, axis=1)
        preds = preds_scaled * self.y_scale[None, :] + self.y_mean[None, :]

        if self.debug:
            jax.debug.print("[emu] preds finite? {ok}", ok=jnp.all(jnp.isfinite(preds)))

        return preds

    def save(self, file_emul):
        if not self.models:
            raise RuntimeError("train() must be run before save().")
        if self.x_scaler is None or len(self.y_scalers) == 0:
            raise RuntimeError("Missing sklearn scalers; did you train or load properly?")

        data = {}
        data["jitter"] = np.array(self.jitter)
        data["objective"] = np.array(self.objective)
        data["kernel"] = np.array(self.kernel)
        data["max_iters"] = np.array(self.max_iters)
        data["seed"] = np.array(self.seed)
        data["sigma_floor"] = np.array(self.sigma_floor)

        data["x_scaler_mean"] = np.asarray(self.x_scaler.mean_, dtype=np.float64)
        data["x_scaler_scale"] = np.asarray(_fix_scale(self.x_scaler.scale_), dtype=np.float64)
        data["n_models"] = np.array(len(self.models), dtype=np.int64)

        for i, ((post, D), y_scaler) in enumerate(zip(self.models, self.y_scalers)):
            data[f"y{i}_scaler_mean"] = np.asarray(y_scaler.mean_, dtype=np.float64)
            data[f"y{i}_scaler_scale"] = np.asarray(_fix_scale(y_scaler.scale_), dtype=np.float64)

            data[f"X{i}"] = np.asarray(jax.device_get(D.X), dtype=np.float64)
            data[f"y{i}"] = np.asarray(jax.device_get(D.y), dtype=np.float64)

            kpar = post.prior.kernel
            data[f"len{i}"] = np.asarray(jax.device_get(getattr(kpar.lengthscale, "value", kpar.lengthscale)), dtype=np.float64)
            data[f"var{i}"] = np.asarray(jax.device_get(getattr(kpar.variance, "value", kpar.variance)), dtype=np.float64)

            sigma = np.asarray(jax.device_get(getattr(post.likelihood.obs_stddev, "value", post.likelihood.obs_stddev)), dtype=np.float64)
            sigma = _fix_sigma_host(sigma, fallback=self.jitter, floor=self.sigma_floor)
            data[f"noise{i}"] = np.array(sigma, dtype=np.float64)

        np.savez(file_emul, **data)

        if self.debug:
            print(f"[emu] saved: {file_emul}")

    @classmethod
    def load(cls, file_emul, debug=False):
        raw = np.load(file_emul, allow_pickle=True)

        emu = cls(
            jitter=float(raw.get("jitter", 1e-6)),
            restarts=1,  # not used for loaded model
            objective=str(raw.get("objective", "mll")),
            kernel=str(raw.get("kernel", "rbf")),
            max_iters=int(raw.get("max_iters", 2000)),
            seed=int(raw.get("seed", 0)),
            sigma_floor=float(raw.get("sigma_floor", 1e-12)),
            debug=bool(debug),
        )

        emu.x_scaler = StandardScaler()
        emu.x_scaler.mean_ = raw["x_scaler_mean"]
        emu.x_scaler.scale_ = _fix_scale(raw["x_scaler_scale"])
        emu.x_scaler.n_features_in_ = emu.x_scaler.mean_.shape[0]

        n_models = int(raw["n_models"])
        emu.models = []
        emu.y_scalers = []

        for i in range(n_models):
            y_sc = StandardScaler()
            y_sc.mean_ = raw[f"y{i}_scaler_mean"]
            y_sc.scale_ = _fix_scale(raw[f"y{i}_scaler_scale"])
            y_sc.n_features_in_ = 1
            emu.y_scalers.append(y_sc)

            X_i = jnp.asarray(raw[f"X{i}"])
            y_i = jnp.asarray(raw[f"y{i}"])
            D = gpx.Dataset(X=X_i, y=y_i)

            ls_i = jnp.asarray(raw[f"len{i}"])
            var_i = jnp.asarray(raw[f"var{i}"])

            sigma = _fix_sigma_host(raw[f"noise{i}"], fallback=emu.jitter, floor=emu.sigma_floor)

            if emu.kernel == "matern52":
                kernel = gpx.kernels.Matern52(lengthscale=PositiveReal(ls_i), variance=PositiveReal(var_i))
            else:
                kernel = gpx.kernels.RBF(lengthscale=PositiveReal(ls_i), variance=PositiveReal(var_i))

            prior = gpx.gps.Prior(kernel, gpx.mean_functions.Zero())
            like = gpx.likelihoods.Gaussian(
                num_datapoints=D.n,
                obs_stddev=PositiveReal(jnp.asarray(sigma)),
            )
            post = prior * like

            emu.models.append((post, D))

        emu._cache_scalers_for_jax()

        if emu.debug:
            print(f"[emu] loaded: {file_emul}")

        return emu