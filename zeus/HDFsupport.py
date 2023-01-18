import os
import numpy as np
import h5py

class HDFStorage:
    """A HDF Storage utility, based on 21CMMC v3"""

    def __init__(self, filename, name):
        if h5py is None:
            raise ImportError("You must install h5py to use HDFBackend.")
        self.filename = filename
        self.name = name
        print("Hello")

    @property
    def initialized(self):
        """Whether the file object has been initialized."""
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OsError, IoError):
            return False

    def open(self, mode="r"):
        """Open the backend file."""
        return h5py.File(self.filename, mode)

    def reset(self, nwalkers, ndim):
        """Clear the state of the chain and empty the backend.

        Parameters
        ----------
        nwalkers: int
            The size of the ensemble
        ndim: int
            Number of walkers.
        """
        if os.path.exists(self.filename):
            mode = "a"
        else:
            mode = "w"

        with self.open(mode) as f:
            if self.name in f:
                del f[self.name]
            g = f.create_group(self.name)
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["has_blobs"]  =False
            g.attrs["iteration"] = 0

            g.create_dataset(
                "accepted", (0,nwalkers), maxshape=(None,nwalkers), dtype=np.int64
            )
            g.create_dataset(
                "chain",
                (0, nwalkers, ndim),
                maxshape = (None, nwalkers, ndim),
                dtype = np.float64,
            )
            g.create_dataset(
                "log_prob",
                (0, nwalkers),
                maxshape = (None, nwalkers),
                dtype = np.float64
            )
            g.create_dataset(
                "trials",
                (0, nwalkers, ndim),
                maxshape = (None, nwalkers, ndim),
                dtype = np.float64,
            )
            g.create_dataset(
                "trial_log_prob",
                (0,nwalkers),
                maxshape = (None, nwalkers),
                dtype = np.float64,
            )

    @property
    def blob_names(self):
        """Names for each of the arbitrary blobs."""
        if not self.has_blobs:
            return None

        empty_blobs = self.get_blobs(discard=self.iteration)
        return empty_blobs.dtype.names

    @property
    def has_blobs(self):
        """Whether this files has blobs in it."""
        with self.open() as f:
            return f[self.name].attrs["has_blobs"]

    def get_value(self, name, flat=False, thin=1, discard = 0):
        """Get a particular kind of entry from the backend file."""
        if not self.initialized:
            raise AttributeError("Cannot get values from unitialized storage.")

        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError("No iterations performed for this run.")

            if name == "blobs" and not g.attrs["has_blobs"]:
                return None

            v = g[name][discard + thin -1 : iteration : thin]
            if flat:
                s = list(v.shape[1:])
                s[0] = np.prod(v.shape[:2])
                return v.reshape(s)

            return v

    @property
    def size(self):
        """The length of chain."""
        with self.open() as f:
            g = f[self.name]
            return f["chain"].shape[0]

    @property
    def shape(self):
        """Tuple of (nwalkers, ndim)."""
        with self.open() as f:
            g = f[self.name]
            return g.attrs["nwalkers"], g.attrs["ndim"]

    @property
    def iteration(self):
        """The iteration the chain is currently at."""
        with self.open() as f:
            return f[self.name].attrs["iteration"]

    @property
    def accepted_array(self):
        """An array of bools representing whether parameter proposals were accepted"""
        with self.open as f:
            return f[self.name]["accepted"][...]

    @property
    def accepted(self):
        """Number of acceptances for each walker."""
        return np.sum(self.accepted_array, axis=0)

    @property
    def random_state(self):
        """The defining random state of the process."""
        with self.open() as f:
            elements = [
                v
                for k, v in sorted(f[self.name].attrs.items())
                if k.startswith("random_state_")
            ]
        return elements if len(elements) else None

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples.

        Parameters
        ----------
        ngrow : int,
            The number of steps to grow the chain.
        blobs : dict or None
            A dictionary of extra data, or None.
        """

        self._check_blobs(blobs)

        with self.open("a") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            g["chain"].resize(ntot, axis=0)
            g["log_prob"].resize(ntot, axis=0)
            g["trials"].resize(ntot, axis=0)
            g["accepted"].resize(ntot, axis=0)
            g["trial_log_prob"].resize(ntot, axis=0)


            if blobs:
                has_blobs = g.attrs["has_blobs"]
                if not has_blobs:
                    nwalkers = g.attrs["nwalkers"]
                    blobs_dtype = []
                    for k, v in blobs.items():
                        shape = np.atleast_1d(v).shape
                        if len(shape) == 1:
                            shape = shape[0]
                        blobs_dtype += [(k, (np.atleast_1d(v).dtype, shape))]

                    g.create_dataset(
                        "blobs",
                        (ntot, nwalkers),
                        maxshape=(None, nwalkers),
                        dtype=blobs_dtype,
                    )
                else:
                    g["blobs"].resize(ntot, axis=0)

                g.attrs["has_blobs"] = True

    def save_step(
            self, coords, log_prob, blobs
    ):
        """Save a step to the file.

        Parameters
        ----------
        coords : ndarray,
            The coordinates of the walkers in the ensemble.
        log_prob : ndarray,
            The log probability for each walker.
        blobs : ndarray or None
            The blobs for each walker or ``None`` if there are no blobs.
        accepted : ndarray,
            An array of boolean flags indicating whether or not the proposal for each
            walker was accepted.
        random_state :
            The current state of the random number generator.
        """
        if not hasattr(blobs, '__len__'):
            blobs = [blobs] * np.shape(coords)[0]
        self._check(coords, log_prob, blobs)

        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]

            g["chain"][iteration, :, :] = coords
            g["log_prob"][iteration, :] = log_prob

            if blobs[0]:
                blobs = np.array(
                    [tuple(b[name] for name in g["blobs".dtype.names]) for b in blobs],
                    dtype = g["blobs".dtype],
                )
                g["blobs"][iteration, ...] = blobs

            #for i, v in enumerate(random_state):
            #    g.attrs["random_state_{0}".format(i)] = v

            g.attrs["iteration"] = iteration + 1

    def _check_blobs(self, blobs):
        if self.has_blobs and not blobs:
            raise ValueError("inconsistent use of blobs")
        if self.iteration > 0 and blobs and not self.has_blobs:
            raise ValueError("inconsistent use of blobs.")

    def get_chain(self, **kwargs):
        r"""
        Get the stored chain of MCMC samples.

        Parameters
        ----------
        \*\*kwargs:
            flat(Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
        ---------
        array[..., nwalkers]:
            The chain of blobs.
        """
        return self.get_value("chain", **kwargs)

    def get_blobs(self, **kwargs):
        r"""
        Get the chain of blobs for each sample in the chain.

        Parameters
        ----------
        \*\*kwargs:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)


        Returns
        -------
        array[..., nwalkers]:
            The chain of blobs.
        """
        return self.get_value("blobs", **kwargs)

    def get_log_prob(self, **kwargs):
        """
        Get the chain of log probabilities evaluated at the MCMC samples.

        Parameters
        ----------
        kwargs:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns
        -------
        array[..., nwalkers]:
            The chain of log probabilities.
        """
        return self.get_value("log_prob", **kwargs)

    def get_trialled_log_prob(self, **kwargs):
        """
        Get the chain of log probabilities evaluated as *trials* of the MCMC.

        .. note:: these do not correspond to the chain, but instead correspond to the
                  trialled parameters. Check the :attr:`accepted` property to check if
                  each trial was accepted.

        Parameters
        ----------
        kwargs :
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns
        -------
        array[..., nwalkers]:
            The chain of log probabilities.
        """
        return self.get_value("trial_log_prob", **kwargs)

    def get_trials(self, **kwargs):
        r"""
        Get the stored chain of trials.

        Note these do not corresond to the chain, but instead correspond to the
        trialled parameters. Check the :attr:`accepted` property to check if
        each trial was accepted.

        Parameters
        ----------
        \*\*kwargs :
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns
        -------
        array[..., nwalkers, ndim]:
            The MCMC samples.
        """
        return self.get_value("trials", **kwargs)

    def get_last_sample(self):
        """
        Access the most recent sample in the chain.

        Returns
        -------
        coords : ndarray
            A list of the current positions of the walkers in the parameter space.
            The shape of this object will be ``(nwalkers, ndim)``.
        log_prob : list
            The list of log posterior probabilities for the walkers at positions given by
            ``coords``. The shape of this object is ``(nwalkers,)``.
        rstate :
            The current state of the random number generator.
        blobs : dict, optional
            The metadata ``blobs`` associated with the current position. The value is only
            returned if blobs have been saved during sampling.
        """
        if (not self.initialized) or self.iteration <=0:
            raise AttributeError(
                "you must run the sampler with "
                "`store == True` before accessing the "
                "results"
            )
        it = self.iteration
        last = [
            self.get_chain(discard = it - 1)[0],
            self.get_log_prob(discard = it - 1)[0],
            self.random_state,
        ]
        blob = self.get_blobs(discard=it-1)

        if blob is not None:
            last.append(blob[0])
        else:
            last.append(None)

        return tuple(last)

    def _check(self, coords, log_prob, blobs):
        self._check_blobs(blobs[0])
        nwalkers, ndim = self.shape

        if coords.shape != (nwalkers, ndim):
            raise ValueError(
                "invalid coordinate dimensions; expected {0}".format((nwalkers, ndim))
            )
        if log_prob.shape != (nwalkers,):
            raise ValueError(
                "invalid log probability size; expected {0}".format(nwalkers)
            )
        if blobs and len(blobs) != nwalkers:
            raise ValueError("invalid blobs size, expected {0}".format(nwalkers))
