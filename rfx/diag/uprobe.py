from __future__ import print_function

# python CLASS for working on U-Probe
__author__ = "Nicola Vianello"
__version__ = "0.2"
__data__ = "27.09.2016"

import numpy as np
import MDSplus as mds
from scipy import constants
import sys
from ..utils import bw_filter as bw, deriv1d
import matplotlib as mpl

mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rc("font", size=20)
import xarray


class Uprobe:
    """
    Python class for dealing with U-Probe. Restoring
    signal, knowing appropriate configuration and
    geometry.

    Attributes
    ----------
    ElR : float
        Absolute radial position of the Electrostatic probes
    ElZ : float
        Absolute vertical position of the Electrostatic probes
    ElP : float
        Absolute toroidal angle position of the Electrostatic
        probes
    EGrid : :obj: `dict`
        Dictionary containing the previous information with
        keys the probe name. It also add the Rrlcfs which is the
        distance from the LCFS computed using the method from
        Lionello. This is an array with dimension the time basis
        tEq which is the time basis of the of the Equilibria
    MgR : float
        Absolute radial position of the Magnetic probes
    MgZ : float
        Absolute vertical position of the Magnetic probes
    MgP : float
        Absolute toroidal angle position of the Magnetic
        probes
    MGrid : :obj: `dict`
        Dictionary containing the previous information with
        keys the probe nameIt also add the Rrlcfs which is the
        distance from the LCFS computed using the method from
        Lionello. This is an array with dimension the time basis
        tEq which is the time basis of the of the Equilibria

    Dependences
    -----------
    xarray >= 0.8
    bwfilter: Provided
    deriv1d: Provided for derivative on an irregular grid

    """

    def __init__(self, shot, **kwargs):
        """
        Init require only the shot
        Parameters
        ----------
        shot : :obj: `int`
            Shot number
        **kwargs
            trange : trange for limiting the reading of the signal
        """

        self.shot = shot
        self.trange = kwargs.get("trange", np.asarray([None, None]))
        self.tmin = self.trange[0]
        self.tmax = self.trange[1]
        # for this shot restore the gas
        try:
            rfx = mds.Tree("rfx", self.shot)
            gas = rfx.getNode(r"\rfx::v_config:vik1_gas").data()
            rfx.quit()
            if gas == "D2":
                self.Mi = 2 * constants.proton_mass
                self.alpha = 2.7
            else:
                self.Mi = constants.proton_mass
                self.alpha = 3
        except:
            print("Gas Tree not available assuming Hydrogen")
            self.Mi = constants.proton_mass
            self.alpha = 3
        if self.shot <= 35684:
            self.area = 2 * (np.pi * (1.675e-3) ** 2 + 2 * np.pi * 1.675e-3 * 1e-3)
        else:
            self.area = np.pi * (1.675e-3) ** 2 + 2 * np.pi * 1.675e-3 * 1e-3

        # restore the equilibrium rlcfs, zlcfs
        at = mds.Tree("at", self.shot)
        self.rlcfs = at.getNode(r"\s_rlcfs").data()
        self.zlcfs = at.getNode(r"\s_zlcfs").data()
        self.tEq = at.getNode(r"\s_rlcfs").getDimensionAt().data()
        # if mds.__version__:
        #     at.quit
        # else:
        #     at.quit()
        # we use two methods in order to define attributes which
        # are useful throughout the class
        self.dedg = mds.Tree("dedg", self.shot)
        self._getEl()
        self._getMag()
        self._getProbeGrid()

    def _getEl(self):
        """
        Get the name of the electrostatic signal
        for the given shot and write the appropriate
        attribute
        """
        _string = (
            r'getnci(getnci(\TOP.U_PROBE.SIGNALS.ELS,  "MEMBER_NIDS"), "NODE_NAME")'
        )
        self.sigEl = mds.Data.compile(_string).evaluate().data()
        self.sigEl = np.core.defchararray.strip(self.sigEl)
        # we then write the attribute for the available Is, Vp, and Vf
        self.sigElS = np.asarray([self.sigEl[i][:2] for i in range(self.sigEl.size)])
        # define as attribute the Floating potential, Is and Vp
        self.vF = self.sigEl[self.sigElS == "VF"]
        self.iS = self.sigEl[self.sigElS == "IS"]
        self.vP = self.sigEl[self.sigElS == "VP"]
        # this is an array of string corresponding to the towers
        self.ETower = np.asarray([self.sigEl[i][5] for i in range(self.sigEl.size)])

    def _getMag(self):
        """
        Get the name of the magnetic signals and writhe
        the appropriate attribute
        """
        _string = (
            r'getnci(getnci(\TOP.U_PROBE.SIGNALS.MAG,  "MEMBER_NIDS"), "NODE_NAME")'
        )
        self.sigMag = mds.Data.compile(_string).evaluate().data()
        self.sigMag = np.core.defchararray.strip(self.sigMag)
        # we then write the attribute for the available Is, Vp, and Vf
        self.sigMagS = [self.sigMag[i][3:5] for i in range(np.size(self.sigMag))]
        # define as attribute the Radial, Toroidal, Poloidal components
        # if np.__version__ >= 1.12:
        #     self.bR = self.sigMag[
        #         np.asarray(
        #             [
        #                 np.equal(self.sigMagS[i], "BR")
        #                 for i in range(np.size(self.sigMagS))
        #             ],
        #             dtype=bool,
        #         )
        #     ]
        #     self.bT = self.sigMag[
        #         np.asarray(
        #             [
        #                 np.equal(self.sigMagS[i], "BT")
        #                 for i in range(np.size(self.sigMagS))
        #             ],
        #             dtype=bool,
        #         )
        #     ]
        #     self.bP = self.sigMag[
        #         np.asarray(
        #             [
        #                 np.equal(self.sigMagS[i], "BP")
        #                 for i in range(np.size(self.sigMagS))
        #             ],
        #             dtype=bool,
        #         )
        #     ]
        # else:
        self.bR = self.sigMag[self.sigMagS == "BR"]
        self.bT = self.sigMag[self.sigMagS == "BT"]
        self.bP = self.sigMag[self.sigMagS == "BP"]
        # this is an array of A and B corresponding to the Towers
        self.MTower = np.asarray([self.sigMag[i][5] for i in range(self.sigMag.size)])

    def _getProbeGrid(self):
        """
        For the various signals we define the appropriate location
        in absolute values (r, z, phi) as a dictionary with the keys
        equal to the short name electrostatic and magnetics

        """
        # get the absolute position and translate into relative position.
        # This is the position of the edge of the probehead
        dummy = self.dedg.getNode(r"\insertion").data()
        self.rProbe = 2.0 + (
            0.459 - (dummy / 1e3 - 1.460)
        )  # this is the position in [m] with respect
        self.phiProbe = self.dedg.getNode(r"\u_probe_tor_angle").data()
        self.rotProbe = self.dedg.getNode(r"\rotation").data()
        # now we build the appropriate positions (R, Z, PHI)
        # R grid for all the signals
        self.ElR = np.zeros(self.sigEl.size)
        # Z grid for all the electrostatic signa
        self.ElZ = np.zeros(self.sigEl.size)
        # Poloidal positions
        self.ElP = np.zeros(self.sigEl.size)
        self.EGrid = {}
        for probe, i in zip(self.sigEl, range(self.sigEl.size)):
            self.ElR[i] = (
                self.rProbe + self.dedg.getNode("\\" + probe[3:8] + ":dr").data()
            )
            if self.rotProbe == 0:
                dummy = np.arctan2(
                    np.abs(self.dedg.getNode("\\" + probe[3:8] + ":dx").data()),
                    (2 + self.ElDr[i]),
                )
                self.ElP[i] = self.phiProbe + +np.sign(
                    self.dedg.getNode("\\" + probe[3:8] + ":dx").data()
                ) * np.degree(dummy)
            else:
                self.ElZ[i] = self.dedg.getNode("\\" + probe[3:8] + ":dx").data()

            rLcfs = self._getLcfs(self.ElR[i], self.ElZ[i])
            self.EGrid[probe] = dict(
                [
                    ("R", self.ElR[i]),
                    ("Z", self.ElZ[i]),
                    ("Phi", self.ElP[i]),
                    ("Rrlcfs", rLcfs),
                ]
            )
        # now the same for the magnetic
        self.MgR = np.zeros(self.sigMag.size)
        self.MgZ = np.zeros(self.sigMag.size)
        self.MgP = np.zeros(self.sigMag.size)
        self.MGrid = {}
        for probe, i in zip(self.sigMag, range(self.sigMag.size)):
            self.MgR[i] = (
                self.rProbe + self.dedg.getNode("\\" + probe[3:8] + ":dr").data()
            )
            if self.rotProbe == 0:
                dummy = np.arctan2(
                    np.abs(self.dedg.getNode("\\" + probe[3:8] + ":dx").data()),
                    (2 + self.ElDr[i]),
                )
                self.MgP[i] = self.phiProbe + +np.sign(
                    self.dedg.getNode("\\" + probe[3:8] + ":dx").data()
                ) * np.degree(dummy)
            else:
                self.MgZ[i] = (
                    0.459 + self.dedg.getNode("\\" + probe[3:8] + ":dx").data()
                )
            rLcfs = self._getLcfs(self.MgR[i], self.MgZ[i])
            self.MGrid[probe] = dict(
                [
                    ("R", self.MgR[i]),
                    ("Z", self.MgZ[i]),
                    ("Phi", self.MgP[i]),
                    ("Rrlcfs", rLcfs),
                ]
            )

    def _getLcfs(self, R, Z):
        """
        On the same time basis of the equilibrium compute the position of each
        of the probe location as r-rlcfs with negative if point is inside the LCFS.
        It creates an array with length the number of point in equilibrium
        and value the R-Rlcfs
        Parameters
        ----------
        R : float
            The radial position of the point considered
        Z : float
            The vertical position of the point considered

        Returns
        -------
        ndarray
            The array indicating the distance with respect to the LCFS
            computed as Euclidean distance with respect to the tangent
            to closest point of the LCFS
        """
        rDistlcfs = np.zeros(self.tEq.size)
        for i in range(self.tEq.size):
            # create Vertices of the Lcfs path
            Vert = np.asarray([self.rlcfs[:, i], self.zlcfs[:, i]]).transpose()
            prev = 0
            Lambda = np.array([])
            for k in range(len(Vert) - 1):
                svec = Vert[k + 1] - Vert[k]
                snorm = svec / np.linalg.norm(svec)
                nvec = np.asarray([-snorm[1], snorm[0]])
                dist = np.asarray((R, Z) - Vert[k])
                sigma = np.dot(dist, svec) / np.dot(svec, svec)
                if (sigma >= 0) and (sigma <= 1) or (sigma * prev < 0):
                    Lambda = np.append(Lambda, np.dot(-dist, nvec))
                prev = sigma
            rDistlcfs[i] = Lambda[np.argmin(np.abs(Lambda))]

        return rDistlcfs

    def getDensity(self, fhigh=10e3, **kwargs):
        """

        This method read the appropriate signal to
        get the evaluation of density, temperature and pressure
        from 5 pins balanced triple probe. The temperature is evaluated
        with a pass-band filter

        Parameters
        ----------
        fhigh : float, optional
            The frequency in Hz for high passing the temperature
            signal

        Returns
        -------
        triple : :obj: `dict`
            Dictionary of dictionary containing with keys corresponding to
            the short name of the found triple probe. Each subdictionary
            contains the following
                 {
                    't' : time basis,
                    'en' : electron density,
                    'te' : electron temperature,
                    'pe' : electron pressure,
                    'vp' : plasma potential,
                    'Js' : Ion saturation current density,
                    'R' : Absolute radial position of the central pin,
                    'Z' : Absolute vertical position of the central pin,
                    'Phi' : Absolute toroidal angle of the central pin,
                    'Rrlcfs' : Relative position with respect to LCFS,
                 }
        """
        self.fhigh = fhigh
        # get the appropriate rows where we have all the combination
        # of 5 pins
        _dummy5Pins = np.asarray([self.iS[i][5:7] for i in range(self.iS.size)])
        # now we build for each of the row the appropriate density and temperature
        # we build a dictionary where for each row we save the density, temperature
        # plasma potential, R, Z, Phi coordinates
        trange = kwargs.get("trange", self.trange)
        self.triple = {}
        for s in _dummy5Pins:
            t = self.dedg.getNode(r"\is_es" + s + "3").getDimensionAt().data()
            tmin, tmax = trange[0], trange[1]
            _indT = (t >= tmin) & (t <= tmax)
            self.dt = (t.max() - t.min()) / (t.size - 1)
            self.Fs = np.round(1.0 / self.dt)
            vF1 = (
                self.dedg.getNode(r"\vf_es" + s + "1").data()[_indT]
                - self.dedg.getNode(r"\vf_es" + s + "1").data()[(t < 0)].mean()
            )
            vF5 = (
                self.dedg.getNode(r"\vf_es" + s + "5").data()[_indT]
                - self.dedg.getNode(r"\vf_es" + s + "5").data()[(t < 0)].mean()
            )
            vF = 0.5 * (vF1 + vF5)
            iS = (
                self.dedg.getNode(r"\is_es" + s + "3").data()[_indT]
                - self.dedg.getNode(r"\is_es" + s + "3").data()[(t < 0)].mean()
            )
            vP = (
                self.dedg.getNode(r"\vp_es" + s + "2").data()[_indT]
                - self.dedg.getNode(r"\vp_es" + s + "2").data()[(t < 0)].mean()
            )
            te = bw.bw_filter(
                (vP - vF) / np.log(3), self.fhigh, self.Fs, "lowpass"
            )  # electron temperature filtered
            te[(te <= 5)] = 5
            cs = np.sqrt(2 * constants.e * te / (self.Mi))  # ion sound speed
            en = 2 * (iS) / (self.area * cs * constants.e * 1e19)
            vPl = vP + self.alpha * te
            # compute also the plasma pressure
            pe = constants.elementary_charge * te * en * 1e19
            t = t[_indT]
            self.triple[s] = dict(
                [
                    ("ne", en),
                    ("te", te),
                    ("vp", vPl),
                    ("pe", pe),
                    ("t", t),
                    ("r", self.ElR[np.where(self.sigEl == r"IS_ES" + s + "3")][0]),
                    ("z", self.ElZ[np.where(self.sigEl == r"IS_ES" + s + "3")][0]),
                    ("phi", self.ElP[np.where(self.sigEl == r"IS_ES" + s + "3")][0]),
                    ("Rrlcfs", self.EGrid["IS_ES" + s + "3"]["Rrlcfs"]),
                    ("Js", iS / (self.area)),
                    ("t", t),
                ]
            )

    def getFloating(self, **kwargs):
        """

        Get the floating potential and save them in a dictionary
        including their Radial and Vertical Position. Once run the
        floating is an Attribute of the class

        """
        trange = kwargs.get("trange", self.trange)
        tmin, tmax = trange[0], trange[1]
        dummy = []
        for probe in self.vF:
            t = self.dedg.getNode("\\" + probe).getDimensionAt().data()
            _indT = (t >= tmin) & (t <= tmax)
            sig = (
                self.dedg.getNode("\\" + probe).data()[_indT]
                - self.dedg.getNode("\\" + probe).data()[(t < 0)].mean()
            )
            t = t[_indT]
            _dummyX = xarray.DataArray(sig, coords=[t], dims=["time"])
            dummy.append(_dummyX)

        self.vFArr = xarray.concat(dummy, dim="Probe")
        self.vFArr["Probe"] = self.vF
        self.vFArr.attrs["R"] = np.asarray([self.EGrid[k]["R"] for k in self.vF])
        self.vFArr.attrs["Z"] = np.asarray([self.EGrid[k]["Z"] for k in self.vF])
        self.vFArr.attrs["Phi"] = np.asarray([self.EGrid[k]["Phi"] for k in self.vF])

    def FloatingProfile(self, **kwargs):
        """
        Provide a list f DataArray each containing the profiles on different rows
        """
        # check the the floating potential has been properly stored
        try:
            self.vFArr
            trange = kwargs.get(
                "trange", [self.vFArr.time.min().item(), self.vFArr.time.max().item()]
            )
            _dummy = self.vFArr.where(
                ((self.vFArr.time >= trange[0]) & (self.vFArr.time <= trange[1])),
                drop=True,
            )
        except:
            self.getFloating(**kwargs)
            trange = self.vFArr.time.min().item(), self.vFArr.time.max().item()
        # find the corresponding point in time where the equilibrium is within
        # the limit
        _idxEq = (self.tEq >= trange[0]) & (self.tEq <= trange[1])
        # we must select the floating potential according to Z and
        # average over time. We loop and create an appropriate
        # xarray to be used afterwards as groupby
        _dummyP = []
        for i, z in enumerate(np.unique(self.vFArr.Z)):
            a = _dummy[(self.vFArr.Z == z), :].mean(dim="time")
            e = _dummy[(self.vFArr.Z == z), :].std(dim="time")
            r = np.asarray(
                [
                    self.EGrid[p]["Rrlcfs"][_idxEq].mean()
                    for p in self.vFArr[(self.vFArr.Z == z), :].Probe.values
                ]
            )

            _dummyP.append(
                xarray.DataArray(
                    a.values,
                    coords=[r],
                    dims=["r"],
                    attrs={"err": e.values, "Z": z, "trange": trange},
                )
            )
        self.vFProfile = _dummyP
        return self.vFProfile.copy()

    def VfProfilePlot(self, axes=None, **kwargs):
        """
        Plot the profile once it has been calculated taking care of the possible NaNs

        """
        try:
            self.vFProfile
        except:
            _dummy = self.FloatingProfile(**kwargs)
        if axes is None:
            fig, axes = mpl.pylab.subplots(figsize=(7, 6), nrows=1, ncols=1)
            fig.subplots_adjust(bottom=0.17, left=0.17)

        aggregate = kwargs.get("aggregate", True)
        if aggregate is False:
            for k in range(len(self.vFProfile)):
                x = self.vFProfile[k].r
                y = self.vFProfile[k].values
                e = self.vFProfile[k].err / 2
                (l,) = axes.plot(
                    x[~np.isnan(y)],
                    y[~np.isnan(y)],
                    "o--",
                    markersize=10,
                    label="Z = %3.2f" % self.vFProfile[k].Z,
                )
                axes.fill_between(
                    x[~np.isnan(y)],
                    y[~np.isnan(y)] - e,
                    y[~np.isnan(y)] + e,
                    color=l.get_mfc(),
                    edgecolor="white",
                    alpha=0.5,
                )

            axes.set_xlabel(r"R [m]")
            axes.set_ylabel(r"V$_f$ [V]")
            leg = axes.legend(loc="best", prop={"size": 20}, frameon=False, numpoints=1)
        else:
            xAll = np.array([])
            yAll = np.array([])
            eAll = np.array([])
            for k in range(len(self.vFProfile)):
                xAll = np.append(xAll, self.vFProfile[k].r)
                yAll = np.append(yAll, self.vFProfile[k].values)
                eAll = np.append(eAll, self.vFProfile[k].err / 2)
            axes.errorbar(
                xAll[~np.isnan(yAll)],
                yAll[~np.isnan(yAll)],
                yerr=eAll[~np.isnan(yAll)],
                fmt="o",
                ms=12,
                **kwargs
            )
            axes.set_xlabel(r"R [m]")
            axes.set_ylabel(r"V$_f$ [V]")

    def TripleProfile(self, **kwargs):
        """
        Compute the profile as a function of quantities included in the
        triple, which means ne, Te, vP providing the radius in Distance
        from the separatrix.

        Keyword:
        --------
        trange: to provide the range where this profile should be considered
        keys : In case we would like to have profile limiting to some of the available
               triples
        """
        self.fhigh = kwargs.get("fhigh", 10e3)
        try:
            self.triple
        except:
            self.getDensity(fhigh=self.fhigh)

        keys = kwargs.get("keys", self.triple.keys())

        # now determine the ranges
        trange = kwargs.get("trange", [self.t.min(), self.t.max()])
        EnP = np.zeros(len(keys))
        EnE = np.zeros(len(keys))
        TeP = np.zeros(len(keys))
        TeE = np.zeros(len(keys))
        PeP = np.zeros(len(keys))
        PeE = np.zeros(len(keys))
        VpP = np.zeros(len(keys))
        VpE = np.zeros(len(keys))
        JsP = np.zeros(len(keys))
        JsE = np.zeros(len(keys))
        rP = np.zeros(len(keys))
        _idx = (self.t >= trange[0]) & (self.t <= trange[1])
        _idxE = (self.tEq >= trange[0]) & (self.tEq <= trange[1])
        for k, i in zip(keys, range(len(keys))):
            EnP[i] = self.triple[k]["ne"][_idx].mean()
            EnE[i] = self.triple[k]["ne"][_idx].std()
            TeP[i] = self.triple[k]["te"][_idx].mean()
            TeE[i] = self.triple[k]["te"][_idx].std()
            PeP[i] = self.triple[k]["pe"][_idx].mean()
            PeE[i] = self.triple[k]["pe"][_idx].std()
            VpP[i] = self.triple[k]["vp"][_idx].mean()
            VpE[i] = self.triple[k]["vp"][_idx].std()
            JsP[i] = self.triple[k]["Js"][_idx].mean()
            JsE[i] = self.triple[k]["Js"][_idx].std()
            rP[i] = self.triple[k]["Rrlcfs"][_idxE].mean()

        # reorder accoring to rP
        EnP = EnP[np.argsort(rP)]
        EnE = EnE[np.argsort(rP)]
        TeP = TeP[np.argsort(rP)]
        TeE = TeP[np.argsort(rP)]
        VpP = VpP[np.argsort(rP)]
        VpE = VpP[np.argsort(rP)]
        PeP = PeP[np.argsort(rP)]
        PeE = PeP[np.argsort(rP)]
        JsP = JsP[np.argsort(rP)]
        JsE = JsP[np.argsort(rP)]
        rP = rP[np.argsort(rP)]

        out = {
            "r": rP,
            "Dens": EnP,
            "DensErr": EnE,
            "Temp": TeP,
            "TempErr": TeE,
            "Pe": PeP,
            "PeE": PeE,
            "Vp": VpP,
            "VpE": VpE,
            "Js": JsP,
            "JsE": JsE,
        }
        return out

    def _getExB(self, floating=False, **kwargs):
        """
        Get the ExB velocity using plasma potential on probe on
        same triple probe arrays

        """
        if floating is False:
            # first of all check if the triple dictionary is define
            try:
                self.triple
            except:
                self._getDensity()
                # now we distinguish the towers A and B among the triples
                _dummy = np.asarray([k[:1] for k in self.triple.keys()])
                _Idx = np.arange(len(self.triple.keys()))
                self.ExB = {}
                for tow in np.unique(_dummy):
                    # check the number of signals in each of the tower
                    ns = np.count_nonzero(_dummy == tow)
                    r = np.zeros(ns)
                    sig = np.vstack(
                        [
                            self.triple[self.triple.keys()[key]]["vp"]
                            for key in _Idx[_dummy == tow]
                        ]
                    )
                    r = np.asarray(
                        [
                            self.triple[self.triple.keys()[key]]["r"]
                            for key in _Idx[_dummy == tow]
                        ]
                    )
                    # now provide the gradient for each of the time point
                    sig = sig[np.argsort(r), :]
                    r = r[np.argsort(r)]
                    _d = -np.asarray(
                        [deriv1d.deriv(r, sig[:, i]) for i in range(self.nsample)]
                    )
                    self.ExB[tow] = dict([("vel", _d), ("r", r), ("t", self.t)])
        else:
            # we must read the floating potential signals
            try:
                self.vFArr
            except:
                self._getFloating(trange=kwargs.get("trange", self.trange))
            # now we aggregate them along the average r direction

    def getExBprofile(self, **kwargs):
        """
        Compute the average ExB profile based on spline interpolation of Floating
        potential profile and interpolation of Te profile
        Args:
            **kwargs: All the keyword which can be passed to getFloating e getTriple in
            particular the trange
        Returns:
            A tuple with the (r,ExB and corresponding error)

        Returns:

        """
        pass

    def getVfMap(self, **kwargs):
        """
        small method to get the VfMap on a rectangular grip with linear interpolation
        for the missing points. Useful for further spline or map
        Args:
            **kwargs: eventually a time range can be directly passed

        Returns:
            Map in (#time,#R,#Z) with corresponding time, r, z
        """

        trange = kwargs.get("trange", self.trange)
        self.getFloating(trange=trange)
        R, Z = np.sort(np.unique(self.vFArr.R)), np.sort(np.unique(self.vFArr.Z))
        M = np.zeros((self.vFArr.time.size, R.size, Z.size))
        for p in self.vFArr.Probe.values:
            _rr, _zz = self.EGrid[p]["R"], self.EGrid[p]["Z"]
            i, j = np.where(R == _rr)[0][0], np.where(Z == _zz)[0][0]
            M[:, i, j] = self.vFArr.sel(Probe=p).values
        # now sanity check for identically 0 values to be substitute with the mean
        for i in range(R.size):
            for j in range(Z.size):
                if np.mean(M[:, i, j]) == 0:
                    M[:, i, j] = (M[:, i, j - 1] + M[:, i, j + 1]) / 2.0

        return M, self.vFArr.time.values, R, Z
