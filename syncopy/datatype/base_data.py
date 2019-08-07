# -*- coding: utf-8 -*-
# 
# Syncopy BaseData abstract class + helper classes
# 
# Created: 2019-01-07 09:22:33
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-08-07 11:06:46>

# Builtin/3rd party package imports
import numpy as np
import getpass
import socket
import time
import sys
import os
import numbers
import inspect
import h5py
import scipy as sp
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from copy import copy
from datetime import datetime
from hashlib import blake2b
from itertools import islice
from numpy.lib.format import open_memmap, read_magic
import shutil

# Local imports
from .data_methods import definetrial
from syncopy.shared.parsers import scalar_parser, array_parser, io_parser, filename_parser
from syncopy.shared.errors import SPYTypeError, SPYValueError, SPYError
from syncopy import __version__, __storage__, __dask__, __sessionid__
if __dask__:
    import dask
import syncopy as spy

__all__ = ["StructDict"]


class BaseData(ABC):

    # Class properties that are written to JSON/HDF upon save
    _infoFileProperties = ("dimord", "_version", "_log", "cfg",)
    _hdfFileProperties =  ("dimord", "_version", "_log",)

    # Checksum algorithm used
    _checksum_algorithm = spy.__checksum_algorithm__.__name__
    
    # Dummy allocations of class attributes that are actually initialized in subclasses
    _mode = None
    
    @property
    def cfg(self):
        """Dictionary of previous operations on data"""
        return self._cfg

    @cfg.setter
    def cfg(self, dct):
        if not isinstance(dct, dict):
            raise SPYTypeError(dct, varname="cfg", expected="dictionary-like object")
        self._cfg = self._set_cfg(self._cfg, dct)
        
    @property
    def container(self):
        if self.data is not None:
            return filename_parser(self.filename)["container"]
    
    @property
    def data(self):
        """array-like object representing data without trials"""

        if getattr(self._data, "id", None) is not None:
            if self._data.id.valid == 0:
                lgl = "open HDF5 container"
                act = "backing HDF5 container {} has been closed"
                raise SPYValueError(legal=lgl, actual=act.format(self.filename),
                                    varname="data")
        return self._data
    
    @data.setter
    def data(self, in_data):

        # Dimension count is either determined by length of dimord or 2 in case
        # of `EventData` or `SpikeData`
        if any(["DiscreteData" in str(base) for base in self.__class__.__mro__]):
            ndim = 2
        else:
            ndim = len(self.dimord)
                
        # If input is a string, try to load memmap/HDF5 dataset
        if isinstance(in_data, str):
            try:
                fpath, fname = io_parser(in_data, varname="filename", isfile=True, exists=True)
            except Exception as exc:
                raise exc
            in_data = os.path.join(fpath, fname)  # ensure `in_data` is absolute path

            md = self.mode
            if md == "w":
                md = "r+"

            is_npy = False
            is_hdf = False
            try:
                with open(in_data, "rb") as fd:
                    read_magic(fd)
                is_npy = True
            except ValueError as exc:
                err = "NumPy memorymap: " + str(exc)
            try:
                h5f = h5py.File(in_data, mode=md)
                is_hdf = True
            except OSError as exc:
                err = "HDF5: " + str(exc)
            if not is_npy and not is_hdf:
                raise SPYValueError("accessible HDF5 container or memory-mapped npy-file",
                                    actual=err, varname="data")
            
            if is_hdf:
                h5keys = list(h5f.keys())
                idx = [h5keys.count(dclass) for dclass in spy.datatype.__all__ \
                       if not (inspect.isfunction(getattr(spy.datatype, dclass)))]
                if len(h5keys) !=1 and sum(idx) != 1:
                    lgl = "HDF5 container holding one data-object"
                    act = "HDF5 container holding {} data-objects"
                    raise SPYValueError(legal=lgl, actual=act.format(str(len(h5keys))), varname="data")
                if len(h5keys) == 1:
                    self._data = h5f[h5keys[0]]
                else:
                    self._data = h5f[spy.datatype.__all__[idx.index(1)]]
            if is_npy:
                self._data = open_memmap(in_data, mode=md)
            self.filename = in_data

        # If input is already a memmap/HDF5 dataset, check its dimensions
        elif isinstance(in_data, (np.memmap, h5py.Dataset)):
            if isinstance(in_data, h5py.Dataset):
                if in_data.id.valid == 0:
                    lgl = "open HDF5 container"
                    act = "backing HDF5 container is closed"
                    raise SPYValueError(legal=lgl, actual=act, varname="data")
                md = in_data.file.mode
                fn = in_data.file.filename
            else:
                md = in_data.mode
                fn = in_data.filename
            if in_data.ndim != ndim:
                lgl = "{}-dimensional data".format(ndim)
                act = "{}-dimensional HDF5 dataset or memmap".format(in_data.ndim)
                raise SPYValueError(legal=lgl, varname="data", actual=act)
            self.mode = md
            self.filename = os.path.abspath(fn)
            self._data = in_data
            
        # If input is an array, either fill existing data property
        # or create backing container on disk
        elif isinstance(in_data, np.ndarray):
            try:
                array_parser(in_data, varname="data", dims=ndim)
            except Exception as exc:
                raise exc
            if isinstance(self._data, (np.memmap, h5py.Dataset)):
                if self.mode == "r":
                    lgl = "HDF5 dataset/memmap with write or copy-on-write access"
                    act = "read-only memmap"
                    raise SPYValueError(legal=lgl, varname="mode", actual=act)
                if self.data.shape != in_data.shape:
                    lgl = "HDF5 dataset/memmap with shape {}".format(str(self.data.shape))
                    act = "data with shape {}".format(str(in_data.shape))
                    raise SPYValueError(legal=lgl, varname="data", actual=act)
                if self.data.dtype != in_data.dtype:
                    print("SyNCoPy core - data: WARNING >> Input data-type mismatch << ")
                self._data[...] = in_data
            else:
                self.filename = self._gen_filename()
                dsetname = self.__class__.__name__
                with h5py.File(self.filename, "w") as h5f:
                    h5f.create_dataset(dsetname, data=in_data)
                md = self.mode
                if md == "w":
                    md = "r+"
                self._data = h5py.File(self.filename, md)[dsetname]

        # If input is a `VirtualData` object, make sure the object class makes sense
        elif isinstance(in_data, VirtualData):
            if self.__class__.__name__ != "AnalogData":
                lgl = "(filename of) memmap or NumPy array"
                act = "VirtualData (only valid for `AnalogData` objects)"
                raise SPYValueError(legal=lgl, varname="data", actual=act)
            self._data = in_data
            self._filename = [dat.filename for dat in in_data._data]
            self.mode = "r"

        # Whatever type the input is, it's not supported
        else:
            msg = "(filename of) memmap, NumPy array or VirtualData object"
            raise SPYTypeError(in_data, varname="data", expected=msg)

        # In case we're working with a `DiscreteData` object, fill up samples
        if any(["DiscreteData" in str(base) for base in self.__class__.__mro__]):
            self._sample = np.unique(self.data[:,self.dimord.index("sample")])

        # In case we're working with an `AnalogData` object, tentatively fill up channel labels
        if any(["ContinuousData" in str(base) for base in self.__class__.__mro__]):
            channel = ["channel" + str(i + 1) for i in range(self.data.shape[self.dimord.index("channel")])]
            self.channel = np.array(channel)

        # In case we're working with an `EventData` object, fill up eventid's
        if self.__class__.__name__ == "EventData":
            self._eventid = np.unique(self.data[:,self.dimord.index("eventid")])

    @property
    def dimord(self):
        """list(str): ordered list of data dimension labels"""
        return self._dimord
    
    @dimord.setter
    def dimord(self, dims):
        if hasattr(self, "_dimord"):
            print("Syncopy core - dimord: Cannot change `dimord` of object. " +\
                  "Functionality currently not supported")
        # Canonical way to perform initial allocation of dimensional properties 
        # (`self._channel = None`, `self._freq = None` etc.)            
        self._dimord = list(dims)
        if dims.count("channel") == 2:  # account for `ConnectivityData` objects
            dims.remove("channel")
            self._channel1 = None
            dims.remove("channel")
            self._channel2 = None
        for dim in [dlabel for dlabel in dims if dlabel != "time"]:
            setattr(self, "_" + dim, None)
            
    @property
    def filename(self):
        # implicit support for multiple backing filenames: convert list to str
        if isinstance(self._filename, list):
            outname = "".join(fname + ", " for fname in self._filename)[:-2]
        else:
            outname = self._filename
        return outname
    
    @filename.setter
    def filename(self, fname):
        if not isinstance(fname, str):
            raise SPYTypeError(fname, varname="fname", expected="str")
        self._filename = str(fname)

    @property
    def log(self):
        """str: log of previous operations on data"""
        print(self._log_header + self._log)

    @log.setter
    def log(self, msg):
        if not isinstance(msg, str):
            raise SPYTypeError(msg, varname="log", expected="str")
        prefix = "\n\n|=== {user:s}@{host:s}: {time:s} ===|\n\n\t{caller:s}"
        clr = sys._getframe().f_back.f_code.co_name
        if clr.startswith("_") and not clr.startswith("__"):
            clr = clr[1:]
        self._log += prefix.format(user=getpass.getuser(),
                                   host=socket.gethostname(),
                                   time=time.asctime(),
                                   caller=clr + ": " if clr != "<module>" else "")\
            + msg

    @property
    def mode(self):
        """str: write mode for data, 'r' for read-only, 'w' for writable

        FIXME: append/replace with HDF5?
        """
        return self._mode
    
    @property
    def tag(self):
        if self.data is not None:
            return filename_parser(self.filename)["tag"]

    @mode.setter
    def mode(self, md):

        # Ensure input makes sense and we actually have permission to change
        # the data access mode
        if not isinstance(md, str):
            raise SPYTypeError(md, varname="mode", expected="str")
        options = ["r", "r+", "w", "c"]
        if md not in options:
            lgl = "'" + "or '".join(opt + "' " for opt in options)
            raise SPYValueError(lgl, varname="mode", actual=md)
        if isinstance(self.data, VirtualData):
            print("syncopy core - mode: WARNING >> Cannot change read-only " +
                  "access mode of VirtualData datasets << ")
            return

        # If data is already attached to the object, change its access mode
        # as requested (if `md` is actually any different from `self.mode`)
        # NOTE: prevent accidental data loss by not allowing mode = "w" in h5py
        if self.data is not None:
            if md == self._mode:
                return
            if md == "w":
                md = "r+"
            self.data.flush()
            if isinstance(self.data, np.memmap):
                self._data = None
                self._data = open_memmap(self.filename, mode=md)
            else:
                dsetname = self.data.name
                self._data.file.close()
                self._data = h5py.File(self.filename, mode=md)[dsetname]

        self._mode = md

    @property
    def sampleinfo(self):
        """nTrials x 3 :class:`numpy.ndarray` of [start, end, offset] sample indices"""
        return self._sampleinfo

    @sampleinfo.setter
    def sampleinfo(self, sinfo):
        if self.data is None:
            print("SyNCoPy core - sampleinfo: Cannot assign `sampleinfo` without data. "+\
                  "Please assing data first")
            return
        if any(["ContinuousData" in str(base) for base in self.__class__.__mro__]):
            scount = self.data.shape[self.dimord.index("time")]
        else:
            scount = np.inf
        try:
            array_parser(sinfo, varname="sampleinfo", dims=(None, 2), hasnan=False, 
                         hasinf=False, ntype="int_like", lims=[0, scount])
        except Exception as exc:
            raise exc
        self._sampleinfo = np.array(sinfo, dtype=int)

    @property
    def t0(self):
        """FIXME: should be hidden"""
        return self._t0

    @property
    def trials(self):
        """list-like array of trials"""
        return Indexer(map(self._get_trial, range(self.trialinfo.shape[0])),
                       self.trialinfo.shape[0]) if self.trialinfo is not None else None
        
    @property
    def trialinfo(self):
        """nTrials x M :class:`numpy.ndarray` with numeric information about each trial

        Each trial can have M properties (condition, original trial no., ...) coded by 
        """
        return self._trialinfo

    @trialinfo.setter
    def trialinfo(self, trl):
        if self.data is None:
            print("SyNCoPy core - trialinfo: Cannot assign `trialinfo` without data. "+\
                  "Please assing data first")
            return
        try:
            array_parser(trl, varname="trialinfo", dims=(self.sampleinfo.shape[0], None))
        except Exception as exc:
            raise exc
        self._trialinfo = np.array(trl)


    # Selector method
    @abstractmethod
    def selectdata(self, trials=None, deepcopy=False, **kwargs):
        """
        Docstring mostly pointing to ``selectdata``
        """
        pass

    # Helper function that grabs a single trial
    @abstractmethod
    def _get_trial(self, trialno):
        pass

    # Convenience function, wiping contents of backing device from memory
    def clear(self):
        """Clear loaded data from memory

        Calls `flush` method of HDF5 dataset or memory map. Memory maps are
        deleted and re-instantiated.        

        """
        if self.data is not None:
            self.data.flush()
            if isinstance(self.data, np.memmap):
                filename, mode = self.data.filename, self.data.mode
                self._data = None
                self._data = open_memmap(filename, mode=mode)
        return

    # Return a (deep) copy of the current class instance
    def copy(self, deep=False):
        """Create a copy of the data object in memory.

        Parameters
        ----------
            deep : bool
                If `True`, a copy of the underlying data file is created in the temporary Syncopy folder

        Returns
        -------
            BaseData
                in-memory copy of BaseData object

        See also
        --------
        save_spy

        """
        cpy = copy(self)
        if deep and isinstance(self.data, (np.memmap, h5py.Dataset)):
            self.data.flush()
            filename = self._gen_filename()
            shutil.copyfile(self.filename, filename)
            cpy.data = filename
        return cpy

    # Change trialdef of object
    def definetrial(self, trl=None, pre=None, post=None, start=None,
                    trigger=None, stop=None, clip_edges=False):
        """(Re-)define trials for data

        See also
        --------
        syncopy.definetrial

        """
        definetrial(self, trialdefinition=trl, pre=pre, post=post,
                    start=start, trigger=trigger, stop=stop,
                    clip_edges=clip_edges)


    # Wrapper that makes saving routine usable as class method
    def save(self, container=None, tag=None, filename=None, overwrite=False, memuse=100):
        """Save data object as new ``spy`` HDF container to disk (:func:`syncopy.save_data`)
        
        FIXME: update docu
        
        Parameters
        ----------                    
            container : str
                Path to Syncopy container folder (*.spy) to be used for saving. If 
                omitted, a .spy extension will be added to the folder name.
            tag : str
                Tag to be appended to container basename
            filename :  str
                Explicit path to data file. This is only necessary if the data should
                not be part of a container folder. An extension (*.<dataclass>) will
                be added if omitted. The `tag` argument is ignored.      
            overwrite : bool
                If `True` an existing HDF5 file and its accompanying JSON file is 
                overwritten (without prompt). 
            memuse : scalar 
                 Approximate in-memory cache size (in MB) for writing data to disk
                 (only relevant for :class:`VirtualData` or memory map data sources)

        Examples
        --------    
        >>> save_spy(obj, filename="session1")
        >>> # --> os.getcwd()/session1.<dataclass>
        >>> # --> os.getcwd()/session1.<dataclass>.info

        >>> save_spy(obj, filename="/tmp/session1")
        >>> # --> /tmp/session1.<dataclass>
        >>> # --> /tmp/session1.<dataclass>.info

        >>> save_spy(obj, container="container.spy")
        >>> # --> os.getcwd()/container.spy/container.<dataclass>
        >>> # --> os.getcwd()/container.spy/container.<dataclass>.info

        >>> save_spy(obj, container="/tmp/container.spy")
        >>> # --> /tmp/container.spy/container.<dataclass>
        >>> # --> /tmp/container.spy/container.<dataclass>.info

        >>> save_spy(obj, container="session1.spy", tag="someTag")
        >>> # --> os.getcwd()/container.spy/session1_someTag.<dataclass>
        >>> # --> os.getcwd()/container.spy/session1_someTag.<dataclass>.info

        """
        
        # Ensure `obj.save()` simply overwrites on-disk representation of object
        if container is None and tag is None and filename is None:
            if self.container is None:
                raise SPYError("Cannot create spy container in temporary " +\
                               "storage {} - please provide explicit path. ".format(__storage__))
            overwrite = True
            filename = self.filename
            
        # Support `obj.save(tag="newtag")`            
        if container is None and filename is None:
            if self.container is None:
                raise SPYError("Object is not associated to an existing spy container - " +\
                               "please save object first using an explicit path. ")
            container = filename_parser(self.filename)["folder"]
            
        spy.save(self, filename=filename, container=container, tag=tag, 
                 overwrite=overwrite, memuse=memuse)

    # Helper function generating pseudo-random temp file-names    
    def _gen_filename(self):
        fname_hsh = blake2b(digest_size=4, 
                            salt=os.urandom(blake2b.SALT_SIZE)).hexdigest()
        return os.path.join(__storage__,
                            "spy_{sess:s}_{hash:s}{ext:s}".format(
                                sess=__sessionid__, hash=fname_hsh,
                                ext=self._classname_to_extension()))

    def _classname_to_extension(self):
        return "." + self.__class__.__name__.split('Data')[0].lower()

    # Helper function that digs into cfg dictionaries
    def _set_cfg(self, cfg, dct):
        dct = StructDict(dct)
        if not cfg:
            cfg = dct
        else:
            if "cfg" in cfg.keys():
                self._set_cfg(cfg["cfg"], dct)
            else:
                cfg["cfg"] = dct
                return cfg
        return cfg

    # Legacy support
    def __repr__(self):
        return self.__str__()

    # Make class contents readable from the command line
    def __str__(self):

        # Get list of print-worthy attributes
        ppattrs = [attr for attr in self.__dir__()
                   if not (attr.startswith("_") or attr in ["log", "t0"])]
        ppattrs = [attr for attr in ppattrs
                   if not (inspect.ismethod(getattr(self, attr))
                           or isinstance(getattr(self, attr), Iterator))]
        if hasattr(self, "hdr"):
            if getattr(self, "hdr") is None:
                ppattrs.pop(ppattrs.index("hdr"))
        ppattrs.sort()

        # Construct string for pretty-printing class attributes
        if self.__class__.__name__ == "SpikeData":
            dinfo = " 'spike' x "
            dsep = "'-'"
        elif self.__class__.__name__ == "EventData":
            dinfo = " 'event' x "
            dsep = "'-'"
        else:
            dinfo = ""
            dsep = "' x '"
        hdstr = "SyNCoPy{diminfo:s}{clname:s} object with fields\n\n"
        ppstr = hdstr.format(diminfo=dinfo + " '"  + \
                             dsep.join(dim for dim in self.dimord) + "' ",
                             clname=self.__class__.__name__)
        maxKeyLength = max([len(k) for k in ppattrs])
        for attr in ppattrs:
            value = getattr(self, attr)
            if hasattr(value, 'shape') and attr == "data" and self.sampleinfo is not None:
                tlen = np.unique([sinfo[1] - sinfo[0] for sinfo in self.sampleinfo])
                if tlen.size == 1:
                    trlstr = "of length {} ".format(str(tlen[0]))
                else:
                    trlstr = ""
                dsize = np.prod(self.data.shape)*self.data.dtype.itemsize/1024**2
                dunit = "MB"
                if dsize > 1000:
                    dsize /= 1024
                    dunit = "GB"
                valueString = "{} trials {}defined on ".format(str(len(self.trials)), trlstr)
                valueString += "[" + " x ".join([str(numel) for numel in value.shape]) \
                              + "] {dt:s} {tp:s} " +\
                              "of size {sz:3.2f} {szu:s}"
                valueString = valueString.format(dt=self.data.dtype.name,
                                                 tp=self.data.__class__.__name__,
                                                 sz=dsize,
                                                 szu=dunit)
            elif hasattr(value, 'shape'):
                valueString = "[" + " x ".join([str(numel) for numel in value.shape]) \
                              + "] element " + str(type(value))
            elif isinstance(value, list):
                valueString = "{0} element list".format(len(value))
            elif isinstance(value, dict):
                msg = "dictionary with {nk:s}keys{ks:s}"
                keylist = value.keys()
                showkeys = len(keylist) < 7
                valueString = msg.format(nk=str(len(keylist)) + " " if not showkeys else "",
                                         ks=" '" + "', '".join(key for key in keylist) + "'" if showkeys else "")
            else:
                valueString = str(value)
            printString = "{0:>" + str(maxKeyLength + 5) + "} : {1:}\n"
            ppstr += printString.format(attr, valueString)
        ppstr += "\nUse `.log` to see object history"
        return ppstr

    # Destructor
    def __del__(self):
        if self.filename is not None:
            if isinstance(self._data, h5py.Dataset):
                try:
                    self._data.file.close()
                except:
                    pass
            else:
                del self._data
            if __storage__ in self.filename and os.path.exists(self.filename):
                os.unlink(self.filename)
                shutil.rmtree(os.path.splitext(self.filename)[0],
                              ignore_errors=True)

    # Class "constructor"
    def __init__(self, data=None, filename=None, dimord=None, mode="r+", **kwargs):
        """
        Docstring

        filename + data = create memmap @filename
        filename no data = read from file or memmap
        just data = try to attach data (error checking done by data.setter)
        """

        # First things first: initialize (dummy) default values
        self._cfg = {}
        self._data = None
        self.mode = mode
        self._sampleinfo = None
        self._t0 = [None]
        self._trialinfo = None
        self._filename = None
        
        # Set up dimensional architecture (`self._channel = None`, `self._freq = None` etc.)
        self.dimord = dimord

        # Depending on contents of `filename` and `data` class instantiation invokes I/O routines
        if filename is not None:

            # Case 1: filename + data = memmap @filename
            if data is not None:
                read_fl = False
                self.data = filename
                self.data = data

            # Case 2: filename w/o data = read from file/container
            else:
                read_fl = False
                try:
                    fileinfo = filename_parser(filename)
                    if fileinfo["filename"] is not None:
                        read_fl = True
                except:
                    pass
                if not read_fl:
                    self.data = filename
                    
        else:

            # Case 3: just data = if str, it HAS to be the name of a spy-file
            if data is not None:
                if isinstance(data, str):
                    try:
                        fileinfo = filename_parser(data)
                    except Exception as exc:
                        raise exc
                    if fileinfo["filename"] is None:
                        lgl = "explicit file-name to initialize object"
                        raise SPYValueError(legal=lgl, actual=data)
                    read_fl = True
                    filename = data
                else:
                    read_fl = False
                    self.data = data

            # Case 4: nothing here: create empty object
            else:
                read_fl = False
                self._filename = self._gen_filename()
        
        # Warn on effectless assignments
        if read_fl:
            msg = "Syncopy core - __init__: WARNING >> Cannot assign `{}` to object " +\
                  "loaded from spy container << "                
            for key, value in kwargs.items():
                if value is not None:
                    print(msg.format(key))
            
        # Prepare log + header and write first entry
        lhd = "\n\t\t>>> SyNCopy v. {ver:s} <<< \n\n" +\
              "Created: {timestamp:s} \n\n" +\
              "System Profile: \n" +\
              "{sysver:s} \n" +\
              "NumPy: {npver:s}\n" +\
              "SciPy: {spver:s}\n" +\
              "Dask:  {daver:s}\n\n" +\
              "--- LOG ---"
        self._log_header = lhd.format(ver=__version__,
                                      timestamp=time.asctime(),
                                      sysver=sys.version,
                                      npver=np.__version__,
                                      spver=sp.__version__,
                                      daver=dask.__version__ if __dask__ else "--")
        self._log = ""
        self.log = "created {clname:s} object".format(clname=self.__class__.__name__)

        # Write version
        self._version = __version__

        # Finally call spy loader if filename was provided
        if read_fl:
            spy.load(filename=filename, out=self)

        # Make instantiation persistent in all subclasses
        super().__init__()

        
class VirtualData():
    """Class for handling 2D-data spread across multiple files

    Arrays from individual files (chunks) are concatenated along 
    the 2nd dimension (dim=1).

    """

    # Pre-allocate slots here - this class is *not* meant to be expanded
    # and/or monkey-patched at runtime
    __slots__ = ["_M", "_N", "_shape", "_size", "_ncols", "_data", "_cols", "_dtype"]

    @property
    def dtype(self):
        return self._dtype

    @property
    def M(self):
        return self._M

    @property
    def N(self):
        return self._N

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    # Class instantiation
    def __init__(self, chunk_list):
        """
        Docstring coming soon...

        Do not confuse chunks with trials: chunks refer to actual raw binary
        data-files on disk, thus, row- *and* col-numbers MUST match!
        """

        # First, make sure our one mandatory input argument does not contain
        # any unpleasant surprises
        if not isinstance(chunk_list, (list, np.memmap)):
            raise SPYTypeError(chunk_list, varname="chunk_list", expected="array_like")

        # Do not use ``array_parser`` to validate chunks to not force-load memmaps
        try:
            shapes = [chunk.shape for chunk in chunk_list]
        except:
            raise SPYTypeError(chunk_list[0], varname="chunk in chunk_list",
                               expected="2d-array-like")
        if np.any([len(shape) != 2 for shape in shapes]):
            raise SPYValueError(legal="2d-array", varname="chunk in chunk_list")

        # Get row number per input chunk and raise error in case col.-no. does not match up
        shapes = [chunk.shape for chunk in chunk_list]
        if not np.array_equal([shape[0] for shape in shapes], [shapes[0][0]] * len(shapes)):
            raise SPYValueError(legal="identical number of samples per chunk",
                                varname="chunk_list")
        ncols = [shape[1] for shape in shapes]
        cumlen = np.cumsum(ncols)

        # Get hierarchically "highest" dtype of data present in `chunk_list`
        dtypes = []
        for chunk in chunk_list:
            dtypes.append(chunk.dtype)
        cdtype = np.max(dtypes)

        # Create list of "global" row numbers and assign "global" dimensional info
        self._ncols = ncols
        self._cols = [range(start, stop) for (start, stop) in zip(cumlen - ncols, cumlen)]
        self._M = chunk_list[0].shape[0]
        self._N = cumlen[-1]
        self._shape = (self._M, self._N)
        self._size = self._M * self._N
        self._dtype = cdtype
        self._data = chunk_list

    # Compatibility
    def __len__(self):
        return self._size

    # The only part of this class that actually does something
    def __getitem__(self, idx):

        # Extract queried row/col from input tuple `idx`
        qrow, qcol = idx

        # Convert input to slice (if it isn't already) or assign explicit start/stop values
        if isinstance(qrow, numbers.Number):
            try:
                scalar_parser(qrow, varname="row", ntype="int_like", lims=[0, self._M])
            except Exception as exc:
                raise exc
            row = slice(int(qrow), int(qrow + 1))
        elif isinstance(qrow, slice):
            start, stop = qrow.start, qrow.stop
            if qrow.start is None:
                start = 0
            if qrow.stop is None:
                stop = self._M
            row = slice(start, stop)
        else:
            raise SPYTypeError(qrow, varname="row", expected="int_like or slice")

        # Convert input to slice (if it isn't already) or assign explicit start/stop values
        if isinstance(qcol, numbers.Number):
            try:
                scalar_parser(qcol, varname="col", ntype="int_like", lims=[0, self._N])
            except Exception as exc:
                raise exc
            col = slice(int(qcol), int(qcol + 1))
        elif isinstance(qcol, slice):
            start, stop = qcol.start, qcol.stop
            if qcol.start is None:
                start = 0
            if qcol.stop is None:
                stop = self._N
            col = slice(start, stop)
        else:
            raise SPYTypeError(qcol, varname="col", expected="int_like or slice")

        # Make sure queried row/col are inside dimensional limits
        err = "value between {lb:s} and {ub:s}"
        if not(0 <= row.start < self._M) or not(0 < row.stop <= self._M):
            raise SPYValueError(err.format(lb="0", ub=str(self._M)),
                                varname="row", actual=str(row))
        if not(0 <= col.start < self._N) or not(0 < col.stop <= self._N):
            raise SPYValueError(err.format(lb="0", ub=str(self._N)),
                                varname="col", actual=str(col))

        # The interesting part: find out wich chunk(s) `col` is pointing at
        i1 = np.where([col.start in chunk for chunk in self._cols])[0].item()
        i2 = np.where([(col.stop - 1) in chunk for chunk in self._cols])[0].item()

        # If start and stop are not within the same chunk, data is loaded into memory
        if i1 != i2:
            data = []
            data.append(self._data[i1][row, col.start - self._cols[i1].start:])
            for i in range(i1 + 1, i2):
                data.append(self._data[i][row, :])
            data.append(self._data[i2][row, :col.stop - self._cols[i2].start])
            return np.hstack(data)

        # If start and stop are in the same chunk, return a view of the underlying memmap
        else:

            # Convert "global" row index to local chunk-based row-number (by subtracting offset)
            col = slice(col.start - self._cols[i1].start, col.stop - self._cols[i1].start)
            return self._data[i1][:, col][row, :]

    # Legacy support
    def __repr__(self):
        return self.__str__()

    # Make class contents comprehensible when viewed from the command line
    def __str__(self):
        ppstr = "{shape:s} element {name:s} object mapping {numfiles:s} file(s)"
        return ppstr.format(shape="[" + " x ".join([str(numel) for numel in self.shape]) + "]",
                            name=self.__class__.__name__,
                            numfiles=str(len(self._ncols)))

    # Free memory by force-closing resident memory maps
    def clear(self):
        """Clear read data from memory

        Reinstantiates memory maps of all open files.

        """
        shapes = []
        dtypes = []
        fnames = []
        offset = []
        for mmp in self._data:
            shapes.append(mmp.shape)
            dtypes.append(mmp.dtype)
            fnames.append(mmp.filename)
            offset.append(mmp.offset)
        self._data = []
        for k in range(len(fnames)):
            self._data.append(np.memmap(fnames[k], offset=offset[k],
                                        mode="r", dtype=dtypes[k],
                                        shape=shapes[k]))
        return

    # Ensure compatibility b/w `VirtualData`, HDF5 datasets and memmaps
    def flush(self):
        self.clear()


class Indexer():

    __slots__ = ["_iterobj", "_iterlen"]

    def __init__(self, iterobj, iterlen):
        """
        Make an iterable object subscriptable using itertools magic
        """
        self._iterobj = iterobj
        self._iterlen = iterlen

    def __iter__(self):
        return self._iterobj

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Number):
            try:
                scalar_parser(idx, varname="idx", ntype="int_like",
                                  lims=[0, self._iterlen - 1])
            except Exception as exc:
                raise exc
            return next(islice(self._iterobj, idx, idx + 1))
        elif isinstance(idx, slice):
            start, stop = idx.start, idx.stop
            if idx.start is None:
                start = 0
            if idx.stop is None:
                stop = self._iterlen
            index = slice(start, stop, idx.step)
            if not(0 <= index.start < self._iterlen) or not (0 < index.stop <= self._iterlen):
                err = "value between {lb:s} and {ub:s}"
                raise SPYValueError(err.format(lb="0", ub=str(self._iterlen)),
                                    varname="idx", actual=str(index))
            return np.hstack(islice(self._iterobj, index.start, index.stop, index.step))
        elif isinstance(idx, (list, np.ndarray)):
            try:
                array_parser(idx, varname="idx", ntype="int_like", hasnan=False,
                             hasinf=False, lims=[0, self._iterlen], dims=1)
            except Exception as exc:
                raise exc
            return np.hstack([next(islice(self._iterobj, int(ix), int(ix + 1))) for ix in idx])
        else:
            raise SPYTypeError(idx, varname="idx", expected="int_like or slice")

    def __len__(self):
        return self._iterlen

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{} element iterable".format(self._iterlen)

    
class SessionLogger():

    __slots__ = ["sessionfile", "_rm"]

    def __init__(self):
        sess_log = "{user:s}@{host:s}: <{time:s}> started session {sess:s}"
        self.sessionfile = os.path.join(__storage__,
                                        "session_{}_log.id".format(__sessionid__))
        with open(self.sessionfile, "w") as fid:
            fid.write(sess_log.format(user=getpass.getuser(),
                                      host=socket.gethostname(),
                                      time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                      sess=__sessionid__))
        self._rm = os.unlink # workaround to prevent Python from garbage-collectiing ``os.unlink``

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Session {}".format(__sessionid__)

    def __del__(self):
        self._rm(self.sessionfile)


class StructDict(dict):
    """Child-class of dict for emulating MATLAB structs

    Examples
    --------
    cfg = StructDict()
    cfg.a = [0, 25]

    """
    
    def __init__(self, *args, **kwargs):
        """
        Create a child-class of dict whose attributes are its keys
        (thus ensuring that attributes and items are always in sync)
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self        


class FauxTrial():
    
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        
    def __str__(self):
        msg = "Trial placeholder of shape {} and datatype {}"
        return msg.format(str(self.shape), str(self.dtype))

    def __repr__(self):
        return self.__str__()

    def squeeze(self):
        shp = list(self.shape)
        while 1 in shp:
            shp.remove(1)
        return FauxTrial(shp, self.dtype)
    
