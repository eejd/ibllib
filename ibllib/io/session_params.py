"""A module for handling experiment description files.

Each device computer adds its piece of information and consolidates into the final acquisition
description.

The purpose is 3-fold:
    - provide modularity in the extraction: the acquisition description allows to dynamically build
    pipelines.
    - assist the copying of the experimental data from device computers to the server computers, in
    a way that each device is independent from another.
    - assist the copying of the experimental data from device computers to the server computers, in
    a way that intermediate states (failed copies) are easily recoverable from and completion
    criteria (ie. session ready to extract) is objective and simple (all device files copied).

INGRESS
    - each device computer needs to know the session path on the server.
    - create a device file locally in a queue directory. This will serve as a copy flag.
    - copy the device file to the local server.

EGRESS
    - go through the queue and for each item:
        - if the device file is not on the server create it.
        - once copy is complete aggregate the qc from file.
"""
import yaml
import logging
from pathlib import Path
from copy import deepcopy

from one.converters import ConversionMixin
from iblutil.util import flatten
from iblutil.io.params import FileLock
from packaging import version

import ibllib.pipes.misc as misc


_logger = logging.getLogger(__name__)
SPEC_VERSION = '1.0.0'


def write_yaml(file_path, data):
    """
    Write a device file. This is basically just a yaml dump that ensures the folder tree exists.

    Parameters
    ----------
    file_path : pathlib.Path
        The full path to the description yaml file to write to.
    data : dict
        The data to write to the yaml file.

    """
    file_path.parent.mkdir(exist_ok=True, parents=True)
    with open(file_path, 'w') as fp:
        yaml.safe_dump(data, fp)


def _patch_file(data: dict) -> dict:
    """
    Update older description data to conform to the most recent specification.

    Parameters
    ----------
    data : dict
        The description yaml data.

    Returns
    -------
    dict
        The patched description data.
    """
    if data and (v := data.get('version', '0')) != SPEC_VERSION:
        if version.parse(v) > version.parse(SPEC_VERSION):
            _logger.warning('Description file generated by more recent code')
        elif version.parse(v) <= version.parse('0.1.0'):
            # Change tasks key from dict to list of dicts
            if 'tasks' in data and isinstance(data['tasks'], dict):
                data['tasks'] = [{k: v} for k, v in data['tasks'].copy().items()]
        data['version'] = SPEC_VERSION
    return data


def write_params(session_path, data) -> Path:
    """
    Write acquisition description data to the session path.

    Parameters
    ----------
    session_path : str, pathlib.Path
        A session path containing an _ibl_experiment.description.yaml file.
    data : dict
        The acquisition description data to save

    Returns
    -------
    pathlib.Path
        The full path to the saved acquisition description.
    """
    yaml_file = Path(session_path).joinpath('_ibl_experiment.description.yaml')
    write_yaml(yaml_file, data)
    return yaml_file


def read_params(path) -> dict:
    """
    Load an experiment description file.

    In addition to reading the yaml data, this functions ensures that the specification is the most
    recent one.  If the file is missing None is returned.  If the file cannot be parsed an empty
    dict is returned.

    Parameters
    ----------
    path : pathlib.Path, str
        The path to the description yaml file (or it's containing folder) to load.

    Returns
    -------
    dict, None
        The parsed yaml data, or None if the file was not found.

    Examples
    --------
    # Load a session's _ibl_experiment.description.yaml file

    >>> data = read_params('/home/data/subject/2020-01-01/001')

    # Load a specific device's description file

    >>> data = read_params('/home/data/subject/2020-01-01/001/_devices/behaviour.yaml')

    """
    if (path := Path(path)).is_dir():
        yaml_file = next(path.glob('_ibl_experiment.description*'), None)
    else:
        yaml_file = path if path.exists() else None
    if not yaml_file:
        _logger.debug('Experiment description not found: %s', path)
        return

    with open(yaml_file, 'r') as fp:
        data = _patch_file(yaml.safe_load(fp) or {})
    return data


def merge_params(a, b, copy=False):
    """
    Given two experiment descriptions, update first with fields in second.

    Parameters
    ----------
    a : dict
        An experiment description dictionary to be updated with fields from `b`.
    b : dict
        An experiment description dictionary to update `a` with
    copy : bool
        If true, return a deep copy of `a` instead of updating directly.

    Returns
    -------
    dict
        A merged dictionary consisting of fields from `a` and `b`.
    """
    if copy:
        a = deepcopy(a)
    for k in b:
        if k == 'sync':
            assert k not in a or a[k] == b[k], 'multiple sync fields defined'
        if isinstance(b[k], list):
            prev = list(a.get(k, []))
            # For procedures and projects, remove duplicates
            to_add = b[k] if k == 'tasks' else set(b[k]) - set(prev)
            a[k] = prev + list(to_add)
        elif isinstance(b[k], dict):
            a[k] = {**a.get(k, {}), **b[k]}
        else:  # A string
            a[k] = b[k]
    return a


def aggregate_device(file_device, file_acquisition_description, unlink=False):
    """
    Add the contents of a device file to the main acquisition description file.

    Parameters
    ----------
    file_device : pathlib.Path
        The full path to the device yaml file to add to the main description file.
    file_acquisition_description : pathlib.Path
        The full path to the main acquisition description yaml file to add the device file to.
    unlink : bool
        If True, the device file is removed after successfully aggregation.

    Returns
    -------
    dict
        The aggregated experiment description data.

    Raises
    ------
    AssertionError
        Device file contains a main 'sync' key that is already present in the main description
        file.  For an experiment only one main sync device is allowed.
    """
    # reads in the partial device data
    data_device = read_params(file_device)

    if not data_device:
        _logger.warning('empty device file "%s"', file_device)
        return

    with FileLock(file_acquisition_description, log=_logger, timeout_action='delete'):
        # if the acquisition description file already exists, read in the yaml content
        if file_acquisition_description.exists():
            acq_desc = read_params(file_acquisition_description)
        else:
            acq_desc = {}

        # merge the dictionaries (NB: acq_desc modified in place)
        acq_desc = merge_params(acq_desc, data_device)

        with open(file_acquisition_description, 'w') as fp:
            yaml.safe_dump(acq_desc, fp)

    # delete the original file if necessary
    if unlink:
        file_device.unlink()
        stub_folder = file_acquisition_description.with_name('_devices')
        if stub_folder.exists() and not any(stub_folder.glob('*.*')):
            stub_folder.rmdir()

    return acq_desc


def get_cameras(sess_params):
    devices = sess_params.get('devices', {})
    cameras = devices.get('cameras', None)
    return None if not cameras else list(cameras.keys())


def get_sync_label(sess_params):
    if not sess_params:
        return None
    sync_keys = list((sess_params.get('sync') or {}).keys())
    if len(sync_keys) == 0:
        return None
    if len(sync_keys) > 1:
        _logger.warning('Multiple sync keys found in experiment description: %s', sync_keys)
    return sync_keys[0]


def get_sync(sess_params):
    sync_label = get_sync_label(sess_params)
    if sync_label:
        return sync_label, sess_params['sync'][sync_label] or {}
    return None, {}


def get_sync_values(sess_params):
    key = get_sync_label(sess_params)
    if key:
        return sess_params['sync'][key]


def get_sync_collection(sess_params):
    return (get_sync_values(sess_params) or {}).get('collection')


def get_sync_extension(sess_params):
    return (get_sync_values(sess_params) or {}).get('extension')


def get_sync_namespace(sess_params):
    return (get_sync_values(sess_params) or {}).get('acquisition_software')


def get_task_protocol(sess_params, task_collection=None):
    """
    Fetch the task protocol from an experiment description dict.

    Parameters
    ----------
    sess_params : dict
        The loaded experiment.description file.
    task_collection : str, optional
        Return the protocol that corresponds to this collection (returns the first matching
        protocol in the list). If None, all protocols are returned.

    Returns
    -------
    str, set, None
        If task_collection is None, returns the set of task protocols, otherwise returns the first
        protocol that corresponds to the collection, or None if collection not present.
    """
    collections = get_collections({'tasks': sess_params.get('tasks')})
    if task_collection is None:
        if len(collections) == 0:
            return None
        else:
            return set(collections.keys())  # Return all protocols
    else:
        return next((k for k, v in collections.items() if v == task_collection), None)


def get_task_collection(sess_params, task_protocol=None):
    """
    Fetch the task collection from an experiment description dict.

    Parameters
    ----------
    sess_params : dict
        The loaded experiment.description file.
    task_protocol : str, optional
        Return the collection that corresponds to this protocol (returns the first matching
        protocol in the list). If None, all collections are returned.

    Returns
    -------
    str, set, None
        If task_protocol is None, returns the set of collections, otherwise returns the first
        collection that corresponds to the protocol, or None if protocol not present.

    Notes
    -----
    - The order of the set may not be the same as the descriptions tasks order when iterating.
    """
    protocols = sess_params.get('tasks', [])
    if task_protocol is not None:
        task = next((x for x in protocols if task_protocol in x), None)
        return (task.get(task_protocol) or {}).get('collection')
    else:  # Return set of all task collections
        cset = set(filter(None, (next(iter(x.values()), {}).get('collection') for x in protocols)))
        return (next(iter(cset)) if len(cset) == 1 else cset) or None


def get_task_protocol_number(sess_params, task_protocol=None):
    """
    Fetch the task protocol number from an experiment description dict.

    Parameters
    ----------
    sess_params : dict
        The loaded experiment.description file.
    task_protocol : str, optional
        Return the number that corresponds to this protocol (returns the first matching
        protocol in the list). If None, all numbers are returned.

    Returns
    -------
    str, list, None
        If task_protocol is None, returns list of all numbers, otherwise returns the first
        number that corresponds to the protocol, or None if protocol not present.
    """
    protocols = sess_params.get('tasks', [])
    if task_protocol is not None:
        task = next((x for x in protocols if task_protocol in x), None)
        number = (task.get(task_protocol) or {}).get('protocol_number')
        return int(number) if isinstance(number, str) else number
    else:  # Return set of all task numbers
        numbers = list(filter(None, (next(iter(x.values()), {}).get('protocol_number') for x in protocols)))
        numbers = [int(n) if isinstance(n, str) else n for n in numbers]
        return (next(iter(numbers)) if len(numbers) == 1 else numbers) or None


def get_collections(sess_params, flat=False):
    """
    Find all collections associated with the session.

    Parameters
    ----------
    sess_params : dict
        The loaded experiment description map.
    flat : bool (False)
        If True, return a flat set of collections, otherwise return a map of device/sync/task

    Returns
    -------
    dict[str, str]
        A map of device/sync/task and the corresponding collection name.

    set[str]
        A set of unique collection names.

    Notes
    -----
    - Assumes only the following data types contained: list, dict, None, str.
    """
    collection_map = {}

    def iter_dict(d):
        for k, v in d.items():
            if isinstance(v, list):
                for d in filter(lambda x: isinstance(x, dict), v):
                    iter_dict(d)
            elif isinstance(v, dict) and 'collection' in v:
                # if the key already exists, append the collection name to the list
                if k in collection_map:
                    clist = collection_map[k] if isinstance(collection_map[k], list) else [collection_map[k]]
                    collection_map[k] = list(set(clist + [v['collection']]))
                else:
                    collection_map[k] = v['collection']
            elif isinstance(v, dict):
                iter_dict(v)

    iter_dict(sess_params)
    return set(flatten(collection_map.values())) if flat else collection_map


def get_video_compressed(sess_params):
    videos = sess_params.get('devices', {}).get('cameras', None)
    if not videos:
        return None

    # This is all or nothing, assumes either all videos or not compressed
    for key, vals in videos.items():
        compressed = vals.get('compressed', False)

    return compressed


def get_remote_stub_name(session_path, device_id=None):
    """
    Get or create device specific file path for the remote experiment.description stub.

    Parameters
    ----------
    session_path : pathlib.Path
        A remote session path.
    device_id : str, optional
        A device name, if None the TRANSFER_LABEL parameter is used (defaults to this device's
        hostname with a unique numeric ID)

    Returns
    -------
    pathlib.Path
        The full file path to the remote experiment description stub.

    Example
    -------
    >>> get_remote_stub_name(Path.home().joinpath('subject', '2020-01-01', '001'), 'host-123')
    Path.home() / 'subject/2020-01-01/001/_devices/2020-01-01_1_subject@host-123.yaml'
    """
    device_id = device_id or misc.create_basic_transfer_params()['TRANSFER_LABEL']
    exp_ref = '{date}_{sequence:d}_{subject:s}'.format(**ConversionMixin.path2ref(session_path))
    remote_filename = f'{exp_ref}@{device_id}.yaml'
    return session_path / '_devices' / remote_filename
