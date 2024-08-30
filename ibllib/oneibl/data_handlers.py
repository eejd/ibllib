"""Downloading of task dependent datasets and registration of task output datasets.

The DataHandler class is used by the pipes.tasks.Task class to ensure dependent datasets are
present and to register and upload the output datasets.  For examples on how to run a task using
specific data handlers, see :func:`ibllib.pipes.tasks`.
"""
import logging
import pandas as pd
from pathlib import Path
import shutil
import os
import abc
from time import time
from functools import partial

from one.api import ONE
from one.webclient import AlyxClient
from one.util import filter_datasets, ensure_list
from one.alf.io import iter_datasets
from one.alf.files import add_uuid_string, session_path_parts
from one.alf.cache import DATASETS_COLUMNS, _get_dataset_info
from ibllib.oneibl.registration import register_dataset, get_lab, get_local_data_repository
from ibllib.oneibl.patcher import FTPPatcher, SDSCPatcher, SDSC_ROOT_PATH, SDSC_PATCH_PATH


_logger = logging.getLogger(__name__)


class DataHandler(abc.ABC):
    def __init__(self, session_path, signature, one=None):
        """
        Base data handler class
        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        self.session_path = session_path
        self.signature = signature
        self.one = one
        self.processed = {}  # Map of filepaths and their processed records (e.g. upload receipts or Alyx records)

    def setUp(self):
        """Function to optionally overload to download required data to run task."""
        pass

    def getData(self, one=None):
        """Finds the datasets required for task based on input signatures."""
        if self.one is None and one is None:
            return

        one = one or self.one
        session_datasets = one.list_datasets(one.path2eid(self.session_path), details=True)
        dfs = []
        for file in self.signature['input_files']:
            dfs.append(filter_datasets(session_datasets, filename=file[0], collection=file[1],
                       wildcards=True, assert_unique=False))
        if len(dfs) == 0:
            return pd.DataFrame()
        df = pd.concat(dfs)

        # Some cases the eid is stored in the index. If so we drop this level
        if 'eid' in df.index.names:
            df = df.droplevel(level='eid')
        return df

    def getOutputFiles(self):
        assert self.session_path
        # Next convert datasets to frame
        # Create dataframe of all ALF datasets
        dsets = iter_datasets(self.session_path)
        records = [_get_dataset_info(self.session_path, dset, compute_hash=False) for dset in dsets]
        df = pd.DataFrame(records, columns=DATASETS_COLUMNS)
        filt = partial(filter_datasets, df, wildcards=True, assert_unique=False)
        # Filter outputs
        dids = pd.concat(filt(filename=file[0], collection=file[1]).index for file in self.signature['output_files'])
        present = df.loc[dids, :].copy()
        return present

    def uploadData(self, outputs, version):
        """
        Function to optionally overload to upload and register data
        :param outputs: output files from task to register
        :param version: ibllib version
        :return:
        """
        if isinstance(outputs, list):
            versions = [version for _ in outputs]
        else:
            versions = [version]

        return versions

    def cleanUp(self):
        """Function to optionally overload to clean up files after running task."""
        pass


class LocalDataHandler(DataHandler):
    def __init__(self, session_path, signatures, one=None):
        """
        Data handler for running tasks locally, with no architecture or db connection
        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signatures, one=one)


class ServerDataHandler(DataHandler):
    def __init__(self, session_path, signatures, one=None):
        """
        Data handler for running tasks on lab local servers when all data is available locally

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signatures, one=one)

    def uploadData(self, outputs, version, clobber=False, **kwargs):
        """
        Upload and/or register output data.

        This is typically called by :meth:`ibllib.pipes.tasks.Task.register_datasets`.

        Parameters
        ----------
        outputs : list of pathlib.Path
            A set of ALF paths to register to Alyx.
        version : str, list of str
            The version of ibllib used to generate these output files.
        clobber : bool
            If True, re-upload outputs that have already been passed to this method.
        kwargs
            Optional keyword arguments for one.registration.RegistrationClient.register_files.

        Returns
        -------
        list of dicts, dict
            A list of newly created Alyx dataset records or the registration data if dry.
        """
        versions = super().uploadData(outputs, version)
        data_repo = get_local_data_repository(self.one.alyx)
        # If clobber = False, do not re-upload the outputs that have already been processed
        outputs = ensure_list(outputs)
        to_upload = list(filter(None if clobber else lambda x: x not in self.processed, outputs))
        records = register_dataset(to_upload, one=self.one, versions=versions, repository=data_repo, **kwargs) or []
        if kwargs.get('dry', False):
            return records
        # Store processed outputs
        self.processed.update({k: v for k, v in zip(to_upload, records) if v})
        return [self.processed[x] for x in outputs if x in self.processed]

    def cleanUp(self):
        """Empties and returns the processed dataset mep."""
        super().cleanUp()
        processed = self.processed
        self.processed = {}
        return processed


class ServerGlobusDataHandler(DataHandler):
    def __init__(self, session_path, signatures, one=None):
        """
        Data handler for running tasks on lab local servers. Will download missing data from SDSC using Globus

        :param session_path: path to session
        :param signatures: input and output file signatures
        :param one: ONE instance
        """
        from one.remote.globus import Globus, get_lab_from_endpoint_id  # noqa
        super().__init__(session_path, signatures, one=one)
        self.globus = Globus(client_name='server', headless=True)

        # on local servers set up the local root path manually as some have different globus config paths
        self.globus.endpoints['local']['root_path'] = '/mnt/s0/Data/Subjects'

        # Find the lab
        self.lab = get_lab(self.session_path, self.one.alyx)

        # For cortex lab we need to get the endpoint from the ibl alyx
        if self.lab == 'cortexlab':
            alyx = AlyxClient(base_url='https://alyx.internationalbrainlab.org', cache_rest=None)
            self.globus.add_endpoint(f'flatiron_{self.lab}', alyx=alyx)
        else:
            self.globus.add_endpoint(f'flatiron_{self.lab}', alyx=self.one.alyx)

        self.local_paths = []

    def setUp(self):
        """Function to download necessary data to run tasks using globus-sdk."""
        if self.lab == 'cortexlab':
            df = super().getData(one=ONE(base_url='https://alyx.internationalbrainlab.org'))
        else:
            df = super().getData(one=self.one)

        if len(df) == 0:
            # If no datasets found in the cache only work off local file system do not attempt to
            # download any missing data using Globus
            return

        # Check for space on local server. If less that 500 GB don't download new data
        space_free = shutil.disk_usage(self.globus.endpoints['local']['root_path'])[2]
        if space_free < 500e9:
            _logger.warning('Space left on server is < 500GB, won\'t re-download new data')
            return

        rel_sess_path = '/'.join(df.iloc[0]['session_path'].split('/')[-3:])
        assert rel_sess_path.split('/')[0] == self.one.path2ref(self.session_path)['subject']

        target_paths = []
        source_paths = []
        for i, d in df.iterrows():
            sess_path = Path(rel_sess_path).joinpath(d['rel_path'])
            full_local_path = Path(self.globus.endpoints['local']['root_path']).joinpath(sess_path)
            if not full_local_path.exists():
                uuid = i
                self.local_paths.append(full_local_path)
                target_paths.append(sess_path)
                source_paths.append(add_uuid_string(sess_path, uuid))

        if len(target_paths) != 0:
            ts = time()
            for sp, tp in zip(source_paths, target_paths):
                _logger.info(f'Downloading {sp} to {tp}')
            self.globus.mv(f'flatiron_{self.lab}', 'local', source_paths, target_paths)
            _logger.debug(f'Complete. Time elapsed {time() - ts}')

    def uploadData(self, outputs, version, **kwargs):
        """
        Function to upload and register data of completed task
        :param outputs: output files from task to register
        :param version: ibllib version
        :return: output info of registered datasets
        """
        versions = super().uploadData(outputs, version)
        data_repo = get_local_data_repository(self.one.alyx)
        return register_dataset(outputs, one=self.one, versions=versions, repository=data_repo, **kwargs)

    def cleanUp(self):
        """Clean up, remove the files that were downloaded from Globus once task has completed."""
        for file in self.local_paths:
            os.unlink(file)


class RemoteHttpDataHandler(DataHandler):
    def __init__(self, session_path, signature, one=None):
        """
        Data handler for running tasks on remote compute node. Will download missing data via http using ONE

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signature, one=one)

    def setUp(self):
        """
        Function to download necessary data to run tasks using ONE
        :return:
        """
        df = super().getData()
        self.one._check_filesystem(df)

    def uploadData(self, outputs, version, **kwargs):
        """
        Function to upload and register data of completed task via FTP patcher
        :param outputs: output files from task to register
        :param version: ibllib version
        :return: output info of registered datasets
        """
        versions = super().uploadData(outputs, version)
        ftp_patcher = FTPPatcher(one=self.one)
        return ftp_patcher.create_dataset(path=outputs, created_by=self.one.alyx.user,
                                          versions=versions, **kwargs)


class RemoteAwsDataHandler(DataHandler):
    def __init__(self, task, session_path, signature, one=None):
        """
        Data handler for running tasks on remote compute node.

        This will download missing data from the private IBL S3 AWS data bucket.  New datasets are
        uploaded via Globus.

        :param session_path: path to session
        :param signature: input and output file signatures
        :param one: ONE instance
        """
        super().__init__(session_path, signature, one=one)
        self.task = task

        self.local_paths = []

    def setUp(self):
        """Function to download necessary data to run tasks using AWS boto3."""
        df = super().getData()
        self.local_paths = self.one._download_aws(map(lambda x: x[1], df.iterrows()))

    def uploadData(self, outputs, version, **kwargs):
        """
        Function to upload and register data of completed task via FTP patcher
        :param outputs: output files from task to register
        :param version: ibllib version
        :return: output info of registered datasets
        """
        # Set up Globus
        from one.remote.globus import Globus # noqa
        self.globus = Globus(client_name='server', headless=True)
        self.lab = session_path_parts(self.session_path, as_dict=True)['lab']
        if self.lab == 'cortexlab' and 'cortexlab' in self.one.alyx.base_url:
            base_url = 'https://alyx.internationalbrainlab.org'
            _logger.warning('Changing Alyx client to %s', base_url)
            ac = AlyxClient(base_url=base_url)
        else:
            ac = self.one.alyx
        self.globus.add_endpoint(f'flatiron_{self.lab}', alyx=ac)

        # register datasets
        versions = super().uploadData(outputs, version)
        response = register_dataset(outputs, one=self.one, server_only=True, versions=versions, **kwargs)

        # upload directly via globus
        source_paths = []
        target_paths = []
        collections = {}

        for dset, out in zip(response, outputs):
            assert Path(out).name == dset['name']
            # set flag to false
            fr = next(fr for fr in dset['file_records'] if 'flatiron' in fr['data_repository'])
            collection = '/'.join(fr['relative_path'].split('/')[:-1])
            if collection in collections.keys():
                collections[collection].update({f'{dset["name"]}': {'fr_id': fr['id'], 'size': dset['file_size']}})
            else:
                collections[collection] = {f'{dset["name"]}': {'fr_id': fr['id'], 'size': dset['file_size']}}

            # Set all exists status to false for server file records
            self.one.alyx.rest('files', 'partial_update', id=fr['id'], data={'exists': False})

            source_paths.append(out)
            target_paths.append(add_uuid_string(fr['relative_path'], dset['id']))

        if len(target_paths) != 0:
            ts = time()
            for sp, tp in zip(source_paths, target_paths):
                _logger.info(f'Uploading {sp} to {tp}')
            self.globus.mv('local', f'flatiron_{self.lab}', source_paths, target_paths)
            _logger.debug(f'Complete. Time elapsed {time() - ts}')

        for collection, files in collections.items():
            globus_files = self.globus.ls(f'flatiron_{self.lab}', collection, remove_uuid=True, return_size=True)
            file_names = [str(gl[0]) for gl in globus_files]
            file_sizes = [gl[1] for gl in globus_files]

            for name, details in files.items():
                try:
                    idx = file_names.index(name)
                    size = file_sizes[idx]
                    if size == details['size']:
                        # update the file record if sizes match
                        self.one.alyx.rest('files', 'partial_update', id=details['fr_id'], data={'exists': True})
                    else:
                        _logger.warning(f'File {name} found on SDSC but sizes do not match')
                except ValueError:
                    _logger.warning(f'File {name} not found on SDSC')

        return response

        # ftp_patcher = FTPPatcher(one=self.one)
        # return ftp_patcher.create_dataset(path=outputs, created_by=self.one.alyx.user,
        #                                   versions=versions, **kwargs)

    def cleanUp(self):
        """Clean up, remove the files that were downloaded from globus once task has completed."""
        if self.task.status == 0:
            for file in self.local_paths:
                os.unlink(file)


class RemoteGlobusDataHandler(DataHandler):
    """
    Data handler for running tasks on remote compute node. Will download missing data using Globus.

    :param session_path: path to session
    :param signature: input and output file signatures
    :param one: ONE instance
    """
    def __init__(self, session_path, signature, one=None):
        super().__init__(session_path, signature, one=one)

    def setUp(self):
        """Function to download necessary data to run tasks using globus."""
        # TODO
        pass

    def uploadData(self, outputs, version, **kwargs):
        """
        Function to upload and register data of completed task via FTP patcher
        :param outputs: output files from task to register
        :param version: ibllib version
        :return: output info of registered datasets
        """
        versions = super().uploadData(outputs, version)
        ftp_patcher = FTPPatcher(one=self.one)
        return ftp_patcher.create_dataset(path=outputs, created_by=self.one.alyx.user,
                                          versions=versions, **kwargs)


class SDSCDataHandler(DataHandler):
    """
    Data handler for running tasks on SDSC compute node

    :param session_path: path to session
    :param signature: input and output file signatures
    :param one: ONE instance
    """

    def __init__(self, task, session_path, signatures, one=None):
        super().__init__(session_path, signatures, one=one)
        self.task = task
        self.SDSC_PATCH_PATH = SDSC_PATCH_PATH
        self.SDSC_ROOT_PATH = SDSC_ROOT_PATH

    def setUp(self):
        """Function to create symlinks to necessary data to run tasks."""
        df = super().getData()

        SDSC_TMP = Path(self.SDSC_PATCH_PATH.joinpath(self.task.__class__.__name__))
        for i, d in df.iterrows():
            file_path = Path(d['session_path']).joinpath(d['rel_path'])
            uuid = i
            file_uuid = add_uuid_string(file_path, uuid)
            file_link = SDSC_TMP.joinpath(file_path)
            file_link.parent.mkdir(exist_ok=True, parents=True)
            try:
                file_link.symlink_to(
                    Path(self.SDSC_ROOT_PATH.joinpath(file_uuid)))
            except FileExistsError:
                pass

        self.task.session_path = SDSC_TMP.joinpath(d['session_path'])

    def uploadData(self, outputs, version, **kwargs):
        """
        Function to upload and register data of completed task via SDSC patcher
        :param outputs: output files from task to register
        :param version: ibllib version
        :return: output info of registered datasets
        """
        versions = super().uploadData(outputs, version)
        sdsc_patcher = SDSCPatcher(one=self.one)
        return sdsc_patcher.patch_datasets(outputs, dry=False, versions=versions, **kwargs)

    def cleanUp(self):
        """Function to clean up symlinks created to run task."""
        assert SDSC_PATCH_PATH.parts[0:4] == self.task.session_path.parts[0:4]
        shutil.rmtree(self.task.session_path)


class PopeyeDataHandler(SDSCDataHandler):

    def __init__(self, task, session_path, signatures, one=None):
        super().__init__(task, session_path, signatures, one=one)
        self.SDSC_PATCH_PATH = Path(os.getenv('SDSC_PATCH_PATH', "/mnt/sdceph/users/ibl/data/quarantine/tasks/"))
        self.SDSC_ROOT_PATH = Path("/mnt/sdceph/users/ibl/data")

    def uploadData(self, outputs, version, **kwargs):
        raise NotImplementedError(
            "Cannot register data from Popeye. Login as Datauser and use the RegisterSpikeSortingSDSC task."
        )

    def cleanUp(self):
        """Symlinks are preserved until registration."""
        pass
