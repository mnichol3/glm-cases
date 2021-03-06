"""
Author: Matt Nicholson

Based on Aaron Anderson's nexradaws module, found here: https://github.com/aarande/nexradaws 
"""
class DownloadResults(object):
    """
    This class contains the results of the download call as well as methods for accessing \
    the results.
    :var success: a list of :class:`LocalGoesFile <goesaws.resources.localgoesfile.LocalGoesFile>` \
    objects representing the successful downloads
    :vartype success: list
    :var failed: a list of any :class:`AwsGoesFile <goesaws.resources.awsgoesfile.AwsGoesFile>` \
    objects that failed to download
    :vartype error: list
    :var success_count: The number of successful downloads
    :vartype success_count: int
    :var failed_count: The number of downloads that failed
    :vartype failed_count: int
    :var total: The total number of nexrad files that were attempted
    :vartype total: int
    """
    def __init__(self,localfiles,failedfiles):
        super(DownloadResults, self).__init__()
        self._successfiles = localfiles
        self._failedfiles = failedfiles



    @property
    def failed_count(self):
        return len(self._failedfiles)



    @property
    def success_count(self):
        return len(self._successfiles)



    @property
    def total(self):
        return self.success_count + self.failed_count



    @property
    def success(self):
        return self._successfiles



    @property
    def failed(self):
        return self._failedfiles



    def iter_success(self):
        """
        A generator function that allows you to iterate over successful downloads
        >>> for localnexradfile in downloads.iter_success():
        >>>      ...do something...
        """
        for awsgoesfile in self.success:
            yield awsgoesfile



    def iter_failed(self):
        """
        A generator function that allows you to iterate over failed downloads
        >>> for remotenexradfile in downloads.iter_failed():
        >>>      ...do something...
        """
        for awsgoesfile in self.failed:
            yield awsgoesfile
