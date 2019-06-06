import os


class LocalGoesFile(object):

    def __init__(self, awsgoesfile, localfilepath):
        super(LocalGoesFile, self).__init__()
        self.key = awsgoesfile.key
        self.shortfname = awsgoesfile.shortfname
        self.filename = awsgoesfile.filename
        self.scan_time = awsgoesfile.scan_time
        self.filepath = localfilepath



    def open(self):
        """
        Provides a file object to the local nexrad radar file. \
        Be sure to close the file object when processing is complete.
        :return: file object ready for reading
        :rtype file:
        """
        return open(self.filepath, 'rb')



    def __repr__(self):
        return '<LocalGoesFile object - {}>'.format(self.filepath)
