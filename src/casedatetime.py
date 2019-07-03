"""
Author: Matt Nicholson
"""

class CaseDateTime(object):

    def __init__(self, date, time, point1, point2):
        """
        Date format: MM-DD-YYYY
        Time format: HH:MM
        point format: (lat, lon)
        """
        super(CaseDateTime, self).__init__()
        self.year = None
        self.month = None
        self.day = None
        self.hour = None
        self.minute = None
        self.min_lat = None
        self.max_lat = None
        self.min_lon = None
        self.max_lon = None
        _parse_date(date)
        _parse_coords(point1, point2)



    def _parse_date(self, date):
        self.month, self.day, self.year = date.split('-')



    def _parse_time(self, time):
        self.hour, self.minute = time.split(':')



    def _parse_coords(self, point1, point2):
        self.min_lat = min(point1[0], point2[0])
        self.max_lat = max(point1[0], point2[0])
        self.min_lon = min(point1[1], point2[1])
        self.max_lon = max(point1[1], point2[1])



    def date(self):
        return '{}-{}-{}'.format(self.month, self.day, self.year)



    def time(self):
        return '{}:{}'.format(self.hour, self.minute)
