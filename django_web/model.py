

class ServiceException(Exception):
    def __init__(self,err='service error!'):
        Exception.__init__(self,err)