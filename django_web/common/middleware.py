# -*- coding: utf-8 -*-
import time
import threading
from django.utils.deprecation import MiddlewareMixin
MAX_REQUEST_PER_SECOND=1 #num per second
mutex = threading.Lock()
class RequestBlockingMiddleware(MiddlewareMixin):
    def process_request(self,request):
        print("加锁")
        mutex.acquire(2)

    def process_response(self,request,response):
        mutex.release()
        print("释放锁")
        return response