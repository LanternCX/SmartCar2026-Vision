"""
Relevant functional functions of the HTTP client, providing various HTTP request methods.

MicroPython module: https://docs.micropython.org/en/preview/library/urequests.html

Relevant functional functions of the HTTP client, providing various HTTP request methods

Response class
--------------
"""

# source version: preview
# origin module:: repos/micropython/docs/library/urequests.rst
from __future__ import annotations
from typing import Dict
from _typeshed import Incomplete

class Response:
    """
    The Response class object contains the server's response to the HTTP request.

    Methods
    ~~~~~~~
    """

    def __init__(self, s) -> None: ...
    def json(self) -> Dict:
        """
        Return response json encoded content and convert to dict type.
        """
        ...

def request(function, url, data=None, json=None, files=None, headers={}, auth=None) -> None:
    """
    Send an HTTP request to the server.

        - ``function`` - HTTP function to use
        - ``url`` - URL to send
        - ``data`` - To append to the body of the request. If a dictionary or tuple list is provided, the form will be encoded.
        - ``json`` - json is used to attach to the body of the request.
        - ``files`` - Used for file upload, the type is 2-tuple, which defines the file name, file path and content type. As follows,{‘name’, (file directory,content-type)}
        - ``headers`` - Dictionary of headers to send.
        - ``auth`` - Auth tuple to enable Basic/Digest/Custom HTTP Auth.
    """
    ...

def head(url, **kw) -> Response:
    """
    Send HEAD request and return Response object.

        - ``url`` - Request object URL
        - ``**kw`` - The parameters of the request function.
    """
    ...

def get(url, **kw) -> Response:
    """
    Send GET request and return Response object.

        - ``url`` - Request object URL
        - ``**kw`` - Parameters of request function.
    """
    ...

def post(url, **kw) -> Response:
    """
    Send POST request and return Response object.

        - ``url`` - Request object URL
        - ``**kw`` - Parameters of request function.
    """
    ...

def put(url, **kw) -> Response:
    """
    Send PUT request and return Response object.

        - ``url`` - RRequest object URL
        - ``**kw`` - Parameters of request function.
    """
    ...

def patch(url, **kw) -> Response:
    """
    Send PATCH request, return Response object.

        - ``url`` - Request object URL
        - ``**kw`` - Parameters of request function.
    """
    ...

def delete(url, **kw) -> Request:
    """
    Send a DELETE request. Return Response object。

        - ``url`` - Request object URL
        - ``**kw`` - Parameters of request function.
    """
    ...
