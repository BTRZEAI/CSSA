#!/usr/bin/env python

from urllib.request import urlopen
import ssl
import certifi
import json


def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    context = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, context=context)
    data = response.read().decode("utf-8")
    return json.loads(data)


url = "https://financialmodelingprep.com/api/v3/earnings-surprises/AAPL?apikey=5d1a94f98fcd7b987c1187895de56ff5"
print(get_jsonparsed_data(url))
