# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np

_im_zfile = []
_xml_path_zip = []
_xml_zfile = []


def imread(filename, flags=cv2.IMREAD_COLOR):
    global _im_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        print("character '@' is not found from the given path '%s'" % (path))
        raise
    path_zip = path[0:pos_at]
    path_img = path[pos_at + 2:]

    # 在 Windows 上，路径分隔符将是反斜杠，但在 Linux/WSL 上，将是正斜杠
    # 使用 os.path.normpath 来规范化路径分隔符
    path_zip = os.path.normpath(path_zip.replace("\\", "/"))
    path_img = os.path.normpath(path_img.replace("\\", "/"))
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found" % (path_zip))
        raise
    for i in range(len(_im_zfile)):
        if _im_zfile[i]['path'] == path_zip:
            data = _im_zfile[i]['zipfile'].read(path_img)
            return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    _im_zfile.append({
        'path': path_zip,
        'zipfile': zipfile.ZipFile(path_zip, 'r')
    })
    print(f"Reading image from: {path_img}")
    data = _im_zfile[-1]['zipfile'].read(path_img)
    # if path_img not in _im_zfile[-1]['zipfile'].namelist():
    #     print(f"Error: Key '{path_img}' not found in the ZIP file.")
    #     # 处理错误或提前返回
    # else:
    #     data = _im_zfile[-1]['zipfile'].read(path_img)

    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


def xmlread(filename):
    global _xml_path_zip
    global _xml_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        print("character '@' is not found from the given path '%s'" % (path))
        assert 0
    path_zip = path[0:pos_at]
    path_xml = path[pos_at + 2:]
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found" % (path_zip))
        assert 0
    for i in xrange(len(_xml_path_zip)):
        if _xml_path_zip[i] == path_zip:
            data = _xml_zfile[i].open(path_xml)
            return ET.fromstring(data.read())
    _xml_path_zip.append(path_zip)
    print("read new xml file '%s'" % (path_zip))
    _xml_zfile.append(zipfile.ZipFile(path_zip, 'r'))
    data = _xml_zfile[-1].open(path_xml)
    return ET.fromstring(data.read())
