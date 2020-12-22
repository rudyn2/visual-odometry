from datetime import datetime
import pickle
import cv2 as cv
import xmltodict


if __name__ == '__main__':

    with open('data/2011_09_26/2011_09_26_drive_0001_sync/tracklet_labels.xml') as fd:
        doc = xmltodict.parse(fd.read())