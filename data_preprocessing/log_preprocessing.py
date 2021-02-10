"""
Log preprocessing
Create a CSV file from the log.
Each column of this file is a value of the Characteristic

Instruction:
$python log_preprocessing.py path_log path_output

author: nakmuaycoder
date: 01/2021
"""

import struct
import pandas as pd
import argparse
import os
from datetime import datetime
import json


class LogParser(object):
    """Log parser object"""

    def _parse_log(self, path_log):
        """
        :param path_log: path of the file
        """

        with open(path_log, 'r') as f:
            file = f.readlines()

        out = dict()
        out["date"] = []
        date = os.path.basename(path_log).split("_")[2].replace('-', '')  # Date, format %y%m%d
        lbl = 0

        if hasattr(self, "label"):
            out["label"] = []

        for line in file:

            if "Wrote to characteristic 0000{}".format(self.label) in line:
                lbl = line[-12:-1].replace(" ", "")
                try:
                    lbl = bytes.fromhex(lbl)
                    lbl = struct.unpack('i', lbl)[0]
                except:
                    pass

            if 'changed | value:' in line:

                uuid = line.split(" ")[7][4:8]
                if uuid in self.header.keys():

                    try:
                        value = struct.unpack('f', bytes.fromhex(line[-13:-1].replace(" ", "")))[
                            0]  # Conv bytes into float32

                        if uuid in out.keys():
                            out[uuid] += [value]
                        else:
                            out[uuid] = [value]

                        if uuid == '2102':
                            hour = line.split(' ')[3].replace(":", "")  # hour, format %H:%M:%S
                            dte = datetime.strptime(date + ' ' + hour, "%Y%m%d %H%M%S")
                            out['date'] += [dte]

                            if hasattr(self, "label"):
                                out["label"] += [lbl]

                    except:
                        pass

        data = pd.DataFrame.from_dict(out)
        header = {**self.header, **{"label": "label"}}
        data.columns = [header[col] for col in data.columns]
        return data

    def __init__(self, header):
        """
        Instantiation of the LogParser
        :param path_log: path of the log
        :param header: dict, "column": uuid, label is a special value
        """
        if 'label' in header.keys():
            self.label = header["label"]

        self.header = {k: header[k] for k in header.keys() if k != "label"}

    def write_file(self, path_output, mode="w"):
        """
        Save the table as csv file
        :param path_output:
        :param mode:
        :return:
        """
        if not mode.upper() in ["W", "A"]:
            raise ValueError("mode must be 'w' or 'a'")

        if hasattr(self, "data"):
            self.data.to_csv(path_output, mode=mode, index=False)
        else:
            raise AttributeError("You must parse a log first")

    def __call__(self, path_log):
        """append records"""
        file = self._parse_log(path_log)
        if hasattr(self, "data"):
            self.data = self.data.append(file, ignore_index=True)
        else:
            self.data = file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset from the log")
    parser.add_argument("log", type=str, help="path of the log", default=None)
    parser.add_argument("output", type=str, help="path of the dataset", default=None)
    parser.add_argument("uuid", type=str, help="Path of the uuid", default=None)
    parser.add_argument("mode", type=str, help="Overwrite any existing output", default="w")
    args = parser.parse_args()

    inpt = args.log
    output = args.output
    mode = args.mode
    uuid = args.uuid

    with open(uuid, "r") as f:
        header = json.loads(f.read())

    parser = LogParser(header)
    parser(inpt)

    parser.write_file(output, mode=mode)
