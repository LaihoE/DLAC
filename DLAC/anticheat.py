import csv
import time
import sklearn
import joblib
import pandas as pd
import math
import numpy as np
import onnxruntime
import os
from ctypes import *
from ctypes import cdll


# Send paths to Golang
class go_string(Structure):
    _fields_ = [
        ("p", c_char_p),
        ("n", c_int)]


class Model:

    def __init__(self, dem_folder, model_type='small'):
        self.model_type = model_type
        dirname = os.path.dirname(__file__)
        self._parser(dem_folder, dirname)
        # Parser outputs 1x1728 row
        self.scaler = joblib.load(os.path.join(dirname, 'scaler.gz'))
        self.X = pd.read_csv(os.path.join(dirname, 'data/data.csv'), header=None).to_numpy().reshape(-1, 192, 9)
        # Remove file after reading
        os.remove(os.path.join(dirname, 'data/data.csv'))
        # ONNX

        if model_type == 'small':
            self.ort_session = onnxruntime.InferenceSession(os.path.join(dirname, 'models/small.onnx'))
        elif model_type == 'big':
            self.ort_session = onnxruntime.InferenceSession(os.path.join(dirname, 'models/bigger.onnx'))
        else:
            raise FileNotFoundError('Unknown model_type, use "small" or "big"')


    # Calls Go parser
    def _parser(self, dem_folder, dirname):
        lib = cdll.LoadLibrary(os.path.join(dirname, 'parser.so'))
        lib.startparsing.restype = c_char_p
        lib.startparsing(c_char_p(dirname.encode()), c_char_p(dem_folder.encode()))

    def _predict(self, batch_size):
        # Parser outputs other information that isn't used in the prediction and is combined after prediction
        self.file = self.X[:, 0, 0]
        self.name = self.X[:, 0, 1]
        self.id = self.X[:, 0, 2]
        self.tick = self.X[:, 0, 3]

        self.X = np.float32(self.X[:, :, 4:])  # Data that is input into the model.
        # Only the bigger model uses scaling
        if self.model_type == 'big':
            self.X = self.scaler.transform(self.X.reshape(-1, 5)).reshape(-1, 192, 5)
        else:
            # smaller model uses 128 tick and bigger uses 192
            self.X = self.X[:, 192-128:, :]

        total_batches = math.ceil(self.X.shape[0] / batch_size)
        confidence = []
        for batch in range(total_batches):
            # Slice current batch
            data_this_batch = self.X[batch_size * batch: batch_size * batch + batch_size, :, :]
            # ONNX inputs
            ort_inputs = {self.ort_session.get_inputs()[0].name: data_this_batch}
            # Predict
            ort_outs = np.array(self.ort_session.run(None, ort_inputs))
            # index the cheating conf
            cheating_confidence = list(ort_outs[0][:, 1])
            # Add batch to output
            confidence.extend(cheating_confidence)
        self.confidence = np.array(confidence)

    def predict_to_terminal(self, threshold=0.95, batch_size=1000):
        self._predict(batch_size)     # Does the actual prediction
        # Loop trough each shot in the game
        print("Name", "Confidence", "ID", "File")
        for shot in range(len(self.confidence)):
            if self.confidence[shot] > threshold:
                print(self.name[shot], self.confidence[shot], self.id[shot], self.file[shot])

    def predict_to_csv(self, out_file, threshold=0.95, batch_size=1000):
        self._predict(batch_size)     # Does the actual prediction
        # Create headers for csv
        with open(out_file, 'w', newline='\n')as f:
            thewriter = csv.writer(f)
            thewriter.writerow(["name", "confidence", "id", "file"])
        # Loop trough each shot in the game
        for shot in range(len(self.confidence)):
            if self.confidence[shot] > threshold:
                with open(out_file,'a',newline='\n')as f:
                    thewriter = csv.writer(f)
                    thewriter.writerow([self.name[shot], self.confidence[shot], self.id[shot], self.file[shot]])

    def predict_to_list(self, threshold=0.95, batch_size=1000):
        # Returns a list for each shot, so the output is a list of lists
        outputs = []
        self._predict(batch_size)     # Does the actual prediction
        # Loop trough each shot in the game
        for shot in range(len(self.confidence)):
            if self.confidence[shot] > threshold:
                outputs.append([self.name[shot], self.confidence[shot], self.id[shot], self.file[shot]])
        return outputs

    def predict_total_detections_to_dict(self, batch_size=1000):
        output_dict = {}
        self._predict(batch_size)  # Does the actual prediction
        # Loop trough each shot in the game
        for shot in range(len(self.confidence)):
            if self.name[shot] not in output_dict:
                output_dict[self.name[shot]] = 1
            else:
                output_dict[self.name[shot]] += 1
        out = {k: v for k, v in sorted(output_dict.items(),reverse=True, key=lambda item: item[1])}
        return out

    def predict_averages_to_dict(self, batch_size=1000):
        output_dict = {}
        self._predict(batch_size)  # Does the actual prediction
        # Loop trough each shot in the game
        for shot in range(len(self.confidence)):
            if self.name[shot] not in output_dict:
                output_dict[self.name[shot]] = [self.confidence[shot]]
            else:
                output_dict[self.name[shot]].append(self.confidence[shot])
        out = {k: sum(v) / len(v) for k, v in sorted(output_dict.items(), key=lambda item: item[1])}
        out = {k: v for k, v in sorted(out.items(),reverse=True ,key=lambda item: item[1])}
        return out