# import necessary pacakges
import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        # check to see if outputPath exists. if so, raise an exception
        if os.path.exists(outputPath):
            raise ValueError("the supplied 'outputPath' already exists"
                "and cannot be overwritten . Manually delete the file"
                "before continuing", outputPath)

        # open HDF5 database for writing and create two datasets
        # one to store the images/features and other to store the class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey,dims,dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],),dtype="int")
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels":[]}
        self.idx = 0

    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check  to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # write buffers to disk and reset the buffers
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx= i
        self.buffer = {"data":[],"labels":[]}

    def close(self):
        # check to see if there are any other entries in the budder
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()
