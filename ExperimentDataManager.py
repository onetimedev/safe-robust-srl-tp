import os
class ExperimentDataManager:

    """
    Experiment data manager is used to create CSV data files, and write experiment data to those files.
    The constructor requires a directory string, a filename string, and a data header for the CSV file.
    """
    def __init__(self, directory: str, filename: str, data_header: str):

        # Create the directory if not exists.
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = directory + "/" + filename
        if not os.path.isfile(filepath):
            f = open(filepath, "w")
            f.close()

        f = open(filepath, "w")
        f.write(data_header + "\n")
        f.close()
        self.filepath = filepath

    def append_data_entry(self, entry):
        f = open(self.filepath, "a")
        f.write(entry + "\n")
        f.close()
