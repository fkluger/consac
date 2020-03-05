import sys


class Tee(object):
    def __init__(self, name, mode, file_only=False):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        self.file_only = file_only

    def __del__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        self.file.write(data)
        if not self.file_only:
            self.stdout.write(data)

    def flush(self):
        self.file.flush()
