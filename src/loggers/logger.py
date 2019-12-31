class Logger:
    def __init__(self, output_path):
        self.reset(output_path)
    
    def reset(self, output_path):
        self.output_path = output_path
    
    def log(self, out_string, std_out=True):
        if std_out:
            print(out_string)
        f=open(self.output_path, "a+")
        f.write(out_string)
        f.write("\n")
        f.close()
        