"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""


import visdom
import numpy as np
vis = visdom.Visdom()
vis.text('Hello, world!')
vis.image(np.ones((3, 10, 10)))





class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)










from pycrayon import CrayonClient
import time

# Connect to the server
cc = CrayonClient(hostname="server_machine_address")

# Create a new experiment
foo = cc.create_experiment("foo")

# Send some scalar values to the server
foo.add_scalar_value("accuracy", 0, wall_time=11.3)
foo.add_scalar_value("accuracy", 4, wall_time=12.3)
# You can force the time and step values
foo.add_scalar_value("accuracy", 6, wall_time=13.3, step=4)

# Get the datas sent to the server
foo.get_scalar_values("accuracy")
# >>> [[11.3, 0, 0.0], [12.3, 1, 4.0], [13.3, 4, 6.0]])

# backup this experiment as a zip file
filename = foo.to_zip()

# delete this experiment from the server
cc.remove_experiment("foo")
# using the `foo` object from now on will result in an error

# Create a new experiment based on foo's backup
bar = cc.create_experiment("bar", zip_file=filename)

# Get the name of all scalar plots in this experiment
bar.get_scalar_names()
# >>> ["accuracy"]

# Get the data for this experiment
bar.get_scalar_values("accuracy")
# >>> [[11.3, 0, 0.0], [12.3, 1, 4.0], [13.3, 4, 6.0]])
