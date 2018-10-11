"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

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
