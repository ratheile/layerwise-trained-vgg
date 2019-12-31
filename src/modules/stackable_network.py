class StackableNetwork(object):

  def __init__(self):
    pass

  """
  Interface, not a Class! Do not implement anything here
  Classes that inherit from StackableNetwork are required,
  (in python by convention, not enforced) to provide an
  implementation of these functions.
  """

  def calculate_upstream(self, previous_network):
    """
    Calculate the output to the point where the
    map function will be applied after.

    Make sure those layers appear in parameters()
    """
    raise "Provide an upstream function"
  