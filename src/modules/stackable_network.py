r"""
stackable_network.py
====================
Interface class that is used in classes that implement the
specification necessary to be used as a horizontal training layer.

.. autosummary::
  modules.StackableNetwork
"""

class StackableNetwork(object):
  r"""
  Interface, not a Class! Do not implement anything here
  Classes that inherit from StackableNetwork are required,
  (in python by convention, not enforced) to provide an
  implementation of these functions.
  """

  def __init__(self):
    pass


  def calculate_upstream(self, previous_network):
    r"""
    Calculate the output to the point where the
    map function will be applied after.

    Make sure those layers appear in parameters()
    """
    raise "Provide an upstream function"
  