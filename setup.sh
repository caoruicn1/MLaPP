# add the top level directory to the python module search path.  by
# sourcing this BASH script
#
#   $ . setup.sh
#
# appending the top level to sys.path is not necessary for proper
# module imports.

DIR="`dirname $BASH_SOURCE`"
ABSDIR="`cd $DIR; pwd`"
export PYTHONPATH="$ABSDIR:$PYTHONPATH"
