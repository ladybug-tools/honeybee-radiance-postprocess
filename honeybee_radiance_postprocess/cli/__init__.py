"""Modules for post-processing simulation outputs."""
import click
import logging

from honeybee.cli import main
from .mtxop import mtxop
from .translate import translate
from .grid import grid
from .postprocess import post_process


_logger = logging.getLogger(__name__)


# command group for all postprocess extension commands.
@click.group(help='honeybee radiance postprocess commands.')
@click.version_option()
def postprocess():
    pass


# add sub-commands to postprocess
postprocess.add_command(mtxop)
postprocess.add_command(translate)
postprocess.add_command(grid)
postprocess.add_command(post_process, name='post-process')

# add postprocess sub-commands to honeybee CLI
main.add_command(postprocess)
