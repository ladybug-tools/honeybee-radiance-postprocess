"""Modules for post-processing simulation outputs."""
import click
import sys
import logging
import json

from honeybee.cli import main
from .mtxop import mtxop
from .translate import translate


_logger = logging.getLogger(__name__)


# command group for all radiance extension commands.
@click.group(help='honeybee radiance commands.')
@click.version_option()
def radiance():
    pass


# add sub-commands to radiance
radiance.add_command(mtxop)
radiance.add_command(translate)

# add radiance sub-commands to honeybee CLI
main.add_command(radiance)
