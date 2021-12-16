# -*- coding: utf-8 -*-
import argparse
import logging
import traceback
import time
from cascade.models import Project
from mln.file_generators import FileCreator, PracmlnCreator, Alchemy2Creator
import settings


logging.basicConfig(format=settings.LOG_FORMAT)
logger = logging.getLogger('createevidence')
logger.setLevel(settings.LOG_LEVEL)


class Command:
    help = ''

    CREATORS = {
        FileCreator.FORMAT_PRACMLN: PracmlnCreator,
        FileCreator.FORMAT_ALCHEMY2: Alchemy2Creator
    }

    def add_arguments(self, parser):
        parser.add_argument(
            "-p",
            "--project",
            type=str,
            dest="project",
            help="project name",
        )
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            dest="format",
            default=FileCreator.FORMAT_PRACMLN,
            help="format of files. Valid values are '{}'. The default value is '{}'.".format(
                "', '".join(self.CREATORS.keys()),
                FileCreator.FORMAT_PRACMLN
            )
        )
        parser.add_argument(
            "-s",
            "--set",
            type=str,
            dest="set",
            help="create evidence for training set or test set; valid values are 'train' or 'test'"
        )
        parser.add_argument(
            "-m",
            "--multiple",
            action="store_true",
            dest="multiple",
            help="create multiple test evidence files each for a cascade"
        )

    def __init__(self):
        super(Command, self).__init__()

    def handle(self, args):
        try:
            start = time.time()

            # Get project or raise exception.
            project_name = args.project
            if project_name is None:
                raise Exception('project not specified')
            project = Project(project_name)

            # Validate set option.
            if args.set not in ['train', 'test', None]:
                raise 'invalid set "%s"' % args.set

            creator_clazz = self.CREATORS[args.format]
            creator = creator_clazz(project)
            creator.create_evidence(args.set, args.multiple)

            logger.info('command done in %f min' % ((time.time() - start) / 60))

        except:
            logger.info(traceback.format_exc())
            raise


if __name__ == '__main__':
    c = Command()
    parser = argparse.ArgumentParser(c.help)
    c.add_arguments(parser)
    args = parser.parse_args()
    c.handle(args)
