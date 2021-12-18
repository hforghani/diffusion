import sys

from cascade.models import Project
from db.managers import MEMMManager

sys.path.append('.')

from settings import logger


def main():
    if len(sys.argv) < 2:
        raise ValueError('project name must be given')
    project_name = sys.argv[1]
    project = Project(project_name)
    manager = MEMMManager(project)
    memms = manager.fetch_all()

    i = 0
    for memm in memms:
        # TODO
        i += 1
        if i % 10000 == 0:
            logger.info(f'{i} documents inserted')


if __name__ == '__main__':
    main()
