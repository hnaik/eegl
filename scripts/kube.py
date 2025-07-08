import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from tempfile import NamedTemporaryFile

from egr.log import init_logging
from jinja2 import (
    Environment,
    select_autoescape,
    FileSystemLoader,
)

LOG = logging.getLogger('kube')

MACHINE_CLASS = {
    'V100': 'Tesla-V100-PCIE-32GB',
    'T4': 'Tesla-T4',
}


def get_kube_op(name: str) -> str:
    if name == 'start':
        return 'apply'
    elif name == 'stop':
        return 'delete'
    raise RuntimeError(f'Unknown op {name}')


def main(args):
    loader = FileSystemLoader(args.search_path)
    env = Environment(loader=loader, autoescape=select_autoescape())
    template = env.get_template(args.template)

    optional_args = {}
    if args.input_file:
        optional_args['input_file'] = args.input_file
    yaml_tmp: str = template.render(
        job_id=args.job_id,
        machine_class=MACHINE_CLASS[args.machine_class],
        num_gpus=args.num_gpus,
        docker_image=args.docker_image,
        **optional_args,
    )

    command_bin = (
        '/usr/bin/kubectl'
        if Path('/usr/bin/kubectl').exists()
        else '/usr/local/bin/kubectl'
    )
    op_name: str = get_kube_op(args.op)

    kwargs = {}
    if args.keep_config:
        kwargs['delete'] = False
    with NamedTemporaryFile(mode='w+b', suffix='.yml', **kwargs) as tf:
        tf.write(yaml_tmp.encode('utf-8'))
        tf.seek(0)
        cmd = [command_bin, op_name, '-f', tf.name]
        LOG.info(' '.join(cmd))
        try:
            subprocess.run(cmd, capture_output=True)
            LOG.info('%s succeeded', args.op)
        except subprocess.CalledProcessError as err:
            LOG.info('%s', err)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-l',
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
    )
    parser.add_argument(
        '--machine-class', default='V100', choices=MACHINE_CLASS.keys()
    )
    parser.add_argument('--job-id', type=str, required=True)
    parser.add_argument('--op', required=True, choices=['start', 'stop'])
    parser.add_argument('--search-path', type=str, default='cr')
    parser.add_argument('--template', type=str, default='pod-template.yml')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument(
        '--docker-image', type=str, default='hnaik2/cuda-user:latest'
    )
    parser.add_argument('--input-file', type=str)
    parser.add_argument('--keep-config', action='store_true', default=False)

    args = parser.parse_args()
    init_logging(args.log_level)
    main(args)
