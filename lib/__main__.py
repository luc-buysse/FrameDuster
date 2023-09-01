import argparse
import time
from datetime import datetime
import sys
from loguru import logger

# wrap user code in _user_functions
from lib import user_functions, pbar_config
from lib.iterable import _flush
from lib.toppings import _add_decorators, _setup_pbar, _setup_mixed_context
from lib.mongodb import _reset_function, _state_function
from lib.s3 import _delete_field
from lib.local import _update_local, _set_local, _unbind_local, _scan_local
from lib.config import config

import torchvision

torchvision.disable_beta_transforms_warning()


def create_parser():
    parser = argparse.ArgumentParser(prog=config['project_name'])

    subparsers = parser.add_subparsers(title="Subcommands for this program.", dest='subargs')

    # scrap command
    scrap_parser = subparsers.add_parser("scrap", help="Launches a @Scraper service : scrap <name>")
    scrap_parser.add_argument(dest='scrap_args')

    # preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Launches a @Preprocessor service : preprocess <name>")
    preprocess_parser.add_argument(dest='preprocess_args')

    # train command
    train_parser = subparsers.add_parser("train", help="Launches a @Trainer service : train <name>")
    train_parser.add_argument(dest='train_args')

    # import command
    import_parser = subparsers.add_parser("import", help="Launches an @Importer service : import <name>")
    import_parser.add_argument(dest='import_args')

    # pipe command
    pipe_parser = subparsers.add_parser("pipe",
                                        help="Pipes multiple services together : pipe -s <command1> -s <command2> ... ")
    pipe_parser.add_argument('-s',
                             '--sub',
                             nargs="+",
                             action="append",
                             dest='pipe_args')

    # delete command
    delete_parser = subparsers.add_parser("delete",
                                          help="Deletes all s3 files of a dataset in a specific compartment : delete <dataset> <compartment>")
    delete_parser.add_argument(dest='delete_args', nargs='+')
    delete_parser.add_argument('-f', '--full', dest='full', action='store_true')

    # reset command
    reset_parser = subparsers.add_parser("reset",
                                         help="Resets a service's progress : reset (--full) <cmd>")
    reset_parser.add_argument(nargs='+',
                              action="append",
                              dest='reset_args')
    reset_parser.add_argument('-f',
                              '--full',
                              action="store_true",
                              dest='full',
                              required=False)
    reset_parser.add_argument('-d',
                              '--delete',
                              action="store_true",
                              dest='delete',
                              required=False)

    # state command
    state_parser = subparsers.add_parser("state", help="Shows the progress of a service : state <service command>")
    state_parser.add_argument(nargs='+',
                              action="append",
                              dest='state_args')

    # local command
    local_parser = subparsers.add_parser("local", help="""Set of commands to manage local storages :
        ; 'update' to update the database with the current content of your local storage
        ; 'scan' to list all local storages registered in database
        ; 'unbind <machine_id>' to delete all traces of a local storage in database
        ; 'set <new_machine_id>' to change the current storage id
    """)
    local_parser.add_argument(nargs='+',
                              action='append',
                              dest='local_args')

    subparsers.add_parser("terminal", help='Opens a terminal.')

    return parser


def parse_args(args, error_message="Unvalid arguments"):
    if hasattr(args, "scrap_args"):
        if isinstance(args.scrap_args, str):
            try:
                return f'_scrap_{args.scrap_args}'
            except KeyError:
                raise Exception(f"{args.scrap_args} is not a valid scraper name")
        else:
            raise Exception(f"One argument was expected : scrap <name>"
                            f", {len(args.scrap_args)} were given)")
    elif hasattr(args, "preprocess_args"):
        if isinstance(args.preprocess_args, str):
            try:
                return f'_preprocess_{args.preprocess_args}'
            except KeyError:
                raise Exception(f"{args.preprocess_args} is not a valid preprocessor name")
        else:
            raise Exception(f"One argument was expected : preprocess <step>"
                            f", {len(args.preprocess_args)} were given)")
    else:
        raise Exception(error_message)


def main(args):
    if hasattr(args, "pipe_args"):
        _setup_mixed_context()

        # chain the sub-procs
        generator = None
        n_steps = len(args.pipe_args)
        for i, sub_proc_args in enumerate(args.pipe_args):
            function_id = parse_args(arg_parser.parse_args(sub_proc_args),
                                     "Only scrapers and preprocessors can be piped together.")
            sub_proc_wrapped = user_functions[function_id]

            if i == 0 and 'input' in pbar_config[function_id]:
                config['mixed_context']['pbar_input_service'] = function_id
                _setup_pbar('input')
            elif i == n_steps and 'output' in pbar_config[function_id]:
                _setup_pbar('output')

            if generator is None:
                generator = _add_decorators(*sub_proc_wrapped)()
            else:
                generator = _add_decorators(*sub_proc_wrapped)(generator)
        # flush
        for _ in generator:
            pass
    elif hasattr(args, "train_args"):
        if isinstance(args.train_args, str):
            try:
                function_id = f'_train_{args.train_args}'
                _flush(_add_decorators(*user_functions[function_id]))
            except KeyError:
                raise Exception(f"{args.train_args} is not a valid trainer name")
        else:
            raise Exception(f"One argument was expected : train <step>"
                            f", {len(args.train_args)} were given)")

    elif hasattr(args, 'reset_args'):
        if isinstance(args.reset_args[0], str):
            raise Exception('Reset arguments must be a function command')
        else:
            function_id = "".join(['_' + arg for arg in args.reset_args[0]])
            _reset_function(function_id, args.full, args.delete)
    elif hasattr(args, 'state_args'):
        if isinstance(args.state_args[0], str):
            raise Exception('State arguments must be a function command')
        else:
            function_id = "".join(['_' + arg for arg in args.state_args[0]])
            _state_function(function_id)
    elif hasattr(args, 'import_args'):
        if not isinstance(args.import_args, str):
            raise Exception('Argument of import must be the name of the import.')
        try:
            user_functions[args.import_args]()
        except KeyError:
            raise Exception(f"{args.import_args} is not a valid importer name")
    elif hasattr(args, 'delete_args'):
        if isinstance(args.delete_args, str) or len(args.delete_args) > 2:
            raise Exception('Delete command takes two arguments : delete <dataset> <field> (-f/--full)')
        else:
            _delete_field(*args.delete_args, args.full)
    elif hasattr(args, 'local_args'):
        if args.local_args[0][0] == 'update':
            _update_local()
        elif args.local_args[0][0] == 'scan':
            _scan_local()
        elif len(args.local_args[0]) < 2:
            raise Exception('Missing argument for local')
        elif args.local_args[0][0] == 'unbind':
            _unbind_local(args.local_args[0][1])
        elif args.local_args[0][0] == 'set':
            _set_local(args.local_args[0][1])
        else:
            raise Exception('Invalid local arguments')
    elif hasattr(args, 'subargs') and args.subargs == 'terminal':
        run_continuous()
    elif hasattr(args, 'scrap_args') or hasattr(args, 'preprocess_args'):
        # Run a regular @Preprocessor @Scraper service
        _setup_mixed_context()

        function_id = parse_args(args)

        config['mixed_context']['pbar_input_service'] = function_id

        # setup progress bar
        for type in pbar_config[function_id]:
            _setup_pbar(type)

        _flush(_add_decorators(*user_functions[function_id]))
    else:
        logger.warning('Invalid argument, type --help for help.')
        run_continuous()


arg_parser = create_parser()


def run_cmd():
    logger.remove()
    logger.add(sys.stdout, enqueue=True)

    logger.info('**********************************')
    logger.info('*         PROGRAM START          *')
    logger.info('*     ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + '        *')
    logger.info('**********************************')

    main(arg_parser.parse_args())

    logger.success('**********************************')
    logger.success('*          PROGRAM END           *')
    logger.success('*     ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + '        *')
    logger.success('**********************************')


def run_continuous():
    while True:
        try:
            logger.complete()
            cmd = input(f"{config['project_name']}$").split()
            if len(cmd) == 0:
                continue
            if cmd[0] == 'exit':
                break
            main(arg_parser.parse_args(cmd))
            time.sleep(0.2)
        except BaseException as e:
            logger.error(e)
            continue
