# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import os
import os.path as osp
import re
import sys
import traceback

from datumaro.cli import commands, contexts
from datumaro.util import telemetry_stub
from datumaro.util.os_util import is_subpath

try:
    import openvino_telemetry as tm
except ImportError:
    tm = telemetry_stub

NO_TELEMETRY_KEY = "DATUMARO_NO_OV_TELEMETRY"


def _get_action_name(command):
    # TODO: We should clean these nonsense if-else branches.
    if command is commands.require_project.versioning.info.info_command:
        return "project_info_result"
    elif command is commands.stats.stats_command:
        return "project_stats_result"
    elif command is contexts.project.migrate_command:
        return "project_migrate_result"
    elif command is commands.require_project.modification.export.export_command:
        return "project_export_result"
    elif command is commands.validate.validate_command:
        return "project_validate_result"
    elif command is commands.filter.filter_command:
        return "project_filter_result"
    elif command is commands.transform.transform_command:
        return "project_transform_result"
    elif command is commands.require_project.modification.import_.import_command:
        return "source_add_result"
    elif command is commands.require_project.modification.remove.remove_command:
        return "source_remove_result"
    elif command is contexts.source.info_command:
        return "source_info_result"
    elif command is contexts.model.add_command:
        return "model_add_result"
    elif command is contexts.model.remove_command:
        return "model_remove_result"
    elif command is contexts.model.run_command:
        return "model_run_result"
    elif command is contexts.model.info_command:
        return "model_info_result"
    elif command is commands.require_project.versioning.checkout.checkout_command:
        return "checkout_result"
    elif command is commands.require_project.versioning.commit.commit_command:
        return "commit_result"
    elif command is commands.convert.convert_command:
        return "convert_result"
    elif command is commands.require_project.modification.create.create_command:
        return "create_result"
    elif command is commands.compare.compare_command:
        return "diff_result"
    elif command is commands.explain.explain_command:
        return "explain_result"
    elif command is commands.info.info_command:
        return "info_result"
    elif command is commands.require_project.versioning.log.log_command:
        return "log_result"
    elif command is commands.merge.merge_command:
        return "merge_result"
    elif command is commands.patch.patch_command:
        return "patch_result"
    elif command is commands.require_project.versioning.status.status_command:
        return "status_result"

    return f"{command.__name__}_result"


ARG_USED_FLAG = "True"


def _cleanup_params_info(args, sensitive_args):
    fields_to_exclude = (
        "command",
        "_positionals",
    )
    cli_params = {}
    for arg in vars(args):
        if arg in fields_to_exclude:
            continue
        arg_value = getattr(args, arg)
        if arg in sensitive_args:
            # If a command line argument is a file path, it must not be sent,
            # because it can contain confidential information.
            # A placeholder value is used instead.
            cli_params[arg] = str(ARG_USED_FLAG)
        else:
            cli_params[arg] = str(arg_value)
    return cli_params


def _cleanup_stacktrace():
    installation_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))

    def clean_path(match):
        file_path = match.group(1)
        if is_subpath(file_path, base=installation_dir):
            file_path = osp.relpath(file_path, installation_dir)
        else:
            file_path = osp.basename(file_path)

        return f'File "{file_path}"'

    exc_type, _, exc_traceback = sys.exc_info()
    tb_lines = traceback.format_list(traceback.extract_tb(exc_traceback))
    tb_lines = [re.sub(r'File "([^"]+)"', clean_path, line, count=1) for line in tb_lines]

    return exc_type.__name__, "".join(tb_lines)


def init_telemetry_session(app_name, app_version):
    no_telemetry_flag = os.environ.get(NO_TELEMETRY_KEY, None)

    if no_telemetry_flag is None:
        telemetry = tm.Telemetry(app_name=app_name, app_version=app_version, tid="UA-17808594-29")
        telemetry.start_session("dm")
        send_version_info(telemetry, app_version)
        return telemetry

    return telemetry_stub.Telemetry(
        app_name=app_name, app_version=app_version, tid="UA-17808594-29"
    )


def close_telemetry_session(telemetry):
    telemetry.end_session("dm")
    telemetry.force_shutdown(1.0)


def _send_result_info(result, telemetry, args, sensitive_args):
    payload = {
        "status": result,
        **_cleanup_params_info(args, sensitive_args),
    }
    action = _get_action_name(args.command)
    telemetry.send_event("dm", action, json.dumps(payload))


def send_version_info(telemetry, version):
    telemetry.send_event("dm", "version", str(version))


def send_command_success_info(telemetry, args, *, sensitive_args):
    _send_result_info("success", telemetry, args, sensitive_args)


def send_command_failure_info(telemetry, args, *, sensitive_args):
    _send_result_info("failure", telemetry, args, sensitive_args)


def send_command_exception_info(telemetry, args, *, sensitive_args):
    _send_result_info("exception", telemetry, args, sensitive_args)
    send_error_info(telemetry, args, sensitive_args)


def send_error_info(telemetry, args, sensitive_args):
    exc_type, stack_trace = _cleanup_stacktrace()
    payload = {
        "exception_type": exc_type,
        "stack_trace": stack_trace,
        **_cleanup_params_info(args, sensitive_args),
    }

    telemetry.send_event("dm", "error", json.dumps(payload))
