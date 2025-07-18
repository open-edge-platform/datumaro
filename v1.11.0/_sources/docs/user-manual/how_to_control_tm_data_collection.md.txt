# How to control telemetry data collection

The [OpenVINO™ telemetry library](https://github.com/openvinotoolkit/telemetry/)
is used to collect basic information about Datumaro usage.

A short description of the information collected:
| Event             | Description |
| ----------------- | ----------- |
| version           | Datumaro version |
| session start/end | Accessory event, there is no additional info here |
| {cli_command}_result | Datumaro command result with arguments passed* |
| error | Stack trace in case of exception* |

> \* All sensitive arguments, such as filesystem paths or names, are sanitized

To enable the collection of telemetry data, the ISIP consent file
must exist and contain `1`, otherwise telemetry will be disabled.
The ISIP file can be created/modified by an OpenVINO installer
or manually and used by other OpenVINO™ tools.

The location of the ISIP consent file depends on the OS:
- Windows: `%localappdata%\Intel Corporation\isip`,
- Linux: `$HOME/intel/isip`.
