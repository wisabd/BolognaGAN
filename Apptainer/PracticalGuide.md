# Apptainer Commands and Global Options

## Commands

| Command   | Purpose                                         | Example Usage                                         |
|-----------|-------------------------------------------------|------------------------------------------------------|
| `run`     | Executes the container’s `%runscript`          | `apptainer run my_container.sif`                    |
| `exec`    | Runs a specific command inside the container.  | `apptainer exec my_container.sif ls /app`           |
| `shell`   | Opens an interactive shell inside the container.| `apptainer shell my_container.sif`                  |
| `instance`| Manages persistent container instances.        | `apptainer instance start my_container.sif my_inst` |
| `build`   | Builds a container from a definition file or another image.| `apptainer build my_container.sif my_def.def` |
| `pull`    | Downloads a container from a remote source.    | `apptainer pull library://ubuntu:20.04`             |
| `push`    | Uploads a container to a remote library.       | `apptainer push my_container.sif library://`        |
| `inspect` | Displays metadata about a container.           | `apptainer inspect --labels my_container.sif`       |
| `test`    | Runs the `%test` section of a container.       | `apptainer test my_container.sif`                   |
| `help`    | Shows help for a command.                      | `apptainer help build`                              |
| `version` | Displays the current version of Apptainer.     | `apptainer version`                                 |

## Global Options

| Option    | Purpose                                         | Example Usage                                         |
|-----------|-------------------------------------------------|------------------------------------------------------|
| `--help`  | Displays help information for a command.        | `apptainer run --help`                              |
| `--debug` | Enables debug-level logging.                    | `apptainer --debug build my_container.sif`          |
| `--quiet` | Suppresses output messages.                     | `apptainer --quiet exec my_container.sif ls`        |
| `--bind`  | Binds host directories to the container.        | `apptainer exec --bind /data:/mnt my_container.sif` |



