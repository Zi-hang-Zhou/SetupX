"""Docker interaction layer (per blueprint section 1.1).

Manages container lifecycle, command execution, snapshots and rollback.

This module is the Setup Agent's core gateway to Docker:
- container creation / destruction (create_container / destroy)
- attaching to external containers (from_container / from_dockerfile)
- in-container command execution (exec_run)
- file I/O (read_file / write_file)
- environment variables (set_env / get_env)
- snapshot / rollback (create_checkpoint / rollback_to_checkpoint)

Snapshot model:
  We use `docker commit` to save the current container state as an image, then
  recreate the container from the image on rollback. Tags are kept in a LIFO
  stack; rollback pops the top tag.
"""

import re

import docker
from docker.models.containers import Container
from docker.errors import DockerException, ImageNotFound

from .config import get_config, DockerConfig
from .logger import get_logger
from .models import CommandResult

logger = get_logger("docker")

# Output truncation thresholds (per blueprint 4.1: head 1000 + tail 1000 chars).
# Container output can be very long (e.g., compile logs); truncating preserves
# the most informative head and tail.
MAX_HEAD_LENGTH = 1000
MAX_TAIL_LENGTH = 1000


def truncate_output(text: str) -> tuple[str, bool]:
    """Truncate long output (keep head 1000 + tail 1000 chars).

    Returns:
        (possibly truncated text, whether truncation happened).
    """
    if len(text) <= MAX_HEAD_LENGTH + MAX_TAIL_LENGTH:
        return text, False

    head = text[:MAX_HEAD_LENGTH]
    tail = text[-MAX_TAIL_LENGTH:]
    return f"{head}\n...[Truncated {len(text) - MAX_HEAD_LENGTH - MAX_TAIL_LENGTH} chars]...\n{tail}", True


class EnvironmentManager:
    """Docker environment manager (per blueprint section 1.1).

    Handles all Agent <-> Docker interactions. Three init modes:
    1. create_container():  spin up from a base image and clone the repo.
    2. from_container():    attach to an existing container (e.g., produced by an external tool).
    3. from_dockerfile():   build an image from a Dockerfile, then start a container.
    """

    def __init__(self):
        self._client = docker.from_env()  # connect to the Docker daemon via env
        self._container: Container | None = None
        self._config = get_config().docker  # docker config (image, work_dir, timeout)
        self._env_vars: dict[str, str] = {}  # vars set via set_env()
        self._repo_dir_override: str | None = None  # work_dir override from attach()
        # Snapshot stack (LIFO) of commit tags; rollback pops the top.
        self._history_snapshots: list[str] = []
        # _repo_subdir flag: drives the default working directory for exec_run.
        # True  = create_container() path: repo at {work_dir}/repo, default exec_run cd's to it.
        # False = from_dockerfile()/from_container() path: repo lives directly under work_dir.
        self._repo_subdir = True
        # Clean up checkpoint images left from a previous crash.
        self._cleanup_stale_checkpoints()

    def _cleanup_stale_checkpoints(self) -> None:
        """Remove `setup_agent_checkpoint` images left over from a previous crash."""
        try:
            stale = self._client.images.list(name="setup_agent_checkpoint")
            if not stale:
                return
            logger.info(f"found {len(stale)} stale checkpoint images; cleaning...")
            for img in stale:
                try:
                    self._client.images.remove(img.id, force=True)
                    logger.debug(f"removed stale image: {img.tags}")
                except DockerException as e:
                    logger.warning(f"failed to remove stale image {img.tags}: {e}")
            logger.info("stale checkpoint cleanup done")
        except DockerException as e:
            logger.warning(f"failed to query stale checkpoint images: {e}")

    @classmethod
    def _create_with_work_dir(cls, work_dir: str) -> "EnvironmentManager":
        """Internal factory: build an instance with a custom work_dir.

        Used by from_container() / from_dockerfile() where work_dir may differ
        from the default config. Skips __init__ via __new__ and rebuilds the
        DockerConfig (frozen dataclass) with the custom work_dir.
        """
        instance = cls.__new__(cls)
        instance._client = docker.from_env()
        instance._container = None
        instance._env_vars = {}
        instance._repo_dir_override = None
        instance._history_snapshots = []
        instance._repo_subdir = False  # external container / Dockerfile: do not append /repo
        orig = get_config().docker
        instance._config = DockerConfig(
            base_image=orig.base_image,
            work_dir=work_dir,
            timeout=orig.timeout,
        )
        return instance

    @classmethod
    def from_container(cls, container_id: str, work_dir: str = "/workspace") -> "EnvironmentManager":
        """Attach to an existing Docker container (no new container created).

        Used by the Verifier to independently check containers produced by
        external tools (OpenHands / Repo2Run / ...), fully decoupled from the
        Setup Agent's container creation path.

        Args:
            container_id: target container ID or name (short ID accepted).
            work_dir: in-container project root (default /workspace; Repo2Run uses /repo).
        """
        instance = cls._create_with_work_dir(work_dir)
        instance._container = instance._client.containers.get(container_id)
        logger.info(f"attached to existing container: {container_id[:12]}, work_dir={work_dir}")
        return instance

    @classmethod
    def from_dockerfile(cls, dockerfile_dir: str, work_dir: str = "/repo") -> "EnvironmentManager":
        """Build an image from a Dockerfile, start a container, and attach.

        Used to verify whether a Dockerfile produced by tools like Repo2Run
        actually configures the environment correctly.
        Flow: build image -> run sleep-infinity container -> return attached instance.
        """
        instance = cls._create_with_work_dir(work_dir)
        logger.info(f"building image from dockerfile dir: {dockerfile_dir}")
        # rm=True deletes intermediate layers; network_mode=host lets the build
        # access host networking (e.g., to download deps).
        image, build_logs = instance._client.images.build(
            path=dockerfile_dir,
            rm=True,
            network_mode="host",
        )
        for chunk in build_logs:
            if "stream" in chunk:
                line = chunk["stream"].strip()
                if line:
                    logger.debug(f"[build] {line}")
        logger.info(f"image built: {image.id[:12]}")

        # Start a sleep-infinity container so exec_run has a target.
        instance._container = instance._client.containers.run(
            image.id,
            command="sleep infinity",
            detach=True,
            network_mode="host",
        )
        logger.info(f"container started: {instance._container.id[:12]}, work_dir={work_dir}")
        return instance

    @property
    def container(self) -> Container:
        """The current Container; raises if not yet initialized."""
        if self._container is None:
            raise RuntimeError("container not initialized; call create_container() first")
        return self._container

    @property
    def container_id(self) -> str | None:
        """Container ID, or None if no container."""
        return self._container.id if self._container else None

    @property
    def image_name(self) -> str:
        """Configured base-image name."""
        return self._config.base_image

    @property
    def history_snapshots(self) -> list[str]:
        """A copy of the snapshot tag stack (defensive copy)."""
        return self._history_snapshots.copy()

    def attach(self, container_id: str, repo_dir: str | None = None) -> None:
        """Take over an existing container (no creation, no clone).

        Args:
            repo_dir: full in-container path to the repo; if set, exec_run cd's here by default.
        """
        self._container = self._client.containers.get(container_id)
        if repo_dir:
            self._repo_dir_override = repo_dir
        logger.info(f"attached to container: {self._container.id[:12]}, repo_dir={repo_dir or 'default'}")

    def create_container(self, repo_url: str) -> str:
        """Create and start a container, then clone the repo.

        Used by the Setup Agent's main flow.
        Flow: pull base image -> create container -> install git -> clone repo -> initial snapshot.
        """
        logger.info(f"creating container, base image: {self._config.base_image}")

        # Pull base image if not present locally.
        try:
            self._client.images.get(self._config.base_image)
        except ImageNotFound:
            logger.info(f"pulling image: {self._config.base_image}")
            self._client.images.pull(self._config.base_image)

        # Create + start the container.
        # network_mode=host avoids the bridge-network DNS issues docker has by default.
        self._container = self._client.containers.run(
            self._config.base_image,
            command="sleep infinity",
            detach=True,
            working_dir=self._config.work_dir,
            environment=self._env_vars.copy(),
            network_mode="host",
        )

        logger.info(f"container created: {self._container.id[:12]}")

        # Install basic tools (git) and clone the repo.
        self._setup_container(repo_url)

        return self._container.id

    def _setup_container(self, repo_url: str) -> None:
        """Bootstrap the container: install git, clone the repo, take the initial snapshot."""
        logger.info("bootstrapping container...")

        root = "/"  # run bootstrap commands from / so we are not subject to a missing work_dir

        # Install git if missing.
        result = self.exec_run(
            "which git || (apt-get update && apt-get install -y git)",
            work_dir=root,
        )
        if not result.success:
            raise RuntimeError(f"failed to install git: {result.stderr}")

        # Make sure the work_dir exists.
        self.exec_run(f"mkdir -p {self._config.work_dir}", work_dir=root)

        # Clone the repo (with retries for flaky networks).
        logger.info(f"cloning repo: {repo_url}")
        # HTTP/1.1 avoids HTTP2 framing issues behind some proxies.
        self.exec_run("git config --global http.version HTTP/1.1", work_dir=root)
        # Repo lands in {work_dir}/repo.
        clone_cmd = f"git clone --depth=1 {repo_url} {self._config.work_dir}/repo"
        result = None
        for attempt in range(3):
            result = self.exec_run(clone_cmd, work_dir=root)
            if result.success:
                break
            logger.warning(f"clone attempt {attempt+1} failed; retrying...")
        if not result.success:
            raise RuntimeError(f"clone failed: {result.stderr}")

        # Sanity check: print the repo dir.
        result = self.exec_run("pwd")
        logger.info(f"repo dir: {result.stdout.strip()}")

        # Initial snapshot: a clean post-clone state. Repeated rollbacks can ultimately
        # land here.
        self.create_checkpoint("initial_clone")

        logger.info("container bootstrap done")

    def exec_run(
        self,
        command: str,
        timeout: int | None = None,
        work_dir: str | None = None,
    ) -> CommandResult:
        """Execute a command in the container (per blueprint 1.1).

        Steps:
        1. Resolve the working directory (default driven by _repo_subdir).
        2. Build a `bash -c`, wrap with `timeout` to bound runtime.
        3. Execute via the Docker SDK.
        4. Decode and truncate output.
        5. Return CommandResult.
        """
        if timeout is None:
            timeout = self._config.timeout

        # Resolve work_dir.
        if work_dir is None:
            if self._repo_dir_override:
                work_dir = self._repo_dir_override
            elif self._repo_subdir:
                # create_container() path: repo at {work_dir}/repo
                work_dir = f"{self._config.work_dir}/repo"
            else:
                # from_dockerfile / from_container path: repo directly under work_dir
                work_dir = self._config.work_dir

        # Build the actual command: cd then run.
        if work_dir == "/":
            exec_command = command
        else:
            exec_command = f"cd {work_dir} && {command}"

        logger.debug(f"exec: {command}")

        # `timeout N` bounds runtime; demux=True splits stdout/stderr;
        # `environment` injects vars set via set_env().
        exit_code, output = self.container.exec_run(
            cmd=["timeout", str(timeout), "bash", "-c", exec_command],
            demux=True,
            environment=self._env_vars if self._env_vars else None,
        )

        stdout_raw = (output[0] or b"").decode("utf-8", errors="replace")
        stderr_raw = (output[1] or b"").decode("utf-8", errors="replace")

        stdout, stdout_truncated = truncate_output(stdout_raw)
        stderr, stderr_truncated = truncate_output(stderr_raw)
        truncated = stdout_truncated or stderr_truncated

        result = CommandResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            truncated=truncated,
        )

        logger.debug(f"result: exit_code={exit_code}, truncated={truncated}")
        if not result.success:
            logger.warning(f"command failed: {command}, exit_code={exit_code}")

        return result

    def read_file(self, path: str) -> str:
        """Read a file in the container; raise on failure."""
        result = self.exec_run(f"cat {path}")
        if not result.success:
            raise RuntimeError(f"read_file failed for {path}: {result.stderr}")
        return result.stdout

    def write_file(self, path: str, content: str) -> bool:
        """Write a file in the container."""
        # Escape backslashes and single-quotes to prevent shell injection.
        escaped_content = content.replace("\\", "\\\\").replace("'", "'\\''")
        result = self.exec_run(f"echo '{escaped_content}' > {path}")
        return result.success

    def set_env(self, key: str, value: str) -> None:
        """Set an environment variable.

        Stored in _env_vars; injected into every subsequent exec_run via Docker's
        native `environment` parameter — visible to the bash process and all child
        processes. This is more robust than a shell prefix (KEY=VAL cmd), which
        does not survive `&&` chains.

        Note: Docker's `environment` does not expand shell refs (e.g. $PATH);
        we expand them here against the actual container values before storing.
        """
        resolved_value = self._resolve_env_refs(value)
        self._env_vars[key] = resolved_value
        logger.info(f"set_env: {key}={resolved_value}")

    def _resolve_env_refs(self, value: str) -> str:
        """Expand $VAR / ${VAR} references against in-container values."""
        pattern = re.compile(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?')
        refs = pattern.findall(value)
        if not refs:
            return value

        for var_name in set(refs):
            # Prefer already-set vars; otherwise read from the container.
            actual = self._env_vars.get(var_name)
            if actual is None:
                result = self.container.exec_run(
                    cmd=["bash", "-c", f"echo -n ${var_name}"],
                    demux=True,
                )
                actual = (result[1][0] or b"").decode("utf-8", errors="replace").strip()
            value = value.replace(f"${{{var_name}}}", actual)
            value = value.replace(f"${var_name}", actual)
        return value

    def get_env(self, key: str) -> str | None:
        """Return the stored value for `key`, or None."""
        return self._env_vars.get(key)

    def get_env_snapshot(self) -> str:
        """Container-environment snapshot for the prosecutor / judge to quickly
        read the current container state."""
        cmd = (
            "echo '--- python interpreter ---' && "
            "which python3 python 2>/dev/null; python3 --version 2>/dev/null; "
            "echo '--- virtualenv ---' && "
            "ls -d */venv */env */.venv venv .venv env 2>/dev/null || echo '(none found)'; "
            "echo \"VIRTUAL_ENV=$VIRTUAL_ENV\"; "
            "echo \"CONDA_PREFIX=$CONDA_PREFIX\"; "
            "echo '--- PATH ---' && echo \"$PATH\"; "
            "echo '--- project marker files ---' && "
            "ls pyproject.toml setup.py setup.cfg requirements.txt "
            "CMakeLists.txt Makefile configure meson.build "
            "package.json Cargo.toml go.mod pom.xml build.gradle 2>/dev/null || echo '(none)'; "
            "echo '--- working dir ---' && pwd && ls | head -30"
        )
        result = self.exec_run(cmd, timeout=15)
        if result.success:
            return result.stdout[:2000]
        return f"(snapshot failed: exit_code={result.exit_code})"

    # ========================================================================
    # Snapshot / rollback
    # `docker commit` saves the current state as an image; rollback recreates
    # the container from the image.
    # ========================================================================

    def create_checkpoint(self, tag: str) -> str:
        """Create a snapshot (per blueprint 1.1).

        `docker commit`s the current container into a `setup_agent_checkpoint:tag`
        image and pushes the tag onto the snapshot stack.
        """
        logger.info(f"create_checkpoint: {tag}")

        image = self.container.commit(repository="setup_agent_checkpoint", tag=tag)

        self._history_snapshots.append(tag)
        logger.info(f"checkpoint created: {image.id[:12]}, stack depth: {len(self._history_snapshots)}")

        return image.id

    def rollback_to_checkpoint(self, n_frames: int = 1) -> bool:
        """Roll back `n_frames` frames.

        Pops `n_frames` tags off the snapshot stack and recreates the container
        from the last-popped tag. If `n_frames` exceeds the stack depth, pops
        all frames and lands on the earliest recoverable state.

        Steps:
        1. Pop n tags; target = the last popped one.
        2. Stop + remove the current container.
        3. Recreate from the target snapshot image.
        4. Carry over the env vars.
        """
        if not self._history_snapshots:
            logger.warning("no snapshots; cannot rollback")
            return False

        if n_frames < 1:
            n_frames = 1
        actual_pop = min(n_frames, len(self._history_snapshots))
        for _ in range(actual_pop):
            tag = self._history_snapshots.pop()
        logger.info(
            f"rollback to checkpoint: {tag} (popped {actual_pop}/{n_frames}, "
            f"remaining stack depth {len(self._history_snapshots)})"
        )

        # Stop + delete the current container.
        old_container_id = self.container.id
        self.container.stop()
        self.container.remove()
        logger.debug(f"removed old container: {old_container_id[:12]}")

        # Recreate from the snapshot image.
        image_name = f"setup_agent_checkpoint:{tag}"
        self._container = self._client.containers.run(
            image_name,
            command="sleep infinity",
            detach=True,
            working_dir=self._config.work_dir,
            environment=self._env_vars.copy(),
        )

        logger.info(f"rolled back; new container: {self._container.id[:12]}")
        return True

    def list_checkpoints(self) -> list[str]:
        """List all snapshot tags (defensive copy)."""
        return self._history_snapshots.copy()

    def cleanup_snapshots(self) -> None:
        """Delete only the snapshot images (free disk); leave the container running."""
        logger.info("cleaning up snapshot images...")
        for tag in self._history_snapshots:
            try:
                image_name = f"setup_agent_checkpoint:{tag}"
                self._client.images.remove(image_name, force=True)
                logger.debug(f"removed snapshot image: {tag}")
            except DockerException as e:
                logger.warning(f"failed to remove snapshot image {tag}: {e}")
        self._history_snapshots.clear()
        logger.info("snapshot image cleanup done")

    # ========================================================================
    # Resource cleanup
    # ========================================================================

    def destroy(self) -> None:
        """Stop and remove the container (snapshot images untouched)."""
        logger.info("destroying container...")
        if self._container:
            try:
                self._container.stop()
                self._container.remove()
                logger.debug(f"removed container: {self._container.id[:12]}")
            except DockerException as e:
                logger.warning(f"failed to remove container: {e}")
            self._container = None
        logger.info("container destroyed")

    def cleanup(self) -> None:
        """Free everything: container + snapshot images.

        Called when the Agent finishes or aborts.
        """
        logger.info("cleaning up all resources...")
        self.destroy()
        self.cleanup_snapshots()
        logger.info("resource cleanup done")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False  # do not swallow exceptions
