import logging
import os

from balsam.platform.compute_node import ComputeNode

from .app_run import SubprocessAppRun

logger = logging.getLogger(__name__)


class CruxRun(SubprocessAppRun):
  """
  Implements application launch for the Crux system.

  This class constructs the appropriate command line for launching applications
  using `mpiexec`, tailored for the Crux hardware and scheduler.

  Crux Specifications:
  - CPU-only system with dual AMD EPYC 7742 64-Core Processors per node.
  - Each core supports up to two hyperthreads (total 256 threads per node).
  - Uses PBS scheduler for job management.

  Example mpiexec command from Crux submission script:
  mpiexec -n total_ranks --ppn ranks_per_node --depth=depth --cpu-bind depth \
      --env OMP_NUM_THREADS=num_threads --env OMP_PROC_BIND=true --env OMP_PLACES=cores \
      executable
  """

  def _build_cmdline(self) -> str:
    node_hostnames = [h for h in self._node_spec.hostnames]
    ntasks = self.get_num_ranks()
    nranks_per_node = self._ranks_per_node
    nthreads = self._threads_per_rank
    cpus_per_rank = self.get_cpus_per_rank()
    cpu_bind = self._launch_params.get("cpu_bind", "depth")
    
    depth = nthreads
    if cpu_bind == "core":
        depth = cpus_per_rank  
    
    mpi_args = [
        "mpiexec",
        "-n", ntasks,
        "--ppn", nranks_per_node,
        "--hosts", ",".join(node_hostnames),
        "--depth", depth,
        "--cpu-bind", cpu_bind,
    ]

    # Add any additional launch parameters
    for key, value in self._launch_params.items():
        if key not in ["--ppn", "ppn", "--cpu-bind", "cpu-bind", "--depth", "depth"]:
          mpi_args.append(str(key))
          if value: # if value is not empty; like the flag --verbose has no value
            mpi_args.append(value)

    mpi_args.append(self._cmdline)
    
    cmd = " ".join(str(arg) for arg in mpi_args)
    return cmd

  def _set_envs(self) -> None:
    envs = os.environ.copy()
    envs.update(self._envs)
    # Note app_run.py handles setting omp_num_threads (envs["OMP_NUM_THREADS"] = str(self._threads_per_rank), line 159)
    envs["OMP_NUM_THREADS"] = str(self._threads_per_rank)
    envs["OMP_PROC_BIND"] = "true"
    envs["OMP_PLACES"] = "cores"
    self._envs = envs

