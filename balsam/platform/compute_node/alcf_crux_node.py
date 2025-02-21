import logging
import os
from typing import List, Optional, Union

from .compute_node import ComputeNode

logger = logging.getLogger(__name__)
IntStr = Union[int, str]


class CruxNode(ComputeNode):
    cpu_ids = list(range(128))  # Crux has 128 CPU cores
    # No need to define gpu_ids; it will default to [] from ComputeNode

    @classmethod
    def get_job_nodelist(cls) -> List["CruxNode"]:
        """
        Get all compute nodes allocated in the current job context.
        """
        nodefile = os.environ.get("PBS_NODEFILE")
        if not nodefile or not os.path.exists(nodefile):
            logger.error("PBS_NODEFILE environment variable is not set or file does not exist.")
            return []

        # Read hostnames from the nodefile
        with open(nodefile) as fp:
            hostnames = [line.strip() for line in fp if line.strip()]

        node_ids: Union[List[str], List[int]] = hostnames[:]
        node_list = []
        for nid, hostname in zip(node_ids, hostnames):
            # Since Crux does not have GPUs, no need to pass gpu_ids
            node_list.append(cls(nid, hostname))
        return node_list

    @staticmethod
    def get_scheduler_id() -> Optional[int]:
        job_id_str = os.environ.get("PBS_JOBID")
        if job_id_str is not None:
            try:
                return int(job_id_str.split(".")[0])
            except ValueError:
                logger.error(f"Unable to parse PBS_JOBID: {job_id_str}")
                return None
        return None
      
