import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GEE_PROJECT_NAME = os.getenv("GEE_PROJECT_NAME", "")


def _resolve_java_home() -> str:
    """Resolve the JDK location for Spark.

    JDK_HOME (from .env) may be an absolute JDK path, or just a JDK directory
    name installed under `<DATA_DIR parent>/.jdk/` or `~/.jdk/` (the layout
    used by the install-jdk package). Falls back to a pre-set JAVA_HOME.
    """
    jdk = os.getenv("JDK_HOME", "")
    if jdk:
        p = Path(jdk)
        if p.is_absolute():
            return str(p)
        candidates = [Path.home() / ".jdk" / jdk]
        data_dir = os.getenv("DATA_DIR", "")
        if data_dir:
            candidates.insert(0, Path(data_dir).parent / ".jdk" / jdk)
        for c in candidates:
            if c.exists():
                return str(c)
    return os.getenv("JAVA_HOME", "")


JAVA_HOME = _resolve_java_home()
