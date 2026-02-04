
from utils.constants import JAVA_HOME

import jdk

def install_jdk(version: str='8',path: str=JAVA_HOME) -> str:
    """
    Installs the specified version of the JDK (Java Development Kit) to the given path.

    Args:
        version (str): The version of the JDK to install. Defaults to '8'.
        path (str): The path where the JDK should be installed. Defaults to JAVA_HOME.

    Returns:
        str: The path where the JDK was installed.
    """
    
    jdk_path = jdk.install(version, path=path)
    return jdk_path
