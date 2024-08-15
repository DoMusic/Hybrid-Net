import uuid


def generate_uuid(version=4, namespace=None, name=None):
    """
    Generate a UUID string based on the specified version.

    Parameters:
    - version (int): The UUID version (1, 3, 4, or 5). Default is 4.
    - namespace (str): The namespace for UUID3 and UUID5. Must be a valid UUID string.
    - name (str): The name for UUID3 and UUID5.

    Returns:
    - str: The generated UUID string.
    """
    if version == 1:
        return str(uuid.uuid1())
    elif version == 3:
        if namespace is None or name is None:
            raise ValueError("Namespace and name must be provided for UUID3.")
        namespace_uuid = uuid.UUID(namespace)
        return str(uuid.uuid3(namespace_uuid, name))
    elif version == 4:
        return str(uuid.uuid4())
    elif version == 5:
        if namespace is None or name is None:
            raise ValueError("Namespace and name must be provided for UUID5.")
        namespace_uuid = uuid.UUID(namespace)
        return str(uuid.uuid5(namespace_uuid, name))
    else:
        raise ValueError("Unsupported UUID version. Supported versions are 1, 3, 4, and 5.")
