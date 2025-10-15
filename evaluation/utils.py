import json
import types
import inspect


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle class objects (not instances)
        if isinstance(obj, type):
            return {
                "__type__": "class",
                "name": obj.__name__,
                "module": obj.__module__,
                "file": inspect.getfile(obj),  # Get file where the class is defined
            }

        # Handle function objects
        elif isinstance(obj, types.FunctionType):
            return {
                "__type__": "function",
                "name": obj.__name__,
                "module": obj.__module__,
                "file": inspect.getfile(obj),  # Get file where the function is defined
            }

        # Handle class instances
        elif hasattr(obj, "__class__") and not isinstance(obj, type):
            cls = obj.__class__
            return {
                "__type__": "instance",
                "class_name": cls.__name__,
                "module": cls.__module__,
                "file": inspect.getfile(cls),  # Get file where the class is defined
            }

        # Failsafe: Handle any object by serializing it as a string with its type information
        elif isinstance(obj, object):
            return {
                "__type__": "object",
                "class_name": obj.__class__.__name__,
                "module": obj.__class__.__module__,
                "file": inspect.getfile(
                    obj.__class__
                ),  # Get file where the class is defined
                "repr": repr(obj),  # Fallback representation
            }

        # Default behavior for other types
        return super().default(obj)
