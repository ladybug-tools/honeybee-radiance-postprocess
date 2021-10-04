import re


def _filter_by_pattern(input_objects, filter):
    """Filter model grids and views based on user input."""
    if not filter or filter == '*':
        return input_objects

    if not isinstance(filter, (list, tuple)):
        filter = [filter]
    patterns = [
        re.compile(f.replace('*', '.+').replace('?', '.')) for f in filter
        ]

    indexes = []

    for count, obj in enumerate(input_objects):
        try:
            id_ = obj.full_identifier
        except AttributeError:
            id_ = obj['full_id']
        for pattern in patterns:
            if re.search(pattern, id_):
                indexes.append(count)
    indexes = list(set(indexes))
    indexes.sort()

    return [input_objects[i] for i in indexes]
