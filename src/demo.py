def get_classes(filename="../tiny-imagenet-200/val/val_annotations.txt"):
    """
    Get corresponding class name for each val image.


    Args:
        filename: txt file which contains image name and corresponding class name

    Returns:
        class_dict: A dictionary which maps from image name to class name
    """
    class_dict = {}
    for line in open(filename):
        line_array = line.rstrip("\n").split("\t")
        class_dict[line_array[0]] = line_array[1]

    i = 0
    for k, v in class_dict.items():
        print("{} ==> {}".format(k, v))
        i += 1
        if i > 5:
            break

    return class_dict

get_classes()
