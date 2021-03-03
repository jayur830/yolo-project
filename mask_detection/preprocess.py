from xml.etree.ElementTree import parse
from glob import glob

# TODO convert xml -> txt and should to add classes.txt
if __name__ == '__main__':
    root = parse("E:/Dataset/image/mask_detection/annotations/maksssksksss0.xml").getroot()


    print(root.find("size").findtext("width"))

    # file_list = glob("E:/Dataset/image/mask_detection/annotations/*")
    #
    # for filepath in file_list:
    #     tree = parse(filepath)
    #     root = tree.getroot()
    #
    #
