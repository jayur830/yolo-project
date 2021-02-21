import json

if __name__ == '__main__':
    with open("E:/Dataset/image/text_in_the_wild/textinthewild_data_info.json", "r", encoding="utf-8") as reader:
        annotations = json.load(reader)
    images = annotations["images"]
    annotations = annotations["annotations"]
    print(len(images))
    print(len(annotations))
    print(images[2])
    print(annotations[2])

