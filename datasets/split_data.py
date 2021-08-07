import os
import sys
import random
import json
import argparse
from collections import defaultdict
from tqdm import tqdm


def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_path', type=str, default="anno")
    parser.add_argument('--data_name', type=str, required=True, choices=("TUBerlin", "sketchy"))

    config = parser.parse_args()
    if not os.path.lexists(config.save_path):
        os.makedirs(config.save_path)
    config.test_num = 10 if config.data_name == "TUBerlin" else 50

    return config

def add_label(lst, label, *prefix):
    return list(map(lambda x: (os.path.join(*prefix, x), label), lst))

tuberlin_category_correlation = {"person sitting": "personsitting", "bear (animal)": "bearanimal",
                                 "cell phone": "cellphone", "trombone": "tromobone", "crane (machine)": "cranemachine",
                                 "fire hydrant": "firehydrant", "standing bird": "standingbird", "t-shirt": "tshirt",
                                 "beer-mug": "beermug", "santa claus": "santaclaus", "bottle opener": "bottleopener",
                                 "baseball bat": "baseballbat", "giraffe": "griaffe", "race car": "racecar",
                                 "satellite dish": "satellitedish", "power outlet": "poweroutlet", "spider": "spidar",
                                 "traffic light": "trafficlight", "sea turtle": "seaturtle", "floor lamp": "floorlamp",
                                 "parking meter": "parkingmeter", "speed-boat": "speedboat", "wine-bottle": "winebottle",
                                 "hot-dog": "hotdog", "door handle": "doorhandle", "tennis-racket": "tennisracket",
                                 "computer-mouse": "computermouse", "ice-cream-cone": "icecreamcone",
                                 "chandelier": "chandeler", "head-phones": "headphones", "flying bird": "flyingbird",
                                 "flying saucer": "flyingsaucer", "human-skeleton": "humanskeleton",
                                 "paper clip": "paperclip", "frying-pan": "fryingpan", "space shuttle": "spacesshuttle",
                                 "alarm clock": "alarmclock", "walkie talkie": "walkietalkie", "teddy-bear": "teddybear",
                                 "computer monitor": "computermonitor", "hot air balloon": "hotairballoon",
                                 "palm tree": "palmtree",  "sponge bob": "spongebob", "person walking": "personwalking",
                                 "car (sedan)": "carsedan", "pipe (for smoking)": "pipe", "mouse (animal)": "mouse",
                                 "pickup truck": "pickuptruck", "wrist-watch": "wristwatching", "diamond": "diamod",
                                 "potted plant": "pottedplant", "flower with stem": "flowerwithstem"
                                }

if __name__ == "__main__":
    config = get_parameter()

    # construct path
    path_photo = os.path.join(config.data_root, config.data_name, "photos")
    path_sketch = os.path.join(config.data_root, config.data_name, "sketches")

    # check category
    cate_photo = os.listdir(path_photo)
    cate_sketch = os.listdir(path_sketch)
    cate_sketch.sort()

    # add images
    photos, sketches = defaultdict(list), defaultdict(list)
    for i, cate in tqdm(enumerate(cate_sketch)):
        sketches_cate = os.listdir(os.path.join(path_sketch, cate))
        random.shuffle(sketches_cate)
        sketches['test'].extend(add_label(sketches_cate[:config.test_num], i, "sketches", cate))
        sketches['train'].extend(add_label(sketches_cate[config.test_num:], i, "sketches", cate))

        cate = tuberlin_category_correlation.get(cate, cate) if (config.data_name == "TUBerlin") else cate
        photo_cate = os.listdir(os.path.join(path_photo, cate))
        photos["train"].extend(add_label(photo_cate, i, "photos", cate))
        photos["test"] = photos["train"]

    # save
    with open(os.path.join(config.save_path, config.data_name)+"_photo.json", 'w') as f:
        json.dump(photos, f, indent=1)

    with open(os.path.join(config.save_path, config.data_name)+"_sketch.json", 'w') as f:
        json.dump(sketches, f, indent=1)


