"""
To train with crowd and far regions are masked.
This script will generate images to train a detector with crowd/far regions masked.
DISCLAIMER: It will take time.
"""
import os, argparse
from glob import glob
from tqdm import tqdm
from skimage import io
from analysis.annotations import read_vatic, find_boundary

CADP_IMAGE_HOME="/media/tuananhn/903a7d3c-0ce5-444b-ad39-384fcda231ed/CADP/extracted_frames/"
CADP_MASK_HOME="/media/tuananhn/903a7d3c-0ce5-444b-ad39-384fcda231ed/CADP/masked_frames/"


def get_crowd_far(vatic_file, output=None):
    if output is None:
        output = {}
    annotations = read_vatic(vatic_file)
    vid = os.path.basename(vatic_file).split(".")[0]
    boundary = find_boundary(annotations)
    for trackId in annotations:
        tracklet = annotations[trackId]
        label = tracklet["label"]
        if label in ["CrowdRegion", "FarRegion"]:
            for fid in tracklet["frames"]:
                img_path = os.path.join("{:06d}".format(int(vid)), "{}.jpg".format(fid))
                if boundary[1] == -1 or fid in range(boundary[0], boundary[1] + 1):
                    frame = tracklet["frames"][fid]
                    if frame["visible"]:
                        y1, x1, y2, x2 = frame["box"]
                        if img_path not in output:
                            output[img_path] = [[x1,y1,x2,y2]]
                        else:
                            output[img_path].append([x1,y1,x2,y2])
    return output


def create_crowd_far(anno_dir):
    txtfiles = glob(anno_dir + "/*.txt")
    for txtfile in tqdm(txtfiles):
        output = get_crowd_far(txtfile)
        vid = os.path.basename(txtfile).split(".")[0]
        if not os.path.exists(os.path.join(CADP_MASK_HOME, "{:06d}".format(int(vid)))):
            os.makedirs(os.path.join(CADP_MASK_HOME, "{:06d}".format(int(vid))))
        for img_path in output:
            img = io.imread(os.path.join(CADP_IMAGE_HOME, img_path))
            for crowd_far in output[img_path]:
                x1, y1, x2, y2 = crowd_far
                img[y1:y2, x1:x2, :] = 0 # masking the region
            io.imsave(os.path.join(CADP_MASK_HOME, img_path), img)
    return


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    create_crowd_far(anno_dir=args.anno_dir)


