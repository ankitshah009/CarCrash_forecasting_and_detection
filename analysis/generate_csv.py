import csv, os
import argparse
from glob import glob
from analysis.annotations import read_vatic, find_boundary
CADP_IMAGE_HOME="/media/tuananhn/903a7d3c-0ce5-444b-ad39-384fcda231ed/CADP/extracted_frames/"
CADP_MASK_HOME="/media/tuananhn/903a7d3c-0ce5-444b-ad39-384fcda231ed/CADP/masked_frames/"


def generate_csv(writer, vatic_file, use_mask=True):
    annotations = read_vatic(vatic_file)
    vid = os.path.basename(vatic_file).split(".")[0]
    boundary = find_boundary(annotations)
    for trackId in annotations:
        tracklet = annotations[trackId]
        label = tracklet["label"]
        if label not in ["Separator", "CrowdRegion", "FarRegion"]:
            for fid in tracklet["frames"]:
                img_path = os.path.join(CADP_IMAGE_HOME, "{:06d}".format(int(vid)), "{}.jpg".format(fid))
                if use_mask and os.path.exists(os.path.join(CADP_MASK_HOME, "{:06d}".format(int(vid)), "{}.jpg".format(fid))):
                    img_path = os.path.join(CADP_MASK_HOME, "{:06d}".format(int(vid)), "{}.jpg".format(fid))
                if boundary[1] == -1 or fid in range(boundary[0], boundary[1]+1):
                    frame = tracklet["frames"][fid]
                    if frame["visible"]:
                        y1, x1, y2, x2 = frame["box"]
                        writer.writerow([img_path, x1, y1, x2, y2, label])
    return


def write_csv(anno_dir, csv_output, use_mask=True):
    with open(csv_output, "w") as f:
        writer = csv.writer(f, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        txtfiles = glob(anno_dir+"/*.txt")
        for txtfile in txtfiles:
            generate_csv(writer, txtfile, use_mask)
    f.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_dir", type=str)
    parser.add_argument("--csv_output", type=str)
    parser.add_argument("--use_mask", type=bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    write_csv(anno_dir=args.anno_dir, csv_output=args.csv_output, use_mask=args.use_mask)


