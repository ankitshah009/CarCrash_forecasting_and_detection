import xml.etree.ElementTree as ET
import cv2
import csv, os
import argparse
from tqdm import tqdm

"""
Sample output:
<annotation>
	<folder>GeneratedData_Train</folder>
	<filename>000001.png</filename>
	<path>/my/path/GeneratedData_Train/000001.png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>224</width>
		<height>224</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>21</name>
		<pose>Frontal</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<occluded>0</occluded>
		<bndbox>
			<xmin>82</xmin>
			<xmax>172</xmax>
			<ymin>88</ymin>
			<ymax>146</ymax>
		</bndbox>
	</object>
</annotation>
"""

CADP_CLASSES = {"bg": 0, "Car": 3, "Bus": 5, "Others": 6, "Person": 1, "Two-wheeler": 2, "Three-wheeler": 4}


def read_csv(csv_file_path):
    annotations = {}
    with open(csv_file_path) as f:
        r = csv.reader(f, delimiter=",")
        for row in r:
            # print(row)
            if row[0] not in annotations:
                annotations[row[0]] = {"objects": [{"name": row[-1], "x1": row[1], "y1": row[2], "x2": row[3], "y2": row[4]}]}
            else:
                annotations[row[0]]["objects"].append({"name": row[-1], "x1": row[1], "y1": row[2], "x2": row[3], "y2": row[4]})
    f.close()
    return annotations


def build_xml(folder, filename, path, objects):
    top = ET.Element("annotation")
    folder_tag = ET.SubElement(top, "folder")
    folder_tag.text = folder
    fn_tag = ET.SubElement(top, "filename")
    fn_tag.text = filename
    path_tag = ET.SubElement(top, "path")
    path_tag.text = path
    source_tag = ET.SubElement(top, "source")
    db_tag = ET.SubElement(source_tag, "database")
    db_tag.text = "CADP"
    img = cv2.imread(path)
    h,w,c = img.shape
    size_tag = ET.SubElement(top, "size")
    w_tag = ET.SubElement(size_tag, "width")
    w_tag.text = str(w)
    h_tag = ET.SubElement(size_tag, "height")
    h_tag.text = str(h)
    d_tag = ET.SubElement(size_tag, "depth")
    d_tag.text = str(c)
    seg_tag = ET.SubElement(top, "segmented")
    seg_tag.text = "0"
    for obj in objects:
        obj_tag = ET.SubElement(top, "object")
        name_tag = ET.SubElement(obj_tag, "name")
        name_tag.text = str(CADP_CLASSES[obj["name"]])
        pose_tag = ET.SubElement(obj_tag, "pose")
        pose_tag.text = "Unknown"
        trunc_tag = ET.SubElement(obj_tag, "truncated")
        trunc_tag.text = "0"
        diff_tag = ET.SubElement(obj_tag, "difficult")
        diff_tag.text = "0"
        occluded_tag = ET.SubElement(obj_tag, "occluded")
        occluded_tag.text = "0"
        bndbox_tag = ET.SubElement(obj_tag, "bndbox")
        xmin_tag = ET.SubElement(bndbox_tag, "xmin")
        xmin_tag.text = str(obj["x1"])
        xmax_tag = ET.SubElement(bndbox_tag, "xmax")
        xmax_tag.text = str(obj["x2"])
        ymin_tag = ET.SubElement(bndbox_tag, "ymin")
        ymin_tag.text = str(obj["y1"])
        ymax_tag = ET.SubElement(bndbox_tag, "ymax")
        ymax_tag.text = str(obj["y2"])
    return top


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to csv annotation file.")
    parser.add_argument("--output_xml_dir", type=str, help="Where to put the PASCAL VOC annotations.")
    parser.add_argument("--listfile", type=str, help="List of files.")
    return parser.parse_args()


def main(csv_file_path, output_xml_dir, listfile):
    if not os.path.exists(output_xml_dir):
        os.makedirs(output_xml_dir)
    annotations = read_csv(csv_file_path)
    f = open(listfile, "w")
    for fp in tqdm(annotations):
        objects = annotations[fp]["objects"]
        fn = os.path.basename(fp)
        foldername = fp.split("/")[-2]
        top = build_xml(folder="", filename=fn, path=fp, objects=objects)
        if not os.path.exists("{}/{}".format(output_xml_dir, foldername)):
            os.makedirs("{}/{}".format(output_xml_dir, foldername))
        ET.ElementTree(top).write("{}/{}/{}.xml".format(output_xml_dir, foldername, fn.split(".")[0]))
        f.write("{} {}\n".format(os.path.abspath(fp), os.path.abspath("{}/{}/{}.xml".format(output_xml_dir, foldername, fn.split(".")[0]))))
    f.close()
    return


if __name__ == '__main__':
    args = parse_arguments()
    main(args.csv, args.output_xml_dir, args.listfile)
