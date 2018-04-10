import colorsys
from xml.etree import cElementTree as ET


def read_polygons_xml(xml_path):
    root = ET.parse(xml_path).getroot()

    all_annotations = root.find('Annotations').iter('Annotation')
    polygon_annotations = filter(lambda annot: annot.get('Type') == 'Polygon', all_annotations)

    polygons_coordinates_annotations = [x.find('Coordinates') for x in polygon_annotations]
    polygons = [[(int(float(x.get('X'))), int(float(x.get('Y'))))
                 for x in polygon.iter('Coordinate')]
                for polygon in polygons_coordinates_annotations]

    return polygons


def create_polygons_xml(polygons, predicted_labels, scores, xml_path):
    root = ET.Element("ASAP_Annotations")
    ET.SubElement(root, "Annotations")

    _write_polygons_to_annotations(root, polygons, predicted_labels, scores)

    tree = ET.ElementTree(root)
    tree.write(xml_path)
    return xml_path


def _write_polygons_to_annotations(root, polygons, predicted_labels, scores):
    anns_label = "Annotations"
    anns = _find_or_create_subelement(root, label=anns_label)

    group_label = 'NN Prediction'

    [_create_polygon(anns, *params, group_label, i) for i, params in
     enumerate(zip(polygons, predicted_labels, scores))]

    _create_group(root, label=group_label)


def _find_or_create_subelement(root, label):
    return root.find(label) if root.find(label) else ET.SubElement(root, label)


def append_polygons_to_existing_xml(polygons, predicted_labels, scores, source_xml_path,
                                    output_xml_path=None):
    if not output_xml_path:
        output_xml_path = source_xml_path

    root = ET.parse(source_xml_path).getroot()

    _write_polygons_to_annotations(root, polygons, predicted_labels, scores)

    tree = ET.ElementTree(root)
    tree.write(output_xml_path)
    return output_xml_path


def _create_polygon(anns, polygon, label, score, group_label, i):
    ann = ET.SubElement(anns, "Annotation", Name="{0}_({1})".format(i, score), Type="Polygon",
                        label=str(label), score=str(score),
                        Color=_get_hex_color_from_score(score), PartOfGroup=group_label)
    coords = ET.SubElement(ann, "Coordinates")

    [ET.SubElement(coords, "Coordinate", Order=str(idx), X=str(point[0]), Y=str(point[1])) for
     idx, point in
     enumerate(polygon)]


def _create_group(root, label):
    groups = _find_or_create_subelement(root, "AnnotationGroups")
    group = ET.SubElement(groups, "Group", Name=label)
    return group


def _get_hex_color_from_score(score):
    rgb0_1 = colorsys.hls_to_rgb((1 - score) * 84 / 255, 0.5, 0.5)
    rgb0_255 = [int(x * 255) for x in rgb0_1]
    return '#%02x%02x%02x' % tuple(rgb0_255)

# write_polygons_xml([[1,2,3,4],[5,6,7,8]], [0.9, 0.2], '/home/vozman/projects/slides/slide-analysis-nn/prediction/asap_annotations/xm.xml')

# read_xml = '/home/vozman/projects/slides/slide-analysis-nn/train/datasets/source/slide_images/Tumor_016pred.xml'
# write_xml = '/home/vozman/projects/slides/slide-analysis-nn/train/datasets/source/slide_images/Tumor_016.xml'
#
# root = ET.parse(read_xml).getroot()
#
# all_annotations = root.find('Annotations').iter('Annotation')
#
# root = ET.Element("ASAP_Annotations")
# anns = ET.SubElement(root, "Annotations")
#
# for read_annot in all_annotations:
#     name = read_annot.get('Name')
#     idx = name.index('_')
#     if random.random()<0.95:
#         continue
#     read_annot.set('PartOfGroup', 'None')
#     anns.append(read_annot)
#
# tree = ET.ElementTree(root)
# tree.write(write_xml)
