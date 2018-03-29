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


def write_polygons_xml(polygons, scores, xml_path):
    root = ET.Element("ASAP_Annotations")
    anns = ET.SubElement(root, "Annotations")

    [_create_subelement(anns, polygon, score, i) for i, (polygon, score) in enumerate(zip(polygons, scores))]

    tree = ET.ElementTree(root)
    tree.write(xml_path)


def _create_subelement(anns, polygon, score, i):
    ann = ET.SubElement(anns, "Annotation", Name="{0}_({1})".format(i, score), Type="Polygon",
                        Color=_get_hex_from_score(score))
    coords = ET.SubElement(ann, "Coordinates")

    x1, y1, x2, y2 = polygon
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    [ET.SubElement(coords, "Coordinate", Order=str(idx), X=str(point[0]), Y=str(point[1])) for idx, point in
     enumerate(points)]

def _get_hex_from_score(score):
    rgb0_1 = colorsys.hls_to_rgb((1 - score) * 84 / 255, 0.5, 0.5)
    rgb0_255 = [int(x*255) for x in rgb0_1]
    return '#%02x%02x%02x' % tuple(rgb0_255)

# write_polygons_xml([[1,2,3,4],[5,6,7,8]], [0.9, 0.2], '/home/vozman/projects/slides/slide-analysis-nn/prediction/asap_annotations/xm.xml')