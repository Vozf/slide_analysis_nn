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

def write_polygons_xml(polygons, xml_path):
    root = ET.Element("ASAP_Annotations")
    anns = ET.SubElement(root, "Annotations")

    [_create_subelement(anns, polygon, i) for i, polygon in enumerate(polygons)]

    tree = ET.ElementTree(root)
    tree.write(xml_path)

def _create_subelement(anns, polygon, i):
    ann = ET.SubElement(anns, "Annotation", Name=str(i), Type="Polygon")
    coords = ET.SubElement(ann, "Coordinates")

    x1, y1, x2, y2 = polygon
    points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    [ET.SubElement(coords, "Coordinate", Order=str(idx), X=str(point[0]), Y=str(point[1])) for idx, point in
     enumerate(points)]
