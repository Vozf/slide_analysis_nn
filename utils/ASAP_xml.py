from xml.etree import ElementTree


def read_polygons_xml(xml_path):
    root = ElementTree.parse(xml_path).getroot()

    all_annotations = root.find('Annotations').iter('Annotation')
    polygon_annotations = filter(lambda annot: annot.get('Type') == 'Polygon', all_annotations)

    polygons_coordinates_annotations = [x.find('Coordinates') for x in polygon_annotations]
    polygons = [[(int(float(x.get('X'))), int(float(x.get('Y'))))
                 for x in polygon.iter('Coordinate')]
                for polygon in polygons_coordinates_annotations]

    return polygons

def write_polygons_xml(polygons, xml_path):
    pass