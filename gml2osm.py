#!/usr/bin/env python3
# -*- coding: utf8

# gml2osm
# Converts GML to geojson format.
# Supports UTM reprojection - dependency of separate utm.py file.
# Supports input from url (including zip + named file), and local file.
# Use as library for other programs which needs to load GML.
# Usage: gml2osm <input_url or file> <file in zip> <object type>


import urllib.request, urllib.parse, urllib.error
import zipfile
from io import BytesIO
import json
import sys
import math
import copy
from xml.etree import ElementTree as ET
import utm

version = "0.3.0"

header = {"User-Agent": "osmno/gml2osm"}

coordinate_decimals = 7

municipalities = {}



def message (output_text):
	'''
	Output message to console
	'''
	sys.stdout.write (output_text)
	sys.stdout.flush()



def line_distance(s1, s2, p3):
	'''
	Compute closest distance from point p3 to line segment [s1, s2].
	Works for short distances.
	Arguments:
		- s1, s2: Two points (lon, lat) which defines a line segment.
		- p3: Point (lon, lat) from which the distance is measured to the line.
	Returns distance in meters from p3 to line segment [s1, s2].
	'''

	x1, y1, x2, y2, x3, y3 = map(math.radians, [s1[0], s1[1], s2[0], s2[1], p3[0], p3[1]])  # Note: (x,y)

	# Simplified reprojection of latitude
	x1 = x1 * math.cos( y1 )
	x2 = x2 * math.cos( y2 )
	x3 = x3 * math.cos( y3 )

	A = x3 - x1
	B = y3 - y1
	dx = x2 - x1
	dy = y2 - y1

	dot = (x3 - x1)*dx + (y3 - y1)*dy
	len_sq = dx*dx + dy*dy

	if len_sq != 0:  # in case of zero length line
		param = dot / len_sq
	else:
		param = -1

	if param < 0:
		x4 = x1
		y4 = y1
	elif param > 1:
		x4 = x2
		y4 = y2
	else:
		x4 = x1 + param * dx
		y4 = y1 + param * dy

	# Also compute distance from p to segment

	x = x4 - x3
	y = y4 - y3
	distance = 6371000 * math.sqrt( x*x + y*y )  # In meters

	'''
	# Project back to longitude/latitude

	x4 = x4 / math.cos(y4)

	lon = math.degrees(x4)
	lat = math.degrees(y4)

	return (lon, lat, distance)
	'''

	return distance



def get_municipality (parameter):
	'''
	Identify municipality name, unless more than one hit
	Argument: Municipality/county code or name.
	Returns municipality/county code, or None if not found. Norway is 0000.

	'''

	if not municipalities:
		load_municipalities()

	if parameter.isdigit():
		if parameter in municipalities:
			return parameter
		else:
			return None

	else:
		found_id = ""
		duplicate = False
		for mun_id, mun_name in iter(municipalities.items()):
			if parameter.lower() == mun_name.lower():
				return mun_id
			elif parameter.lower() in mun_name.lower():
				if found_id:
					duplicate = True
				else:
					found_id = mun_id

		if found_id and not duplicate:
			return found_id
		else:
			return None



def load_municipalities():
	'''
	Load dict of all municipalities and counties.
	Updates global "municipalities" dict with county and municipality code + name.
	'''

	url = ("https://ws.geonorge.no/kommuneinfo/v1/fylkerkommuner?"
			"filtrer=fylkesnummer%2Cfylkesnavn%2Ckommuner.kommunenummer%2Ckommuner.kommunenavnNorsk")
	try:
		file = urllib.request.urlopen(url)
	except urllib.error.HTTPError as err:
		message("\n\t\t*** Unable to load municipalities from GeoNorge - %s\n" % err)
	data = json.load(file)
	file.close()

	municipalities['0000'] = "Norge"
	for county in data:
		for municipality in county['kommuner']:
			municipalities[ municipality['kommunenummer'] ] = municipality['kommunenavnNorsk']
		municipalities[ county['fylkesnummer'] ] = county['fylkesnavn']



def clean_url (url):
	'''
	Replace Scandinavian characters in url/filename (Kartverket/GeoNorge standard).
	'''
	return url.replace("Æ","E").replace("Ø","O").replace("Å","A").replace("æ","e").replace("ø","o").replace("å","a").replace(" ", "_")



def load_gml (url, filename="", object_filter=[], verbose=False):
	'''
	Load gml from url/file.
	Arguments:
		- url: URL for input file.
		- filenamme: file within zip-file, or empty.
		- object_filter: List of object types to include, or empty to get all.
		- verbose: True to get messages regarding progress.
	Returns: List of loaded features from GML.
	'''

	def parse_coordinates (coord_text):
		'''
		Get list of coordinates from GML.
		Each point is a tuple of (lon, lat), corresponding to GeoJSON format [x, y].
		Uses non-local variables utm_zone and dimension and global variable coordinate_decimals. 
		'''
		split_coord = coord_text.split(" ")
		coordinates = []

		for i in range(0, len(split_coord) - 1, dimension):
			x = float(split_coord[i])
			y = float(split_coord[i+1])

			if utm_zone:
				[lat, lon] = utm.UtmToLatLon (x, y, utm_zone, "N")
			else:
				[lat, lon] = [x, y]

			node = ( round(lon, coordinate_decimals), round(lat, coordinate_decimals) )
			coordinates.append(node)

		return coordinates


	def parse_properties (top):
		'''
		Get all app properties from nested XML; recursive search.
		'''
		properties = {}
		if ns_app in top.tag:
			tag = top.tag[ len(ns_app)+2 : ]
			value = top.text
			if value and value.strip():
				properties[tag] = value

		for child in top:
			properties.update(parse_properties(child))

		return properties


	# Start main function

	url = clean_url(url)
	if verbose:
		message ("Load gml...\n")
		message ("\tUrl: %s\n" % url)

	object_count = {}
	unknown = []

	# Load zip file

	if ".zip" in url.lower():
		request = urllib.request.Request(url, headers=header)
		try:
			file_in = urllib.request.urlopen(request)
		except urllib.error.HTTPError as err:
			message("\n\t\t*** Unable to load file - %s\n" % err)
		zip_file = zipfile.ZipFile(BytesIO(file_in.read()))

		if verbose:
			message ("\tFiles in zip folder:\n")
			for file_entry in zip_file.namelist():
				message ("\t\t%s\n" % file_entry)

		if not filename and len(zip_file.namelist()) == 1:
			filename = zip_file.namelist()[0]
		elif not filename:
			if not verbose:
				message ("\tFiles in zip folder:\n")
				for file_entry in zip_file.namelist():
					message ("\t\t%s\n" % file_entry)				
			sys.exit("Please also provide file name from the liste above\n\n")

		file = zip_file.open(filename)

	# Load regular url/file

	elif "http" in url.lower():
		try:
			request = urllib.request.Request(url, headers=header)
		except urllib.error.HTTPError as err:
			message("\n\t\t*** Unable to load file - %s\n\n" % err)
		file = urllib.request.urlopen(request)

	# Load local file

	else:
		file = open(url)

	if verbose:
		message ("Loading file '%s'\n" % filename)

	tree = ET.parse(file)
	file.close()
	root = tree.getroot()

	# Get namespace

	srs = None
	ns_gml = "http://www.opengis.net/gml/3.2"
	ns_app = None
	schemas = root.get("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation").split(" ")

	for schema in schemas:
		if not ns_app and not "opengis" in schema:
			ns_app = schema
		if not ns_gml and "gml" in schema:
			ns_gml = schema

	if ns_app is None:
		sys.exit("\tNamespace not found\n")

	ns = {
		'gml': ns_gml,
		'app': ns_app
	}

	if verbose:
		message ("\tSchemas:\n\t\t%s\n\t\t%s\n" % (ns_app, ns_gml))

	# Loop features, parse and load into data structure and tag

	if verbose:
		message ("\tParsing...\n")
	features = []

	for feature in root:

		if ns_app not in feature[0].tag:  # Skip pure gml (boundedBy etc)
			continue

		feature_type = feature[0].tag[ len(ns_app)+2 : ]

		if feature_type not in object_count:
			object_count[feature_type] = 0
		object_count[feature_type] += 1

		if object_filter and feature_type not in object_filter: # Skip if not in filter
			continue

		if "{%s}id" % ns_gml in feature[0].attrib:
			gml_id = feature[0].attrib["{%s}id" % ns_gml]
		else:
			gml_id = ""

		entry = {
			'object': feature_type,
			'type': None,
			'gml_id': gml_id,
			'coordinates': [],  # Will contain geometry in geojson structure
			'data': {},         # Will contain raw GML tags
			'tags': {}          # Place to put converted/processed tags for output in geojson properties
		}

		for app in feature[0]:

			# Get app properties/attributes

			tag = app.tag[ len(ns_app)+2 : ]
			entry['data'].update(parse_properties(app))

			# Get geometry/coordinates

			for geo in app:

				# Get projection
				srs_name = geo.get('srsName')
				if srs_name:
					if "4326" in srs_name or "4258" in srs_name:
						utm_zone = None
					elif "258" in srs_name:
						utm_zone = int(srs_name[ srs_name.find("258") + 3 : ])
					elif "59" in srs_name:
						utm_zone = int(srs_name[ srs_name.find("59") + 2 : ]) - 40
					else:
						sys.exit("\n***Not supported projection: %s\n\n" % srs_name)

				# Get dimension (2 or 3)
				dimension = geo.get('srsDimension')
				if dimension is None:
					dimension = 2
				else:
					dimension = int(dimension)

				# Point
				if geo.tag == "{%s}Point" % ns_gml:
					entry['type'] = "Point"
					entry['coordinates'] = parse_coordinates(geo[0].text)[0]

				# MultiPoint
				elif geo.tag == "{%s}MultiPoint" % ns_gml:
					entry['type'] = "Point"
					entry['data']['GML_GEOMETRY_SOURCE'] = "MultiPoint"						
					entry['coordinates'] = parse_coordinates(geo[0][0][0].text)[0]						

				# LineString
				elif geo.tag == "{%s}LineString" % ns_gml:
					entry['type'] = "LineString"
					entry['coordinates'] = parse_coordinates(geo[0].text)

				# Curve, stored as one LineString
				elif geo.tag == "{%s}Curve" % ns_gml:
					entry['type'] = "LineString"
					entry['data']['GML_GEOMETRY_SOURCE'] = "Curve"
					entry['coordinates'] = []
					for patch in geo[0]:
						coordinates =  parse_coordinates(patch[0].text)
						if entry['coordinates']:
							entry['coordinates'] += coordinates[1:]
						else:
							entry['coordinates'] = coordinates  # First patch
		
				# (Multi)Polygon
				elif geo.tag == "{%s}Surface" % ns_gml or geo.tag == "{%s}MultiSurface" % ns_gml:
					entry['type'] = "Polygon"
					entry['coordinates'] = []  # List of patches

					for patch in geo[0][0]:
						role = patch.tag[ len(ns_gml)+2 : ]

						if patch[0].tag == "{%s}Ring" % ns_gml:
							if patch[0][0][0].tag == "{%s}LineString" % ns_gml:
								coordinates = parse_coordinates(patch[0][0][0][0].text)  # Ring->LineString
							else:
								coordinates = parse_coordinates(patch[0][0][0][0][0][0].text)  # Ring->Curve->LineStringSegment
						else:
							coordinates = parse_coordinates(patch[0][0].text)  # LinearRing

						entry['coordinates'].append(coordinates)

				# Multigeometry + MultiSurface
				elif geo.tag != "{%s}Envelope" % ns_gml and ns_gml in geo.tag:
					if geo.tag not in unknown:
						unknown.append(geo.tag)
						message ("\t*** Unknown geometry: %s\n" % geo.tag)

		entry['data']['ID'] = gml_id
		entry['data']['FEATURE'] = feature_type
		if entry['coordinates']:
			features.append(entry)		

	if verbose:
		message("\tObjects loaded:\n")
		for object_type in sorted(object_count):
			message("\t\t%i\t%s\n" % (object_count[object_type], object_type))
		message ("\t%i feature objects\n" % (len(features)))

	return features



def simplify_line(line, epsilon):
	'''
	Simplify line, i.e. reduce nodes within epsilon distance.
	Ramer-Douglas-Peucker method: https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm
	Arguments:
		- line: list of points (lon, lat) to be simplified.
		- epislon: Smallest permitted distance in meters.
	Returns simplified lines (list of points).
	'''
	dmax = 0.0
	index = 0
	for i in range(1, len(line) - 1):
		d = line_distance(line[0], line[-1], line[i])
		if d > dmax:
			index = i
			dmax = d

	if dmax >= epsilon:
		new_line = simplify_line(line[:index+1], epsilon)[:-1] + simplify_line(line[index:], epsilon)
	else:
		new_line = [line[0], line[-1]]

	return new_line



def simplify_line_partitions(line, epsilon, avoid_nodes):
	'''
	Partition line into sublines at intersections or tagged nodes before simplifying each partition.
	Arguments:
		- line: list of points (lon, lat) to be simplified.
		- avoid_nodes: list or set of points (lon, lat) which should not be simplified.
	Returns simplified lines (list of points).
	'''
	remaining = copy.copy(line)
	new_line = [ remaining.pop(0) ]

	while remaining:
		subline = [ new_line[-1] ]

		while remaining and not remaining[0] in avoid_nodes:  # Continue until tagged or intersecting
			subline.append(remaining.pop(0))

		if remaining:
			subline.append(remaining.pop(0))

		new_line += simplify_line(subline, epsilon)[1:]

	return new_line



def simplify_features(features, epsilon, verbose=False):
	'''
	Simplify geometry in all features except nodes which are tagged.
	This function does not identify intersecting nodes between lines.
	Arguments:
		- features: List of features to be simplified.
		- epsilon: Simplify factor in meters. 0.2 is often used.
	Updates coordinates in feature list.
	'''

	if verbose:
		message ("Simplifies geometry ...\n")

	# Identify nodes which are tagged
	nodes = set()
	for feature in features:
		if feature['type'] == "Point" and feature['data']:
			nodes.add(feature['coordinates'])

	# Loop features and simplify
	deleted = 0
	total = 0
	for feature in features:
		if feature['type'] == "LineString":
			total += len(feature['coordinates'])
			coordinates = simplify_line_partitions(feature['coordinates'], epsilon, nodes)
			if coordinates != feature['coordinates']:
				deleted += len(feature['coordinates']) - len(coordinates)
				feature['coordinates'] = coordinates

		elif feature['type'] == "Polygon":
			coordinates = []
			for patch in feature['coordinates']:
				total += len(patch)
				new_patch = simplify_line_partitions(patch, epsilon, nodes)
				deleted += len(patch) - len(new_patch)
				coordinates.append(new_patch)
			if coordinates != feature['coordinates']:
				feature['coordinates'] = coordinates

	if verbose and total > 0:
		message ("\tSimplified %i nodes (%i%%)\n" % (deleted, 100 * deleted/total))



def save_geojson(features, filename, gml_tags=False, verbose=False):
	'''
	Save geojson file, including feature['tags'] as properties.
	Arguments:
		- features: Features from GML to output.
		- filename: File to save.
		- gml_tags: True to include raw GML input data in properties, "prefix" to include "GML_" prefix to the tag.
		- simplify: Simplify geometry if line or polygon by given epsilon factor.
		- verbose: Output messages regarding progress.
	'''
	if verbose:
		message ("Save to '%s' file...\n" % filename)

	json_features = { 
		'type': 'FeatureCollection',
		'features': []
	}

	for feature in features:
		tags = copy.deepcopy(feature['tags'])
		if gml_tags:
			for key, value in iter(feature['data'].items()):
				if gml_tags == "prefix":
					tags["GML_" + key] = value
				else:
					tags[ key ] = value
		entry = {
			'type': 'Feature',
			'geometry': {
				'type': feature['type'],
				'coordinates': feature['coordinates']
			},
			'properties': tags
		}
#		entry['properties']['GML_GEOMETRY'] = feature['type']

		json_features['features'].append(entry)

	file = open(filename, "w")
	json.dump(json_features, file, indent=2, ensure_ascii=False)
	file.close()

	if verbose:
		message ("\t%i features saved\n\n" % len(features))



def convert_to_osm(features, merge_nodes=True, gml_tags=False):
	'''
	Convert feature data to OSM element structure (same as used by Overpass).
	Arguments:
		- features: Features from GML to output. Tags and coordinates will be used.
		- merge_nodes: True to merge nodes with identical location (equal lat/lon).
		- gml_tags: True to include raw GML data from feature['data'], "prefix" to incude "GML_" prefix to the tag.
	Returns OSM elements in same strucutre as used by Overpass.
	'''

	# Create one node
	def create_node(point, tags={}):

		nonlocal osm_id

		updated = False
		if merge_nodes and point in nodes:
			node_id = nodes[ point ]
			index = start_id - node_id - 1
			if not tags or not elements[ index ]['tags']:  # Avoid conflicting tagging
				elements[ index ]['tags'].update(tags)
				updated = True

		if not updated:
			osm_id -= 1
			node_id = osm_id
			nodes[ point ] = node_id
			node_element = {
				'type': 'node',
				'id': node_id,
				'lon': point[0],
				'lat': point[1],
				'tags': tags,
			}
			elements.append(node_element)

		return node_id


	# Create one way and coresponding ndoes
	def create_way(coordinates, tags={}):

		nonlocal osm_id

		# Create way nodes
		way_nodes = []
		for point in coordinates[:-1]:
			node_id = create_node(point)
			way_nodes.append(node_id)

		if coordinates[-1] == coordinates[0]:
			way_nodes.append(way_nodes[0])  # Closure also if merge_nodes is False
		else:
			node_id = create_node(coordinates[-1])
			way_nodes.append(node_id)

		osm_id -= 1
		way_element = {
			'type': 'way',
			'id': osm_id,
			'nodes': way_nodes,
			'tags': tags,
		}
		elements.append(way_element)
		return osm_id


	elements = []
	nodes = {}  # Will contain id of each point
	start_id = -1000  # First id will be -1001
	osm_id = start_id

	for feature in features:

		tags = copy.deepcopy(feature['tags'])
		if gml_tags:
			for key, value in iter(feature['data'].items()):
				if gml_tags == "prefix":
					tags["GML_" + key] = value
				else:
					tags[ key ] = value

		if feature['type'] == "Point":
			create_node(feature['coordinates'], tags)

		elif feature['type'] == "LineString":
			create_way(feature['coordinates'], tags)

		elif feature['type'] == "Polygon":

			if len(feature['coordinates']) == 1:
				create_way(feature['coordinates'][0], tags)  # Single way
			else:
				# Create relation member ways
				role = "outer"
				members = []
				for way in feature['coordinates']:
					way_id = create_way(way)
					member = {
						'type': 'way',
						'ref': way_id,
						'role': role,
					}
					members.append(member)
					role = "inner"

				osm_id -= 1
				relation_element = {
					'type': 'relation',
					'id': osm_id,
					'members': members,
					'tags': tags
				}
				elements.append(relation_element)

	return elements



def indent_tree(elem, level=0):
	'''
	Indent XML output (built-in from Python 3.9)
	'''
	i = "\n" + level*"  "
	if len(elem):
		if not elem.text or not elem.text.strip():
			elem.text = i + "  "
		if not elem.tail or not elem.tail.strip():
			elem.tail = i
		for elem in elem:
			indent_tree(elem, level+1)
		if not elem.tail or not elem.tail.strip():
			elem.tail = i
	else:
		if level and (not elem.tail or not elem.tail.strip()):
			elem.tail = i



def save_osm (elements, filename, create_action=False, generator="gml2osm", verbose=False):
	'''
	Output OSM file from elements dict with same structure as from Overpass.
	Arguments:
		- elements: List of OSM elements in same structure as from Overpass.
		- filename: File to be created.
		- create_action: True if action=modify attribute should be included for all elements.
		- generator: Name of source program.
		- verbose: True to display messages about progress.
	'''

	# Generate XML

	if verbose:
		message ("Saving file ...\n")

	xml_root = ET.Element("osm", version="0.6", generator=generator, upload="false")

	for element in elements:

		if element['type'] == "node":
			xml_element = ET.Element("node", lat=str(element['lat']), lon=str(element['lon']))

		elif element['type'] == "way":
			xml_element = ET.Element("way")
			if "nodes" in element:
				for node_ref in element['nodes']:
					xml_element.append(ET.Element("nd", ref=str(node_ref)))

		elif element['type'] == "relation":
			xml_element = ET.Element("relation")
			if "members" in element:
				for member in element['members']:
					xml_element.append(ET.Element("member", type=member['type'], ref=str(member['ref']), role=member['role']))

		if "tags" in element:
			for key, value in iter(element['tags'].items()):
				xml_element.append(ET.Element("tag", k=key, v=value))

		xml_element.set('id', str(element['id']))
#		xml_element.set('visible', 'true')

		if "user" in element:  # Existing element
			xml_element.set('version', str(element['version']))
			xml_element.set('user', element['user'])
			xml_element.set('uid', str(element['uid']))
			xml_element.set('timestamp', element['timestamp'])
			xml_element.set('changeset', str(element['changeset']))

		if "action" in element:
			xml_element.set('action', element['action'])
		elif create_action:
			xml_element.set('action', 'modify')

		xml_root.append(xml_element)
		
	# Output OSM/XML file

	xml_tree = ET.ElementTree(xml_root)
	indent_tree(xml_root)
	xml_tree.write(filename, encoding="utf-8", method="xml", xml_declaration=True)

	if verbose:
		message ("\t%i elements saved to file '%s'\n" % (len(elements), filename))



# Main program

if __name__ == '__main__':

	message ("--- gml2osm ---\n\n")

	# Get arguments

	# Get URL
	if len(sys.argv) > 1:
		url = sys.argv[1]
	else:
		sys.exit ("*** Please provide url (or zip url + filename)\n\n")

	filename = ""
	object_filter = []

	# Get filename, unless only one is available in zip
	if len(sys.argv) > 2 and "-" not in sys.argv[2]:
		if "_GML" in sys.argv[2]:
			filename = sys.argv[2]
		else:
			object_filter = [ sys.argv[2] ]  # 2nd argument could be object type to filter

	# Get which object to filter, if any
	if len(sys.argv) > 3 and "-" not in sys.argv[3]:
		object_filter = [ sys.argv[3] ]

	# Output filename

	if filename:
		out_filename = filename.lower().replace("_GML", "").replace(".gml", "") + ".geojson"
	else:
		out_filename = url[ url.rfind("/") + 1 : ].lower().replace("_GML", "").replace(".gml", "").replace(".zip", "") + ".geojson"

	features = load_gml(url, filename=filename, object_filter=object_filter, verbose=True)

	if "-osm" in sys.argv:
		elements = convert_to_osm(features, merge_nodes=True, gml_tags=True)
		out_filename = out_filename.replace(".geojson", "") + ".osm"
		save_osm(elements, out_filename, verbose=True)
	else:
#		simplify_features(features, 0.2, verbose=True)
		save_geojson(features, out_filename, gml_tags=True, verbose=True)

	message ("\n")
