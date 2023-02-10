# gml2osm
Convert GML to geojson and OSM format.

### Usage ###

Usage: <code>python3 gml2osm.py \<url\> [filename] [object type] [-osm]</code>

Arguments:
* *url* - URL of GML file, including zip files, for example from [this folder](https://nedlasting.geonorge.no/geonorge/).
* *filename* - Optional filename in zip folder.
* *object type* - Optional name of object type in GML to filter. If omitted all object types will be loaded.
* *<code>-osm</code>* - Optional, will output in .osm file format instead of geojson.

The *utm.py* file should be located in the same folder as *n50osm.py* when running the program.

### Notes ###

* The *gml2osm.py* program loads a GML file and converts it into geojson or OSM format.
* The program has been optimized for and has been tested with Kartverket/GeoNorge GML files.
* The code may be run stand alone but it is also designed to be imported into other Python programs to load, parse, simplify and save GML data. Data transformations will then take place in the other programs.

### Changelog

* 0.3: Initial version.

### References ###

* [Kartverket/GeoNorge download folder](https://nedlasting.geonorge.no/geonorge/)
