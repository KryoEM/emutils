# Converts between .star (Relion) and .box (EMAN2) coordinate files, as well as .star particle files.


from star import *
import argparse
import os



# Parse arguments
def parse_args():
	parser = argparse.ArgumentParser(description='Converts between .star (Relion) and .box (EMAN2) coordinate files. Also accepts .star particle files as inputs.')
	parser.add_argument('--input', metavar='file', type=str, nargs='+', required=True, help="Input file(s) to convert. Wildcards accepted, e.g. 'box/*.box'. Required field.")
	parser.add_argument('--dest', metavar='dir', type=str, default="./", help="Destination directory for newly generated files, e.g. 'Micrographs/'. [Default: current directory]")
	parser.add_argument('--suffix', metavar='str', type=str, default='', help="Suffix for output file(s), e.g. '_pickall'. [Optional]")
	parser.add_argument('--output_type', metavar='str', type=str, required=True, choices=['star_coord', 'box'], help="Type of coordinate file to output. [Options: 'star_coord', 'box']")

	return parser.parse_args()



def main(input_paths, output_dir, output_suffix, output_type):
	print "Reading coordinates in from input file(s)"
	coords_map = readCoordinates(input_paths)
	print "Writing coordinates out to " + output_dir 
	writeCoordinates(coords_map, output_dir, output_suffix, output_type)
	print "Done!"





# Determine the file type of a given path as a .star particle file, .star coordinate file, or a .box coordinate file.
def fileType(path):
	if os.stat(path).st_size == 0:
		print "WARNING: Empty file " + path
		return None
	if isParFile(path):
		return 'star_parfile'
	if isCoordFile(path):
		return 'star_coord'
	if len(open(path).readline().split()) == 4:
		return 'box'
	else:
		raise UserWarning('ERROR: Input file ' + path + ' does not appear be an appropriate .star or .box file. Check that your file path is correct and that the file is formatted correctly (should have an appropriate header if a star file, and should have four columns if a box file). You can override this by indicating the file type with the flag --ftype_in.')



# Returns a dictionary listing the coordinates on each micrograph, given a list of file paths to .box or .star files.
# key: micrograph name
# value: list of coordinate pairs, represented as a tuple. Thus each tuple is a coordinate pair, and each element of the list represents a single position on the micrograph.
# Coordinates are expressed as the position of the center of the particle (consistent with the system used in Relion).
def readCoordinates(paths):
	coordinate_tbl = {}

	# helper function: add a single entry to the coordinate table
	def addCoords(coord_tbl, micName, coordX, coordY):
		fl_coordX = float(coordX)
		fl_coordY = float(coordY)

		if micName in coord_tbl:
			coord_tbl[micName].append((fl_coordX,fl_coordY))
		else:
			coord_tbl[micName] = [(fl_coordX,fl_coordY)]


	for input_path in paths:
		file_type = fileType(input_path)

		if file_type == None:
			continue

		if file_type == 'box':
			micName = rootname(input_path)

			with open(input_path) as box_file:
				for line in box_file:
					line_split = [ float(i) for i in line.split() ]
					coordX = line_split[0] + (line_split[2] / 2)
					coordY = line_split[1] + (line_split[3] / 2)

					addCoords(coordinate_tbl, micName, coordX, coordY)


		# otherwise, assume this is a star file
		else:
			# if this is a coordinate .star file, then the corresponding micrograph name is the rootname fo the coordinate .star file 
			micName_perfile = rootname(input_path) if file_type == 'star_coord' else None

			star_file = starFromPath(input_path)
			for line in star_file.body:
				# if we haven't determined the micrograph name yet, determine it on a per-line basis
				micName_perline = micName_perfile or rootname(star_file.valueOf('MicrographName', line))
				coordX, coordY = star_file.valuesOf(['CoordinateX', 'CoordinateY'], line)

				addCoords(coordinate_tbl, micName_perline, coordX, coordY)


	return coordinate_tbl

# Given a dictionary listing the coordinates in each micrograph, writes it out as coordinate files (either in .star or .box format)
def writeCoordinates(coord_tbl, out_dir, suffix, type):
	makeDir(out_dir)


	if type == 'box':
		for mic in coord_tbl:
			box_path = out_dir + mic + suffix + '.box'

			with open(box_path, 'w') as box_file:
				for coordX,coordY in coord_tbl[mic]:
					box_file.write(makeTabbedLine([coordX, coordY, 0, 0]).lstrip() + '\n')

	elif type == 'star_coord':
		for mic in coord_tbl:
			star_path = out_dir + mic + suffix + '.star'

			with open(star_path, 'w') as star_file:
				star_file.write(textHeader( ['CoordinateX', 'CoordinateY'] ))

				for coordX,coordY in coord_tbl[mic]:
					star_file.write(makeTabbedLine([coordX, coordY]))
	else:
		print "ERROR: No valid output file type indicated."





# Execute if this script is called
if __name__ == '__main__':
	args = parse_args()

	input_paths = args.input
	output_dir = os.path.realpath(args.dest) + "/"
	output_suffix = args.suffix
	output_type = args.output_type

	main(input_paths, output_dir, output_suffix, output_type)


