# Replaces the ctf values in input star file with the values in a reference star file

import argparse
import os
from star import *


def parse_args():
	parser = argparse.ArgumentParser(description="Replaces the ctf values in input star file with the values in a reference star file.")
	parser.add_argument('--input', metavar='f1', type=str, nargs=1, required=True, help="particle file whose ctf values will be changed")
	parser.add_argument('--reference', metavar='f2', type=str, nargs=1, required=True, help="particle file whose ctf values will be used as a reference")
	parser.add_argument('--output', metavar='o', type=str, nargs=1, help="output file name")

	return parser.parse_args()


def main(reference_path,input_path):

	# parameters that are relevant to the CTF estimation
	ctf_params = ['DefocusU','DefocusV','DefocusAngle','CtfFigureOfMerit','SphericalAberration','AmplitudeContrast']


	# dictionary of micrograph name to ctf values
	# key = micrograph name
	# value = ctf values in 4-ple: DefocusU, DefocusV, DefocusAngle, CtfFOM
	mic_to_ctf = {}

	output = ''

	print "Reading in reference CTF estimates"
	ref_star = starFromPath(reference_path)

	params_to_replace = [ params for params in ctf_params if params in ref_star.lookup ]


	for line in ref_star.body:
		mic_root = rootname(ref_star.getMic(line))

		if mic_root in mic_to_ctf:
			continue
		else:
			mic_to_ctf[mic_root] = ref_star.valuesOf(params_to_replace, line)


	print "Reading input file"

	input_star = starFromPath(input_path)
	output += input_star.textHeader()



	fields_to_replace = input_star.numsOf( params_to_replace )
	for line in input_star.body:
		values = line.split()

		mic_root = rootname(input_star.valueOf('MicrographName',line))

		for index,field in enumerate(fields_to_replace):
			values[field] = mic_to_ctf[mic_root][index] or values[field]

		output += makeTabbedLine(values)

	return output






if __name__ == '__main__':


	args = parse_args()

	input_path = args.input[0]
	reference_path = args.reference[0]

	if args.output:
		output_path = args.output[0]
	else:
		root, ext = os.path.splitext(input_path)
		output_path = root + '_replace_ctf' + ext
	

	main(reference_path, input_path)

	with open(output_path, 'w') as output_file:
		output_file.write(output)

	print "Done!"




