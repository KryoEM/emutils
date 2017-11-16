import os
import sys
# Please import specific functions
from   common import firstWord,stripPrefix

#===============================================================================================================
# HANDLING STAR FILES
#===============================================================================================================

def getlist(starfile,key):
	''' returns a list of strings that follows key in startfile '''
	with open(starfile) as f:
		content = f.readlines()
	tf   = [l.find(key)==0 for l in content]
	idxs = [i for i, x in enumerate(tf) if x]
	if len(idxs) == 0:
		raise ValueError("Key %s not found in star file %s" % (key,starfile))
	items = [content[i].strip() for i in range(idxs[0]+1,len(content))]
	items = [c for c in items if len(c)>0]
	return items

def isStar(path):
	''' returns if the file at path 'path' is likely to be a star file or not by looking for a parameter definition (e.g. "_rlnMicrographName #2")'''
	with open(path) as file:
		for line in file:
			if isParameterDef(line):
				return True
		# reached end of file without finding a parameter definition
		return False


def isParFile(path):
	''' returns if the file at path 'path' is likely to be a star file containing particles or not
   	 criterion: if parameter 'ImageName' exists '''
	if isStar(path):
		star_file = starFromPath(path)
		return star_file.paramExists('ImageName')

def isCoordFile(path):
	''' returns if the file at path 'path' is likely to be a star file containing only coordinate information
	    criterion: has 'CoordinateX' and 'CoordinateY' parameters, but NOT 'MicrographName' parameter '''
	if isStar(path):
		star_file = starFromPath(path)
		return not star_file.paramExists('MicrographName') and star_file.paramExists('CoordinateX') and star_file.paramExists('CoordinateY')


def isParameterDef(line):
	''' returns if a line is a parameter definition line e.g. "_rlnMicrographName #2" '''
	return firstWord(line).startswith("_rln")

def isHeader(line):
	''' #returns if a line could be part of the header '''
	first_word = firstWord(line)
	if first_word == "" or first_word.startswith("data_") or first_word.startswith("loop_") or isParameterDef(line):
		return True
	else:
		return False



#===============================================================================================================
# READING IN STAR FILES
#===============================================================================================================


# Returns a lookup table for parameter names and their corresponding column number in a given star file header
# key:value pair is parameter_name:column_number
def headerLookup(star_path):	
	with open(star_path) as file_object:

		lookup = {}

		for line in file_object:

			if isParameterDef(line):
				words = line.split()
				param_name = stripPrefix(words[0], '_rln')
				column = int(words[1].strip('#')) - 1
				lookup[param_name] = column
				# TO DO
				# warning if param already exists

			#stop searching if we are no longer on a header line, and the header list is populated
			elif not isHeader(line) and lookup != {}:
				break

	return lookup


# A generator that yields non-empty lines from the body of a star file
def cleanBody(star_path):
	with open(star_path) as file_object:
		for line in file_object:
			if not isHeader(line):
				yield line




# Returns a Star object from a given star file path
def starFromPath(path):
	return Star(headerLookup(path), cleanBody(path))






#===============================================================================================================
# STAR OBJECTS
#===============================================================================================================

# A Star object consists of two parts:
# (1) a lookup table converting parameter names to their column number
# (2) a generator that yields each line of the star's body

class Star:

	# A Star object is initialized by being given a parameter lookup table, and a body generator
	# Therefore, there does not need to be an existing .star file on the filesystem in order to create a Star object
	# (useful when you want to create a copy of a Star object with modified traits).
	def __init__(self, header_lookup, body_generator):
		#lookup table for k: parameter name, v: column number 
		self.lookup = header_lookup
		#generator that produces body lines from the star file. Does not return empty lines.
		self.body = body_generator


	# Comparators
	# Two Star objects are equal when they have equal lookup tables and equal contents in their generators
	def __eq__(self, other):
		return self.lookup == other.lookup and list(self.body) == list(other.body)
	# Two Star objects are not equal when they do not satisfy the equality requirements
	def __ne__(self,other):
		return not(self.__eq__)


	# Return whether or not the parameter param exists in the star file
	def paramExists(self,param):
		return (param in self.lookup)


	# Return the column number of a given parameter (count begins at 0)
	def numOf(self,param):
		return self.lookup[param]

	# Return the column numbers of a list of parameters (count begins at 0)
	def numsOf(self, params):
		return( map (lambda param: self.numOf(param), params) )
			

	# Find the value associated with the given parameter on a given line.
	# Returns None if the value could not be found.
	def valueOf(self, param, line):
		values = line.split()
		try:
			return values[self.lookup[param]]
		except KeyError:
			print >> sys.stderr, "WARNING: Could not find parameter " + str(param) + " in the header of the given star file."
			return None
		except IndexError:
			# if the list of values is not empty and we have an index error, the line ended before the appropriate column number.
			if values:
				print >> sys.stderr, 'WARNING: value ' + param + ' missing from the following line: '
				print >> sys.stderr, line
			# if the list of values is empty, the line was probably empty.
			else:
				print >> sys.stderr, 'WARNING: Empty lines found in body of star file. If problems arise, please try deleting the empty lines.'
			return None	

	# Find the values associated with the given parameters (provided as a list of strings) in a given line
	# Returns None if the value could not be found.
	def valuesOf(self, params, line):
		return map (lambda param: self.valueOf(param, line), params)

	def pairsOf(self,line):
		pairs = {}
		for key in self.lookup:
			pairs.update({key:self.valueOf(key,line)})
		return pairs

	# Returns the micrograph name of a given line
	def getMic(self, line):
		return self.valueOf("MicrographName", line)

	def readLines(self):
		return [self.pairsOf(line) for line in self.body]

	# Returns the header of this Star object as a printtable string
	def textHeader(self):
		# sorted list of parameters, in order of column number
		parameters = [ param for param in sorted(self.lookup, key=self.lookup.get) ]

		header = "\ndata_\n\nloop_"

		for index, parameter in enumerate(parameters, start=1):
			header += "\n_rln" + parameter + " #" + str(index)

		return header



#===============================================================================================================
# WRITING OUT STAR FILES
#===============================================================================================================

# Given a list of values, return them in a single tab-delimited line
def makeTabbedLine(values):
	line = "\n"
	for value in values:
		line += str(value) + "\t"
	line = line.rstrip("\t") 
	return line


#===============================================================================================================
# FILTERING STAR OBJECTS
#===============================================================================================================

# Filters a Star object so that only the lines that satisfy @test for the parameter @param are returned

# e.g.
# filterStar(input.star, AutopickFigureOfMerit, lambda x: float(x) > 1)
# Will return a Star object with only the lines in which the autopick FOM score is greater than 1.
def filterStar(star, param, test):
	filtered_body = (line for line in star.body if test(star.valueOf(param, line)))
	return Star(star.lookup, filtered_body)


# Filters a Star object so that only the lines with autopick figures of merit greater than @threshold are returned
def filterStarByAutopickFOM(star_file, threshold):
	return filterStar(star_file, AutopickFigureOfMerit, lambda x: float(x) > threshold)



#===============================================================================================================
# CHANGING STAR OBJECTS
#===============================================================================================================



#===============================================================================================================
# DEPRECATED
#===============================================================================================================

# DEPRECATED: MAINTAINED FOR BACKWARD COMPATIBILITY
# Use Star.textHeader() instead
# Given a list of parameter names, returns a printtable Relion header string 
def textHeader(parameters):
	header = "\ndata_\n\nloop_"

	for index, parameter in enumerate(parameters, start=1):
		header += "\n_rln" + parameter + " #" + str(index)

	return header
