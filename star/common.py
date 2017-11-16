import os


#===============================================================================================================
#COMMON FUNCTIONS FOR GENERAL TASKS
#===============================================================================================================

# returns the first word of a string
def firstWord(line):
	try:
		return line.split()[0]
	except IndexError:
		return ""

# strips a given prefix substring from a string
# similar to str.lstrip, which applies only to stripping characters
def stripPrefix(string, prefix):
    if string.startswith(prefix):
        return string[len(prefix):]
    else:
    	return string

# returns the rootname of a given path
# e.g. rootname "/data/autoem2/guot/filename.txt" will return 'filename' 
def rootname(path):
	return os.path.splitext(os.path.basename(path))[0]


# makes a directory if it does not already exist
# similar to mkdir -p
def makeDir(directory):
	if not os.path.exists(directory):
	    os.makedirs(directory)