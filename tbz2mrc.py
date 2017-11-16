#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:58:01 2016

@author: Oleg Kuybeda
"""

# %%
# from   myutils import scratch
# from   myutils import filenames as fn
# from   myutils.formats import dm4tomrc, stackmrcs, untbz, transpose_mrc
# from   myutils import mrc
from   os.path import join, dirname, splitext
# from   myutils.utils import sysrun, tprint
import shutil
from   star import star
# from   myutils import mpi
from   functools import partial
import argparse
import glob
import os
import errno
import multiprocessing
import subprocess as sp
import numpy as np

def mkdir_assure(path,mode=None):
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
        if mode is not None:
            os.chmod(path,mode)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # raises the error again

def sysrun(cmd, **kwargs):
    ce   = kwargs.pop('err_check', True)
    verb = kwargs.pop('verbose', False)
    if verb:
        print cmd
    process = sp.Popen(cmd,shell=True,close_fds=True,
                       stdout=sp.PIPE,stderr=sp.PIPE)
    # wait for the process to terminate
    out, err = process.communicate()
    errcode  = process.returncode
    if (errcode != 0) and ce:
        print(err)
        status = False
    else:
        status = True
    return out,err,status

def replace_ext(path, new_ext):
   root, ext = os.path.splitext(path)
   return root + new_ext

def file_only(path):
    return replace_ext(os.path.basename(path),'')

def untbz(tbzfile,untbzdir,nth):
    # nth = kwargs.pop('nthreads',multiprocessing.cpu_count())
    # ''' Uncompresses list of files from a tbz archive '''
    tarfile = replace_ext(tbzfile, '.tar')
    # Unzip tbz
    cmd     = "lbzip2 -f -d -n %d %s" % (nth,tbzfile)
    out,err,status = sysrun(cmd)
    if not status:
        raise RuntimeError(out + err)
    # Unzip tar
    cmd = "tar -xf " + tarfile + " --directory=" + untbzdir
    out,err,status = sysrun(cmd)
    if not status:
        raise RuntimeError(out + err)
    os.remove(tarfile)
    # Remove tar
    # cmd = "rm -f " +  tarfile
    # out,err,status = sysrun(cmd)
    # if not status:
    #     raise RuntimeError(out + err)

def dm4tomrc(fsrc,fdst):
    cmd     = "dm2mrc %s %s " % (fsrc, fdst)
    out,err,status = sysrun(cmd)
    if not status:
        print out+err
        assert(status)

# def transpose_mrc(srcmrc,dstmrc):
#     g,psize = mrc.load_psize(srcmrc)
#     g       = np.ascontiguousarray(g.swapaxes(-1,-2))
#     mrc.save(g,dstmrc,pixel_size=psize)

def transpose_mrc(fsrc,fdst):
    cmd     = "newstack -ro 90 %s %s " % (fsrc, fdst)
    out,err,status = sysrun(cmd)
    if not status:
        print out+err
        assert(status)

def multgains(srcmrc, gainmrc, dstmrc, transpose_gain=False):
    cmd = 'clip mult -n 16 %s %s %s' % (srcmrc, gainmrc, dstmrc)
    # srcshape = mrcshape(srcmrc)  # mrc.shape(srcmrc)
    # gainshape = mrcshape(gainmrc)  # mrc.shape(gainmrc)
    # ugly, but has to be there, as input data formats keep changing
    # if srcshape[1] != gainshape[1]:
    if transpose_gain:
        transpose_mrc(gainmrc, gainmrc)
    out, err, status = sysrun(cmd)
    if not status:
        print out + err
        assert (status)

def extract_tbz(tbzfile,dstdir,gaindm4,nth=1,transpose_gain=False):
    mname   = os.path.basename(tbzfile)
    nonly   = file_only(mname)
    tmpdir  = os.path.join(dstdir,nonly)
    mkdir_assure(tmpdir)

    dsttbz  = join(tmpdir, mname)
    dstgain = join(tmpdir, nonly + '_gain.dm4')
    gainmrc = join(tmpdir, nonly + '_gain.mrc')
    micdm4  = join(tmpdir, nonly +'.dm4')
    micmrc  = join(tmpdir, nonly + '.mrc')
    dstmrc  = join(dstdir, nonly + '.mrc')
    # copy and extract tbz
    shutil.copyfile(tbzfile, dsttbz)
    untbz(tbzfile,tmpdir,nth)
    # os.remove(dsttbz)
    # convert everything to mrc
    shutil.copyfile(gaindm4, dstgain)
    dm4tomrc(dstgain, gainmrc)
    dm4tomrc(micdm4, micmrc)
    # os.remove(micdm4)
    # os.remove(dstgain)
    # apply gain
    multgains(micmrc,gainmrc,dstmrc,transpose_gain)
    # clean up
    shutil.rmtree(tmpdir)
    # os.remove(gainmrc)
    # remove backup files
    # out, err, status = sysrun('rm %s' % os.path.join(dstdir,'*.mrc~'))
    # if not status:
    #     print out + err
    #     assert (status)

def get_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     description='Running uncompress tbz files and apply gains on them.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog="Example: tbz2mrc -i dir1 -o dir2 -g gain.dm4 -r ")

    parser.add_argument('-i', '--input_dir', help='Input directory with tbz-compressed filenames.',
                        default=argparse.SUPPRESS, type=str, required=True)

    parser.add_argument('-o', '--output_dir', help='Output directory for storing mrc files',
                        default=argparse.SUPPRESS, type=str, required=True)

    parser.add_argument('-g', '--gain_file', help='Gain file name (in .dm4 format) that shall be stored in the input directory',
                        default=argparse.SUPPRESS, type=str, required=True)

    parser.add_argument('-r', '--rotate_gain', help='Flag whether to rotate (transpose) gain file before applying it.',
                        default=False, type=bool)

    return parser


##%%##############################################


###### Main starts here #######################################
if __name__ == "__main__":

    # Parse input and obtain all params
    args, unknown = get_parser().parse_known_args()
    kwargs = vars(args)
    if len(unknown) > 0:
        print "Unkown arguments %s !!! \n Quitting ..." % unknown
        quit()

    srcdir  = kwargs['input_dir']
    dstdir  = kwargs['output_dir']
    gaindm4 = os.path.join(srcdir,kwargs['gain_file'])
    transpose_gain = kwargs['rotate_gain']
    mkdir_assure(dstdir)

    alltbz  = glob.glob(os.path.join(srcdir, '*.tbz'))
    print "Found %d tbz files in %s" % (len(alltbz),srcdir)
    if len(alltbz) == 0:
        exit()
    if not os.path.exists(gaindm4):
        print "Gain file %s not found !" % gaindm4
        exit()

    ncpu  = multiprocessing.cpu_count()
    print "Extracting, converting and gain correcting %d movies using %d CPUs" % (len(alltbz),ncpu)
    pool  = multiprocessing.Pool(np.minimum(1*ncpu,len(alltbz)))
    pool.map(partial(extract_tbz,dstdir=dstdir,gaindm4=gaindm4,nth=4,transpose_gain=True),alltbz)
    pool.close(),pool.join(),pool.terminate()

    # map(partial(extract_tbz,dstdir=dstdir,gaindm4=gaindm4,nth=4,transpose_gain=True),alltbz)



    ################## JUNK ######################################

    # # parser.add_argument('-j', '--nthreads', help='Number of threads', default=4, type=int, required=False)
    # parser.add_argument('-s','--save_movies', help='Flag to save aligned movies',
    #                    default=True, type=bool, required=False)
    # parser.add_argument('-d', '--do_dose', help='Flag to do  dose weighting',
    #                     default=False, type=bool, required=False)
    # parser.add_argument('-a', '--save_aligned_movies', help='Flag whether to save aligned movies',
    #                     default=False, type=bool, required=False)
    # parser.add_argument('-df', '--dose_per_frame', help='',
    #                     default=0.0, type=float, required=False)
    # parser.add_argument('-v', '--voltage', help='Voltage used for dose weighting',
    #                     default=0.0, type=float, required=False)
    # parser.add_argument('-p', '--pre_exp', help='Pre exposure used for dose weighting',
    #                     default=0.0, type=float, required=False)
    # parser.add_argument('-f', '--first_frame_sum', help='First frame to average (starting from 0)',
    #                     default=0, type=int, required=False)
    # parser.add_argument('-l', '--last_frame_sum', help='Number of last frame to average (starting from 0)',
    #                     default=0, type=int, required=False)
    # parser.add_argument('-un', '--unblur_exe', help='Path to unblur executable.',
    #                     default=argparse.SUPPRESS, type=str, required=True)
    # parser.add_argument('-sm', '--summovie_exe', help='Path to summovie executable.',
    #                     default=argparse.SUPPRESS, type=str, required=True)

    # unblurexe = kwargs['unblur_exe']
    # sumexe = kwargs['summovie_exe']
    # nth = kwargs['nthreads']
    # do_aligned_movies = kwargs['save_aligned_movies']
    # dodose = kwargs['do_dose']
    # dose_per_frame = kwargs['dose_per_frame']
    # vol = kwargs['voltage']
    # pre_exp = kwargs['pre_exp']
    # first_frame = kwargs['first_frame_sum']
    # last_frame = kwargs['last_frame_sum']
    # dosummovie = last_frame != 0 or first_frame != 0
    # # call main function with all params

# def tbz2mrc(srcdir, tbzname, dstext, **kwargs):
#     ''' Unzips and converts to mrc. Cleans the tbz and the dm4 files '''
#     # copy gains and convert to mrc
#     mname = fn.file_only(tbzname)
#     # print mname
#     sdir = dirname(tbzname)
#     untbzdir = fn.replace_ext(tbzname, '')
#     sname = join(sdir, mname)
#     dstmrc = sname + dstext
#     fn.mkdir_assure(untbzdir)
#     untbz(tbzname, untbzdir, **kwargs)
#
#     # root,srcext = splitext(glob.glob(join(untbzdir, '*.dm4'))[0])
#     untbzfiles = glob.glob(join(untbzdir, mname + '*'))
#     root, srcext = splitext(untbzfiles[0])
#
#     gainmrc = gain2mrc(srcdir)
#     srcmics = join(untbzdir, mname) + '*' + srcext
#     mrctmp = sname + '_tmp.mrc'
#
#     if srcext == '.dm4':
#         dm4tomrc(srcmics, mrctmp)
#     else:
#         assert (srcext == '.mrc')
#         # if len(untbzfiles) > 1:
#         stackmrcs(srcmics, mrctmp)
#     # else:
#     #  copyfile(untbzfiles[0],mrctmp)
#     # transpose gain mrc data
#     # transpose_mrc(gainmrc,gainmrc)
#
#     # multiply all frames by gains
#     # print mrctmp, gainmrc, dstmrc
#     multgains(mrctmp, gainmrc, dstmrc)
#     out, err, status = sysrun('rm -rf %s' % untbzdir)
#     assert (status)
#     out, err, status = sysrun('rm ' + mrctmp)
#     assert (status)
#     # remove gains
#     out, err, status = sysrun('rm ' + gainmrc)
#     assert (status)

# from   shutil import copyfile
# import mrcfile

# MOVSUFF = '.mrc'
# AVGSUFF = '_avg.mrc'
# DIFF_SUFF = '_avg.mrc'
# ALNSUFF = '_aligned.mrc'
# SCRATCH_DIR = 'Unblur'
# MOVIE_DIR = 'Movies'
# # wildcard for gains filename
# GAINS_KEY = 'Gain Ref'
# MOVIE_KEY = '_rlnTbzMovieName'


####### FUNCTIONS ####################################

# def mrcshape(mrcname):
#     # with mrcfile.open(mrcname) as mrc:
#     #	shape = mrc.data.shape
#     # return shape
#     return mrc.shape(mrcname)


# def tbz2mrc_name(ftbz):
#     mname = fn.file_only(ftbz)
#     sdir = scratch.join(SCRATCH_DIR)
#     mrcin = join(sdir, mname + MOVSUFF)
#     return mrcin

    # def gain2mrc(basedir):
    #     # destination dir
    #     # sdir = scratch.join(SCRATCH_DIR)
    #     path = join(basedir, '*%s*.dm4' % GAINS_KEY)
    #     gaindm4 = glob.glob(path)
    #     if len(gaindm4) == 0:
    #         raise IOError('Gain dm4 file not found in %s !!!' % basedir)
    #     gaindm4 = os.path.abspath(gaindm4[0])
    #     gainmrc = join(sdir, 'gain.mrc')
    #     dstdm4 = join(sdir, 'gain.dm4')
    #     # link dm4 with a simpler filename
    #     cmd = "ln -s \'%s\' %s" % (gaindm4, dstdm4)
    #     out, err, status = sysrun(cmd)
    #     assert (status)
    #     dm4tomrc(dstdm4, gainmrc)
    #     cmd = "rm " + dstdm4
    #     out, err, status = sysrun(cmd)
    #     assert (status)
    #     return gainmrc


    # print  "###########################################"
    # print  do_aligned_movies
    # exit()

    # main_mpi(dstdir, starfile, unblurexe, sumexe, nth, do_aligned_movies, dodose, dosummovie,
    #          dose_per_frame, vol, pre_exp, first_frame, last_frame)


