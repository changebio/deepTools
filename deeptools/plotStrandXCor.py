#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from deeptools import parserCommon
from deeptools.mapReduce import mapReduce
from deeptools import bamHandler
from deeptools.getFragmentAndReadSize import get_read_and_fragment_length
from deeptoolsintervals import GTF


def parseArguments():
    parentParser = parserCommon.getParentArgParse(binSize=False)
    bamParser = parserCommon.read_options(extend=False, center=False, fragLen=False)
    requiredArgs = get_required_args()
    optionalArgs = get_optional_args()
    parser = \
        argparse.ArgumentParser(
            parents=[requiredArgs, optionalArgs, parentParser, bamParser],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='This tool takes an alignment of reads or fragments '
            'as input (BAM file) and computes the cross-correlation of the '
            'signal between the strands. This is useful for assessing ChIP '
            'quality. Typically two peaks are observed, one at the read '
            'length and another at approximately the expected fragment length. '
            'The higher the fragment length peak versus the read length peak, '
            'the better the ChIP quality. This plot and (optionally) the '
            'numbers underlying it are output. Additionally, the normalized '
            'strand coefficient (NSC) and the relative strand correlation (RSC) '
            'are output. See Landt et al. 2012 for further discussion of these. '
            'Note that only the 5-prime most mapped base of each alignment is '
            'used. For paired-end datasets, it is advisable to use only a '
            'single read from each pair.',
            usage='An example usage is: '
            '$ bamStrandXCor -b reads.bam -o xcor.png',
            add_help=False)

    return parser


def get_required_args():
    parser = argparse.ArgumentParser(add_help=False)

    required = parser.add_argument_group('Required arguments')

    # define the arguments
    required.add_argument('--bam', '-b',
                          help='BAM file to process',
                          metavar='BAM file',
                          required=True)

    required.add_argument('--plotFile', '-o',
                          help='File name to save the plot to. '
                          'The extension determines the file format.',
                          metavar='FILE',
                          required=True)

    return parser


def get_optional_args():
    parser = argparse.ArgumentParser(add_help=False)

    optional = parser.add_argument_group('Optional arguments')

    optional.add_argument('--maxLag', '-l',
                          help='The maximum lag to use for computing the cross-correlation. Apparent peaks on one strand will be shifted by at most this amount.',
                          default=500,
                          type=int)

    optional.add_argument('--minSeparation',
                          help='The minimum distance between the median read length and the fragment length. This is used to more easily find the read and background correlation peaks. The minimum value is 1.',
                          default=10,
                          type=int)

    optional.add_argument('--plotFileFormat',
                          metavar='FILETYPE',
                          help='Image format type. If given, this option '
                          'overrides the image format based on the plotFile '
                          'ending. The available options are: png, '
                          'eps, pdf and svg.',
                          choices=['png', 'pdf', 'svg', 'eps'])

    optional.add_argument('--plotTitle', '-T',
                          help='Title of the plot, to be printed on top of '
                          'the generated image. Leave blank for no title.',
                          default='')

    optional.add_argument('--outFileNameData',
                          help='File name to save the data '
                          'underlying data for the plot, e.g., '
                          'values.tab.',
                          metavar="values.tab")

    optional.add_argument('--outFileQualityMetrics',
                          help='File name to which the NSC/RSC and related '
                          'metrics are saved. The columns of this file are: '
                          'the file name (FileName), '
                          'the number of reads sampled (numReads), '
                          'the position of the background peak (estFragLen), '
                          'correlation coefficient at the peak (corr_estFragLen), '
                          'read length/phantom peak position (phantomPeak), '
                          'correlation at the phantomPeak location (corr_phantomPeak), '
                          'ratio of median read length to phantom peak length (argmin_corr), '
                          'minimum correlation (min_corr), '
                          'normalized strand cross-correlation (NSC), '
                          'relative strand cross-correlation (RSC), '
                          'quality tag according to RSC (-2: very low, -1: low, 0: medium, 1: high, 2: very high)',
                          metavar="quality.tab")

    optional.add_argument('--label',
                          help='If --outFileQualityMetrics is used, there is a '
                          'column with the file name. A different label can be '
                          'specified with this option.',
                          metavar='foo.bam')

    optional.add_argument("--help", "-h", action="help",
                          help="show this help message and exit")

    return parser


def process_args(args=None):
    args = parseArguments().parse_args(args)

    return args


def getLagRegionWrapper(args):
    """
    Passes everything to getLagRegionWorker
    """
    return getLagRegionWorker(*args)


def getLagRegionWorker(chrom, start, end, bam=None, maxLag=1500, minMappingQuality=0, ignoreDuplicates=False, samFlag_include=0, samFlag_exclude=0, blackListFileName=None, verbose=False):
    """
    Determine the cross-correlation in a particular range, returning a tuple of (number of reads, [PCC0, PCC1, ..., PCCN]),
    where PCCX is the pearson correlation coefficient at a given lag.

    Note that some positions with > 10x mean coverage of covered bases are ignored.
    """
    # Ignore short regions
    if end - start < maxLag:
        return None

    if verbose:
        sys.stderr.write("[getLagRegionWorker] Processing {}:{}-{}\n".format(chrom, start, end))

    n = 0
    fSignal = np.zeros(end - start)
    rSignal = np.zeros(end - start)
    f = bamHandler.openBam(bam)

    # Only the reverse strand
    prev_start_pos = None  # to store the start positions
    for read in f.fetch(chrom, start, end):
        # Filter
        if read.is_unmapped:
            continue
        if minMappingQuality and read.mapq < minMappingQuality:
            continue
        if samFlag_include and read.flag & samFlag_include != samFlag_include:
            continue
        if samFlag_exclude and read.flag & samFlag_exclude != 0:
            continue
        if ignoreDuplicates and prev_start_pos \
                and prev_start_pos == (read.reference_start, read.pnext, read.is_reverse):
            continue
        prev_start_pos = (read.reference_start, read.pnext, read.is_reverse)

        if read.is_reverse:
            if read.reference_end >= end + maxLag:
                continue
            rSignal[read.reference_end - start] += 1
            # This handles edge-effects without inflating the total read count or the weighting
            if read.reference_end <= end:
                n += 1
        else:
            if read.pos < start:
                continue
            fSignal[read.pos - start] += 1
            if read.pos < end:
                n += 1
    f.close()

    # Handle blacklist regions
    if blackListFileName:
        bl = GTF(blackListFileName)
        for reg in bl.findOverlaps(chrom, start, end):
            if reg[0] >= start:
                s = reg[0] - start
            else:
                s = 0
            if reg[1] <= end:
                e = reg[1] - start
            else:
                e = end - start
            fSignal[s:e] = 0
            rSignal[s:e] = 0
        del(bl)

    # Get valid positions on each strand
    px = fSignal.nonzero()[0]
    py = rSignal.nonzero()[0]
    # If either strand has less than two sites then return None
    if px.shape[0] < 1 or py.shape[0] < 1:
        return None

    # Perform the equivalent of remove.tag.anomalies, which I find to be a bad idea...
    stt = np.sort(np.concatenate([fSignal[px], rSignal[py]]))
    stt = stt[0:int(stt.shape[0] * (0.999))]  # Ignore the top 0.1%
    mtc = np.mean(stt)
    tcd = np.sqrt(np.var(stt) + 0.1)
    thr = max(1, np.ceil(mtc + 5 * tcd))
    thrO = max(1, np.ceil(mtc + 15 * tcd))
    # print("thr {} thrO {} {}/{}>thr, {}/{}>thrO".format(thr, thrO, np.where(fSignal > thr)[0].shape[0], np.where(rSignal > thr)[0].shape[0], np.where(fSignal > thrO)[0].shape[0], np.where(rSignal > thrO)[0].shape[0]))
    # print("{} / {}".format(np.where(fSignal > thr), np.where(rSignal > thr)))
    fSignal2 = np.copy(fSignal)
    rSignal2 = np.copy(rSignal)
    fSignal2[np.where(fSignal2 <= thr)] = 0  # tp
    rSignal2[np.where(rSignal2 <= thr)] = 0
    px2 = fSignal2.nonzero()[0]
    py2 = rSignal2.nonzero()[0]
    foo = np.intersect1d(px2, py2, assume_unique=True)
    fSignal2 = np.copy(fSignal)
    rSignal2 = np.copy(rSignal)
    fSignal2[np.where(fSignal2 < thrO)] = 0
    rSignal2[np.where(rSignal2 < thrO)] = 0
    px2 = fSignal2.nonzero()[0]
    py2 = rSignal2.nonzero()[0]
    # print("len(foo) {} len(px2) {} len(py2) {}".format(len(foo), len(px2), len(py2)))
    it = np.union1d(foo, np.union1d(px2, py2))
    fSignal[it] = 0
    rSignal[it] = 0

    # Filter out sites with coverage > 10x average, comment the next 7 lines out to prevent that
    totalMean = (np.sum(fSignal) + np.sum(rSignal)) / float(len(px) + len(py))
    fSignal[np.where(fSignal >= 10.0 * totalMean)] = 0
    rSignal[np.where(rSignal >= 10.0 * totalMean)] = 0
    px = fSignal.nonzero()[0]
    py = rSignal.nonzero()[0]
    if px.shape[0] < 1 or py.shape[0] < 1:
        return None

    # Number of possibly covered bases
    nCovered = max(px[-1], py[-1]) - min(px[0], py[0]) + 1
    # Mean and error or covered positions
    meanX = np.sum(fSignal) / float(nCovered)
    meanY = np.sum(rSignal) / float(nCovered)
    # print([px[-1], py[-1], px[0], py[0]])
    # print("sum(f) {} sum(r) {} meanX {} meanY {} nCovered {}".format(np.sum(fSignal), np.sum(rSignal), meanX, meanY, nCovered))
    eX = np.zeros(len(fSignal))
    eY = np.zeros(len(fSignal))
    eX[px] = fSignal[px] - meanX
    eY[py] = rSignal[py] - meanY
    # The denominator of the correlation coefficient
    denom = np.sqrt((np.dot(eX, eX) + (nCovered - len(px)) * meanX**2) *
                    (np.dot(eY, eY) + (nCovered - len(py)) * meanY**2))
    # These need to be summed to compute the correlation coefficients, since directly doing a weighted average is incorrect
    t = [nCovered, fSignal, rSignal, px, py]
    # print("np.dot(eX, eX) {}".format(np.dot(eX, eX)))
    # print("A {}".format(nCovered - len(px)))
    # print("B {}".format(meanX**2))
    # print("np.dot(eY, eY) {}".format(np.dot(eY, eY)))
    # print("C {}".format(nCovered - len(py)))
    # print("D {}".format(meanY**2))
    # print("denom {}".format(denom))
    r = []
    # py = set(py)
    for lag in range(maxLag + 1):
        # match up valid values in x and y, this is a bottleneck
        validY = np.intersect1d(px + lag, py, assume_unique=True)
        # validY = np.array(list(set(px + lag).intersection(py)))
        if len(validY) == 0:
            r.append(0)
            t.append(None)
            continue
        validX = validY - lag
        eX2 = eX[validX]
        eY2 = eY[validY]
        # The numerator, which is just an offset dot product
        numer = np.dot(eX2, eY2) - \
            meanY * (np.sum(eX) - np.sum(eX2)) - \
            meanX * (np.sum(eY) - np.sum(eY2)) + \
            meanX * meanY * (nCovered - len(px) - len(py) + len(validX))
        # print("lag {} np.dot(eX2, eY2) {} np.sum(eX) {} np.sum(eX2) {} np.sum(eY) {} np.sum(eY2) {}".format(lag,  np.dot(eX2, eY2), np.sum(eX), np.sum(eX2), np.sum(eY), np.sum(eY2)))
        # print("len(px) {} len(py) {} len(validX) {}".format(len(px), len(py), len(validX)))
        # print("numer {}".format(numer))

        t.append(validX)
        r.append(numer / denom)

    return n, np.array(r)
    # return n, t


def getLagMatrix(bam, maxLag, region=None, blackListFileName=None, minMappingQuality=0, ignoreDuplicates=False, samFlag_include=0, samFlag_exclude=0, numberOfProcessors=4, verbose=False):
    """
    This is function runs the main map-reduce step on regions, calling getLagMatrixWrapper.

    The cross-correlation of each region is determine and, for each, a tuple of the number
    of alignments and a list of correlation coefficients at each lag is returned. The
    weighted average is then taken per-lag (this is the same as SPP).

    Note that the 5' most region of each alignment is used.

    A list of correlation coefficients at each lag is returned.
    """
    f = bamHandler.openBam(bam)
    chromSizes = [(f.references[i], f.lengths[i]) for i in range(len(f.references))]
    f.close()
    maxLen = 0
    for _ in chromSizes:
        if _[1] > maxLen:
            maxLen = _[1]

    foo = mapReduce([bam, maxLag, minMappingQuality, ignoreDuplicates, samFlag_include, samFlag_exclude, blackListFileName, verbose],
                    getLagRegionWrapper,
                    chromSizes,
                    genomeChunkLength=maxLen,
                    region=region,
                    numberOfProcessors=numberOfProcessors,
                    verbose=verbose)

    nTotal = 0
    r = np.zeros(maxLag + 1, dtype="float64")
    # get the total number of alignments used.
    for res in foo:
        if res is not None:
            nTotal += res[0]
    print("nTotal {}".format(nTotal))

    # Go through the lags
    """
    for l in range(maxLag + 1):
        print(l)
        nCovered = 0
        fSignal = np.empty(0)
        rSignal = np.empty(0)
        px = np.empty(0)
        py = np.empty(0)
        validX = np.empty(0)
        for res in foo:
            if res is None:
                continue
            res2 = res[1]  # for convenience
            if any(v is None for v in res2):
                continue
            print(len(res2))
            nCovered += res2[0]  # nCovered
            fSignal = np.concatenate([fSignal, res2[1]])
            rSignal = np.concatenate([rSignal, res2[2]])
            px = np.concatenate([px, len(px) + res2[3]])
            py = np.concatenate([py, len(py) + res2[4]])
            validX = np.concatenate([validX, len(validX) + res2[5 + l]])
        if px.shape[0] == 0 or py.shape[0] == 0 or validX.shape[0] == 0 or nCovered == 0:
            continue

        print("fSignal {}".format(fSignal))
        print(nCovered)
        meanX = np.sum(fSignal) / float(nCovered)
        meanY = np.sum(rSignal) / float(nCovered)
        eX[px] = fSignal[px] - meanX
        eY[py] = rSignal[py] - meanY
        eX2 = eX[validX]
        eY2 = eY[validX + l]

        denom = np.sqrt((np.dot(eX, eX) + (nCovered - len(px)) * meanX**2) *
                        (np.dot(eY, eY) + (nCovered - len(py)) * meanY**2))

        numer = np.dot(eX2, eY2) - \
                meanY * (np.sum(eX) - np.sum(eX2)) - \
                meanX * (np.sum(eY) - np.sum(eY2)) + \
                meanX * meanY * (nCovered - len(px) - len(py) + len(validX))

        r[l] = numer / denom
    """

    # Go through the lags
    for res in foo:
        if res is not None:
            coef = res[0] / float(nTotal)
            r += res[1] * coef

    return nTotal, r


def main(args=None):
    args = process_args(args)

    if args.minSeparation < 1:
        sys.exit("--minSeparation must be at least 1!\n")

    # Get the expected read length
    frag_len_dict, read_len_dict = get_read_and_fragment_length(args.bam,
                                                                return_lengths=False,
                                                                blackListFileName=args.blackListFileName,
                                                                numberOfProcessors=args.numberOfProcessors,
                                                                verbose=args.verbose)

    if args.maxLag - args.minSeparation - read_len_dict['median'] - 1 < 1:
        sys.exit("--maxLag must be more than the median read length ({}) plus --minSeparation.\n".format(read_len_dict['median']))

    if frag_len_dict is not None:
        sys.stderr.write("Warning: A paired-end dataset is being used. It is advisable to use only read #1 or #2, but not both!\n")

    if args.maxLag < read_len_dict['median']:
        sys.exit("Error: the maximim lag ({}) is less than the median read length ({}). This makes absolute no sense.".format(args.maxLag, read_len_dict['median']))

    numReads, cors = getLagMatrix(args.bam,
                                  args.maxLag,
                                  region=args.region,
                                  blackListFileName=args.blackListFileName,
                                  numberOfProcessors=args.numberOfProcessors,
                                  minMappingQuality=args.minMappingQuality,
                                  ignoreDuplicates=args.ignoreDuplicates,
                                  samFlag_include=args.samFlagInclude,
                                  samFlag_exclude=args.samFlagExclude,
                                  verbose=args.verbose)

    if args.outFileNameData is not None:
        of = open(args.outFileNameData, "w")
        of.write("Lag\tCorrelation\n")
        for lag, r in zip(range(args.maxLag + 1), cors):
            of.write("{}\t{}\n".format(lag, r))
        of.close()

    x = np.arange(args.maxLag + 1)

    # Get the values needed for NSC and RSC
    minCor = np.min(cors)
    minCorX = np.argmin(cors)
    smoothed = np.convolve(cors, np.ones((3,)) / 3, mode='valid')  # The first/last 2 bases are trimmed by this
    readCor = np.max(smoothed[:int(read_len_dict['median']) - 1 + args.minSeparation])
    readCorX = np.argmax(smoothed[:int(read_len_dict['median']) - 1 + args.minSeparation])
    fragCor = np.max(smoothed[int(read_len_dict['median']) - 1 + args.minSeparation:])
    fragCorX = np.argmax(smoothed[int(read_len_dict['median']) - 1 + args.minSeparation:]) + int(read_len_dict['median']) - 1 + args.minSeparation
    NSC = fragCor / float(minCor)
    RSC = (fragCor - minCor) / float(readCor - minCor)
    if args.outFileQualityMetrics is not None:
        if RSC < 0:
            QualityTag = "NA"
        elif RSC < 0.25:
            QualityTag = -2
        elif RSC < 0.5:
            QualityTag = -1
        elif RSC < 1:
            QualityTag = 0
        elif RSC < 1.5:
            QualityTag = 1
        else:
            QualityTag = 2

        if args.label is not None:
            label = args.label
        else:
            label = args.bam

        of = open(args.outFileQualityMetrics, "w")
        of.write("FileName\tnumReads\testFragLen\tcorr_estFragLen\tphantomPeak\tcorr_phantomPeak\targmin_corr\tmin_corr\tNSC\tRSC\tQualityTag\n")
        of.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(label,
                                                                       numReads,
                                                                       fragCorX,
                                                                       fragCor,
                                                                       readCorX,
                                                                       readCor,
                                                                       minCorX,
                                                                       minCor,
                                                                       NSC,
                                                                       RSC,
                                                                       QualityTag))
        of.close()

    fig = plt.figure(figsize=(11, 9.5))
    plt.plot(x, cors)
    plt.axvline(readCorX, color='r', linestyle='dotted')  # red line @ median read size
    rline = plt.axhline(readCor, color='r', linestyle='dotted')  # red line @ median read size
    plt.axvline(fragCorX, color='blue', linestyle='dotted')  # blue line at peak fragment size
    fline = plt.axhline(fragCor, color='blue', linestyle='dotted')  # blue line at peak fragment size
    proxy = mpatches.Patch(color='white')
    plt.legend([rline, fline, proxy, proxy], ["phantom peak", "background peak", 'NSC: {}'.format(round(NSC, 3)), 'RSC: {}'.format(round(RSC, 3))])
    plt.suptitle(args.plotTitle)
    plt.ylabel('Cross-correlation')
    plt.xlabel('Strand Lag')
    fig.savefig(args.plotFile, format=args.plotFileFormat)
    plt.close()
