import numpy as np
import encoder
import sys, os
from pathlib import Path
from PIL import Image
import cv2
import pandas as pd
import time as t
import shutil
import utils
import matplotlib.pyplot as plt
from encoder import DCT, padding
from scipy.fftpack import dct
import random as rd

LIM = 2  # number of files to test to


def compare(
    qualities=None,
    dataDirectory=None,
    outputDirectory=None,
    subsamples=None,
    useStdHuffmanTable=None,
    DeleteFilesAfterward=True,
):
    if qualities is None:
        qualities = [np.random.randint(0, 101)]
    if subsamples is None:
        subsamples = ["4:2:2"]
    if dataDirectory is None:
        dataDirectory = "./data/datasetBmp"
    if useStdHuffmanTable is None:
        useStdHuffmanTable = [False]
    stat = np.zeros(
        (LIM * len(qualities) * len(subsamples) * len(useStdHuffmanTable), 6),
        dtype=object,
    )  # dim 0 : quality factor, dim 1 : subsample method, dim 2 : usage of std Hf Tables, dim 3 : size before compression, dim 4 : size after compression, dim 5 : time to compress
    i = 0
    i_max = LIM * len(qualities) * len(subsamples) * len(useStdHuffmanTable)
    filesTreated = rd.choices(os.listdir(dataDirectory), k=LIM)
    for quality in qualities:
        for subsample in subsamples:
            for hfTables in useStdHuffmanTable:
                outputDirectory = f"./data/treated/quality{quality}-subsample{subsample}-stdHf{hfTables}"
                for filename in filesTreated:
                    f = os.path.join(dataDirectory, filename)
                    if not os.path.exists(outputDirectory):
                        os.makedirs(outputDirectory)
                    f_out = os.path.join(outputDirectory, filename + ".jpg")
                    if os.path.isfile(f):
                        previousSize = os.stat(f).st_size
                        image = Image.open(f)
                        time = t.time()
                        encoder.write_jpeg(
                            f_out, np.array(image), quality, subsample, hfTables
                        )
                        time = t.time() - time
                        newSize = os.stat(f_out).st_size
                        stat[i][0] = quality
                        stat[i][1] = subsample
                        stat[i][2] = hfTables
                        stat[i][3] = previousSize
                        stat[i][4] = newSize
                        stat[i][5] = time
                    i += 1
                    print(f"{i}/{i_max}", end="\r")
                if DeleteFilesAfterward:
                    shutil.rmtree(outputDirectory)
    return stat


def write_stat(statFile, stat, quality, subsample, standHuffTables):
    with open(statFile, "a+") as f:
        f.write("\n" * 2)
        f.write("New sample \n")
        f.write(f"Size of sample : {LIM} images \n")
        f.write(
            f"Parameters of compression : (quality) {quality}, (subsample) {subsample}, (usage of standard HuffTables) {'Yes' if standHuffTables else 'No'} \n"
        )
        avgPreviousSize = np.average(stat[:, 0])
        avgNewSize = np.average(stat[:, 1])
        f.write(
            f"Average size of image before compression : {avgPreviousSize} bytes \n"
        )
        f.write(f"Average size of images after compression : {avgNewSize} bytes \n")
        f.write(f"Ratio is {avgPreviousSize / avgNewSize:.2f}")


def write_stat_csv(output, stat):
    if os.path.isfile(output):
        pd.DataFrame(stat).to_csv(output, mode="a", index=False, header=False)
    else:
        pd.DataFrame(stat).to_csv(
            output,
            index=False,
            header=[
                "quality",
                "subsample",
                "stdHuffmanTables",
                "oldSize",
                "newSize",
                "time",
            ],
        )


def csv_to_stat(csvFile):
    stat = pd.read_csv(csvFile)
    return stat


def dataInterpreation(dataFrame):
    df = dataFrame
    qualities = df["quality"].unique()
    qualitySize = {}
    qualityTime = {}
    for quality in qualities:
        qualitySize[quality] = int(df[df["quality"] == quality]["newSize"].mean())
        qualityTime[quality] = round(df[df["quality"] == quality]["time"].mean(), 3)
    stdSize = int(df[df["stdHuffmanTables"] == True]["newSize"].mean())
    stdTime = round(df[df["stdHuffmanTables"] == True]["time"].mean(), 3)
    nonStdSize = int(df[df["stdHuffmanTables"] == False]["newSize"].mean())
    nonStdTime = round(df[df["stdHuffmanTables"] == False]["time"].mean(), 3)

    plt.rcParams["figure.figsize"] = [10, 5]

    fig, (ax1, ax3) = plt.subplots(1, 2)
    ax2 = ax1.twinx()

    fig.suptitle("Comparaison des compressions en fonction du facteur de qualité")

    width = 0.25

    initialSize = 786486
    xaxis = list(qualitySize.keys())
    yaxisSize = np.array(list(qualitySize.values()))
    yaxisTime = np.array(list(qualityTime.values()))
    yaxisRatio = (initialSize - yaxisSize) / yaxisTime

    color1 = "tab:red"
    color2 = "tab:blue"
    color3 = "tab:green"

    ax1.bar(
        np.arange(len(qualitySize)) - width,
        yaxisSize,
        width,
        tick_label=xaxis,
        color=color1,
        label="Taille après compression",
    )
    ax2.bar(
        np.arange(len(qualityTime)),
        yaxisTime,
        width,
        tick_label=xaxis,
        color=color2,
        label="Temps de compression",
    )
    ax3.bar(
        np.arange(len(qualitySize)),
        yaxisRatio,
        width,
        tick_label=xaxis,
        color=color3,
        label="Octets gagnés par seconde",
    )

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left", bbox_to_anchor=(0, 0.9))
    ax3.legend(loc="upper right")

    ax3.yaxis.tick_right()

    ax1.set_xlabel("Facteur de qualité")
    ax1.set_ylabel("Taille (en octets)", color=color1)
    ax2.set_ylabel("Temps (en secondes)", color=color2)
    ax3.set_xlabel("Facteur de qualité")
    ax3.set_ylabel("Taille gagné par unité de temps (octets.Hz)", color=color3)
    ax3.yaxis.set_label_position("right")

    plt.savefig("./data/treated/compressionComparaison", transparent=True)

    plt.rcParams["figure.figsize"] = [7, 5]
    plt.clf()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    fig.suptitle(
        "Comparaison des compressions en fonction des tables de Huffman utilisées"
    )

    yaxisSize = [stdSize, nonStdSize]
    yaxisTime = [stdTime, nonStdTime]

    labels = ["Tables Standards", "Tables Optimales"]
    ax1.bar(
        np.arange(2) - width / 2,
        yaxisSize,
        width,
        tick_label=labels,
        color=color1,
        label="Taille après compression",
    )
    ax2.bar(
        np.arange(2) + width / 2,
        yaxisTime,
        width,
        tick_label=labels,
        color=color2,
        label="Temps de compression",
    )

    ax1.legend(loc="upper center", bbox_to_anchor=(0.45, 1))
    ax2.legend(loc="upper center", bbox_to_anchor=(0.45, 0.9))

    ax1.set_ylabel("Taille (en octets)", color=color1)
    ax2.set_ylabel("Temps (en secondes)", color=color2)

    plt.savefig("./data/treated/compressionComparaison2", transparent=True)


def energyCompaction(imgPath):
    img = cv2.imread(imgPath)

    imgYCrCB = cv2.cvtColor(
        img, cv2.COLOR_RGB2YCrCb
    )  # Convert RGB to YCrCb (Cb applies V, and Cr applies U).

    Y, Cr, Cb = cv2.split(padding(imgYCrCB, 8, 8))
    Y = Y.astype("int") - 128
    blocks_Y = utils.divide_blocks(Y, 8, 8)
    dctBlocks_Y = np.zeros_like(blocks_Y)
    for i in range(len(blocks_Y)):
        dctBlocks_Y[i] = dct(
            dct(blocks_Y[i], axis=0, norm="ortho"), axis=1, norm="ortho"
        )
    avg_Y = utils.averageMatrix(blocks_Y)
    avgDct_Y = utils.averageMatrix(dctBlocks_Y)

    x = np.random.randint(blocks_Y.shape[0])
    arr1 = blocks_Y[x]
    arr2 = dctBlocks_Y[x]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    valueMax, valueMin = max(np.max(arr1), np.max(arr2)), min(
        np.min(arr1), np.min(arr2)
    )
    # fig.suptitle('Matrice de la luminance de "villeLyon.jpg"')

    ax1.matshow(arr1, cmap="cool", vmin=valueMin, vmax=valueMax)
    ax1.set_title("avant DCT")

    ax2.matshow(arr2, cmap="cool", vmin=valueMin, vmax=valueMax)
    ax2.set_title("après DCT")

    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            cNormal = int(arr1[i, j])
            cDct = int(arr2[i, j])
            ax1.text(i, j, str(cNormal), va="center", ha="center")
            ax2.text(i, j, str(cDct), va="center", ha="center")
    plt.savefig("./data/energyCompaction.png", transparent=True)


def rgbToYCbCr_channel_bis():
    img = cv2.imread("./data/villeLyon.jpg")  # Read input image in BGR format

    imgYCrCB = cv2.cvtColor(
        img, cv2.COLOR_BGR2YCrCb
    )  # Convert RGB to YCrCb (Cb applies V, and Cr applies U).

    Y, Cr, Cb = cv2.split(imgYCrCB)

    # Fill Y and Cb with 128 (Y level is middle gray, and Cb is "neutralized").
    onlyCr = imgYCrCB.copy()
    onlyCr[:, :, 0] = 128
    onlyCr[:, :, 2] = 128
    onlyCr_as_bgr = cv2.cvtColor(
        onlyCr, cv2.COLOR_YCrCb2BGR
    )  # Convert to BGR - used for display as false color

    # Fill Y and Cr with 128 (Y level is middle gray, and Cr is "neutralized").
    onlyCb = imgYCrCB.copy()
    onlyCb[:, :, 0] = 128
    onlyCb[:, :, 1] = 128
    onlyCb_as_bgr = cv2.cvtColor(
        onlyCb, cv2.COLOR_YCrCb2BGR
    )  # Convert to BGR - used for display as false color

    cv2.imshow("img", img)
    cv2.imshow("Y", Y)
    cv2.imshow("onlyCb_as_bgr", onlyCb_as_bgr)
    cv2.imshow("onlyCr_as_bgr", onlyCr_as_bgr)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite("./data/treated/villeLyon_Y.jpg", Y)
    cv2.imwrite("./data/treated/villeLyon_Cb.jpg", onlyCb_as_bgr)
    cv2.imwrite("./data/treated/villeLyon_Cr.jpg", onlyCr_as_bgr)


if __name__ == "__main__":
    # compare()
    # rgbToYCbCr_channel_bis()
    # energyCompaction("./data/villeLyon.jpg")
    # test = np.array([[93, 90, 83, 68, 61, 61, 46, 21],
    #                 [102, 92, 95, 77, 65, 60, 49, 32],
    #                 [69, 55, 47, 57, 65, 60, 72, 65],
    #                 [55, 55, 40, 42, 23, 1, 11, 38],
    #                 [55, 57, 47, 53, 35, 59, -2, 26],
    #                 [64, 41, 42, 55, 60, 57, 25, -8],
    #                 [77, 87, 58, -2, -5, 14, -10, -35],
    #                 [38, 14, 33, 33, -21, -23, -43, -34]])
    # print(dct(dct(test, axis=0, norm="ortho"), axis=1, norm='ortho'))

    # stat = compare(qualities = list(range(1, 101, 10)), subsamples=['4:4:4', '4:2:0', '4:1:1', '4:2:2'], useStdHuffmanTable=[True, False], DeleteFilesAfterward=True)
    # write_stat_csv("./data/treated/stat.csv", stat)
    stat = csv_to_stat("./data/treated/stat.csv")
    dataInterpreation(stat)
