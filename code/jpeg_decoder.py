import numpy as np
from collections import deque, namedtuple
from itertools import product
from math import ceil, cos, pi
from scipy.interpolate import griddata
from typing import Callable, Tuple, Union
from pathlib import Path

# JPEG markers (for our supported segments)
SOI = bytes.fromhex("FFD8")  # Start of image
SOF0 = bytes.fromhex("FFC0")  # Start of frame (Baseline DCT)
SOF2 = bytes.fromhex("FFC2")  # Start of frame (Progressive DCT)
DHT = bytes.fromhex("FFC4")  # Define Huffman table
DQT = bytes.fromhex("FFDB")  # Define quantization table
DRI = bytes.fromhex("FFDD")  # Define restart interval
SOS = bytes.fromhex("FFDA")  # Start of scan
DNL = bytes.fromhex("FFDC")  # Define number of lines
EOI = bytes.fromhex("FFD9")  # End of image

# Restart markers
RST = tuple(bytes.fromhex(hex(marker)[2:]) for marker in range(0xFFD0, 0xFFD8))

# Containers for the parameters of each color component
ColorComponent = namedtuple(
    "ColorComponent",
    "name order verticalSampling horizontalSampling dqtId repeat shape",
)
HuffmanTable = namedtuple("HuffmanTable", "dc ac")


class JpegDecoder:
    def __init__(self, file: Path):
        with open(file, "rb") as image:
            self.__file = image.read()
        self.fileSize = len(self.__file)  # Size in bytes of the file
        self.filePath = file if isinstance(file, Path) else Path(file)

        if not self.__file.startswith(SOI + b"\xFF"):
            raise NotJpeg("Error- The given file is not a JPEG ")
        print(f"Reading file '{self.filePath.name}' ({self.fileSize:,} bytes)")

        self.handlers = {
            DHT: self.defineHuffmanTable,
            DQT: self.defineDqt,
            DRI: self.defineRestartInterval,
            SOF0: self.startFrame,
            SOF2: self.startFrame,
            SOS: self.startScan,
            EOI: self.endImage,
        }

        # Initialize decoding parameters
        self.fileHeader = 2  # Offset (in bytes, 0-index) from the beginning of the file
        self.scanFinished = False  # If the 'end of image' marker has been reached
        self.scanMode = None  # Supported modes: 'baselineDct' or 'progressiveDct'
        self.imageWidth = 0  # Width in pixels of the image
        self.imageHeight = 0  # Height in pixels of the image
        self.colorComponents = (
            {}
        )  # Hold each color component and its respective parameters
        self.sampleShape = ()  # Size to upsample the subsampled color components
        self.huffmanTables = {}  # Hold all huffman tables
        self.quantizationTables = {}  # Hold all quantization tables
        self.restartInterval = 0  # How many MCUs before each restart marker
        self.imageArray = None  # Store the color values for each pixel of the image
        self.scanCount = 0  # Counter for the performed scans

        # Main loop to find and process the supported file segments
        while not self.scanFinished:
            try:
                current_byte = self.__file[self.fileHeader]
            except IndexError:
                del self.__file
                break

            # Whether the current byte is 0xFF
            if current_byte == 0xFF:
                # Read the next byte
                segMarker = self.__file[self.fileHeader : self.fileHeader + 2]
                self.fileHeader += 2

                # Whether the two bytes form a marker (and isn't a restart marker)
                if (segMarker != b"\xFF\x00") and (segMarker not in RST):
                    # Attempt to get the handler for the marker
                    segHandler = self.handlers.get(segMarker)
                    if self.fileHeader < self.fileSize - 1:
                        segSize = (
                            (self.__file[self.fileHeader] << 8)
                            + self.__file[self.fileHeader + 1]
                        ) - 2
                        self.fileHeader += 2

                    if segHandler is not None:
                        # If a handler was found, pass the control to it
                        myData = self.__file[
                            self.fileHeader : self.fileHeader + segSize
                        ]
                        segHandler(myData)
                    else:
                        # Otherwise, just skip the data segment
                        self.fileHeader += segSize

            else:
                # Move to the next byte if the current byte is not 0xFF
                self.fileHeader += 1

    def startFrame(self, data: bytes) -> None:
        dataSize = len(data)
        dataHeader = 0

        # Check encoding mode
        mode = self.__file[self.fileHeader - 4 : self.fileHeader - 2]
        if mode == SOF0:
            self.scanMode = "baselineDct"
            print("Scan mode : Sequential")
        elif mode == SOF2:
            self.scanMode = "progressiveDct"
            print("Scan mode : Progressive")
        else:
            raise UnsupportedJpeg("Error - Encoding mode not supported")

        # Check sample precision
        precision = data[dataHeader]
        if precision != 8:
            raise UnsupportedJpeg("Error - Unsupported color depth")
        dataHeader += 1

        # Get image dimensions
        self.imageHeight = (data[dataHeader] << 8) + data[
            dataHeader + 1
        ]  # height could be 0 here, but will be retrievable on DNL segment
        dataHeader += 2

        self.imageWidth = (data[dataHeader] << 8) + data[dataHeader + 1]
        dataHeader += 2

        print(f"Image dimensions : {self.imageWidth} x {self.imageHeight}")

        if self.imageWidth == 0:
            raise CorruptedJpeg("Error - Image width cannot be zero")

        # Check number of color component
        componentsNumber = data[dataHeader]
        if componentsNumber not in (1, 3):
            if componentsNumber == 4:
                raise UnsupportedJpeg("Error -  CMYK color space is not supported")
            else:
                raise UnsupportedJpeg(
                    "Error - Unsupported color space (only RDG and Grayscale are)"
                )
        dataHeader += 1

        if componentsNumber == 3:
            print("Color space : RGB")
        elif componentsNumber == 1:
            print("Color space : Grayscale")

        # Get the color components and their parameters
        components = (
            "Y",  # Luminance
            "Cb",  # Blue chrominance
            "Cr",  # Red chrominance
        )

        try:
            for count, component in enumerate(components, start=1):
                # Get ID of color component
                myId = data[dataHeader]
                dataHeader += 1

                sample = data[dataHeader]
                horizontalSample = sample >> 4
                verticalSample = sample & 0x0F

                dataHeader += 1

                # Get quantization table for the component
                myDqt = data[dataHeader]
                dataHeader += 1

                myComponent = ColorComponent(
                    name=component,  # Name of the color component
                    order=count
                    - 1,  # Order in which the component will come in the image
                    horizontalSampling=horizontalSample,  # Amount of pixels sampled in the horizontal
                    verticalSampling=verticalSample,  # Amount of pixels sampled in the vertical
                    dqtId=myDqt,  # Quantization table selector
                    repeat=horizontalSample
                    * verticalSample,  # Amount of times the component repeats during decoding
                    shape=(
                        8 * horizontalSample,
                        8 * verticalSample,
                    ),  # Dimensions (in pixels) of the MCU for the component
                )

                # Add the component parameters to the dictionary
                self.colorComponents.update({myId: myComponent})

                # Have we parsed all components?
                if count == componentsNumber:
                    break

        except IndexError:
            raise CorruptedJpeg("Error - Failed to parse the start of frame.")

        # Shape of the sampling area
        # (these values will be used to upsample the subsampled color components)
        sampleWidth = max(
            component.shape[0] for component in self.colorComponents.values()
        )
        sampleHeight = max(
            component.shape[1] for component in self.colorComponents.values()
        )
        self.sampleShape = (sampleWidth, sampleHeight)

        # Display the samplings
        print(
            f"Horizontal sampling: {' x '.join(str(component.horizontalSampling) for component in self.colorComponents.values())}"
        )
        print(
            f"Vertical sampling  : {' x '.join(str(component.verticalSampling) for component in self.colorComponents.values())}"
        )

        # Move the file header to the end of the data segment
        self.fileHeader += dataSize

    def defineHuffmanTable(self, data: bytes) -> None:
        """Parse the Huffman tables from the file."""
        dataSize = len(data)
        dataHeader = 0

        while dataHeader < dataSize:
            tableDestination = data[dataHeader]
            dataHeader += 1

            # Count how many codes of each length there are

            codes_count = {
                bitLength: count
                for bitLength, count in zip(
                    range(1, 17), data[dataHeader : dataHeader + 16]
                )
            }
            dataHeader += 16

            # Get the Huffman values (HUFFVAL)

            huffval_dict = (
                {}
            )  # Dictionary that associates each code bit-length to all its respective Huffman values

            for bitLength, count in codes_count.items():
                huffval_dict.update({bitLength: data[dataHeader : dataHeader + count]})
                dataHeader += count

            # Error checking
            if dataHeader > dataSize:
                # If we tried to read more bytes than what the data has, then something is wrong with the file
                raise CorruptedJpeg("Error - Failed to parse Huffman tables.")

            # Build the Huffman tree

            huffmanTree = {}

            code = 0
            for bitLength, values_list in huffval_dict.items():
                code <<= 1
                for huffval in values_list:
                    codeString = bin(code)[2:].rjust(bitLength, "0")
                    huffmanTree.update({codeString: huffval})
                    code += 1

            # Add tree to the Huffman table dictionary
            self.huffmanTables.update({tableDestination: huffmanTree})
            print(f"Parsed Huffman table - ", end="")
            print(
                f"ID: {tableDestination & 0x0F} ({'DC' if tableDestination >> 4 == 0 else 'AC'})"
            )

        # Move the file header to the end of the data segment
        self.fileHeader += dataSize

    def defineDqt(self, data: bytes) -> None:
        dataSize = len(data)
        dataHeader = 0

        while dataHeader < dataSize:
            tableDestination = data[dataHeader] & 0x0F
            tableFormat = (
                data[dataHeader] >> 4
            )  # Check wether the QT is stored using 8bit or 16bit values
            dataHeader += 1

            if tableDestination > 3:
                raise UnsupportedJpeg("Error - Only 3 quantization tables allowed")

            if tableFormat == 0:
                qtValues = np.array(
                    [value for value in data[dataHeader : dataHeader + 64]],
                    dtype="int16",
                )
                dataHeader += 64
            elif tableFormat != 0:
                qtValues = np.empty([64, 1], dtype="int16")
                for i in range(64):
                    qtValues[i] = (data[dataHeader] << 8) + data[dataHeader + 1]
                    dataHeader += 2

            try:
                quantizationTable = undoZigzag(qtValues)
            except ValueError:
                raise CorruptedJpeg("Error - Failed to parse quantization tables")

            self.quantizationTables.update({tableDestination: quantizationTable})
            print(f"Parsed quantization table - ID: {tableDestination}")

            self.fileHeader += dataSize

    def defineRestartInterval(self, data: bytes) -> None:
        """Parse the restart interval value."""
        self.restartInterval = (data[0] << 8) + data[1]
        self.fileHeader += 2
        print(f"Restart interval: {self.restartInterval}")

        """NOTE
            The JPEG standart allow to restart markers to be added to the encoded image data.
            Those are meant to aid in error correction. The restart markers, when present,
            are added each a certain amount of MCUs. This amount is specified in the
            "Define Restart Interval" (DRI) segment, which starts after the 0xFFDD marker.
            The restart markers are the bytes from 0xFFD0 to 0xFFD7. They are used sequentially,
            and wrap back to 0xFFD0 after 0xFFD7.
            
            It is worth noting that the MCUs encoded on the data stream are not necessarily
            aligned to the byte boundary (8-bits). So after reaching the amount of MCUs specified
            on the restart interval, it is necessary to move the bits header to the beginning of
            the next byte, by taking the modulo of the position,if the header isn't already there:
                
                if (headerPosition % 8) != 0:
                    headerPosition += 8 - (headerPosition % 8)
                headerPosition += 16
            
            We also need to jump the marker itself, which is 16-bits long. So we also added 16
            to the header position.
            It is worth noting that the restart interval can be defined again after a scan.
            The latest defined value is what counts for each scan.
            """

    def startScan(self, data: bytes) -> None:
        """Parse the information necessary to decode a segment of encoded image data,
        then passes this information to the method that handles the scan mode used."""

        dataSize = len(data)
        dataHeader = 0

        # Number of color components in the scan
        componentsAmount = data[dataHeader]
        dataHeader += 1

        # Get parameters of the components in the scan
        myHuffmanTables = {}
        myColorComponents = {}
        for component in range(componentsAmount):
            componentId = data[
                dataHeader
            ]  # Should match the component ID's on the 'start of frame'
            dataHeader += 1

            # Selector for the Huffman tables
            tables = data[dataHeader]
            dataHeader += 1
            dcTable = (
                tables >> 4
            )  # Should match the tables ID's on the 'detect huffman table'
            acTable = (tables & 0x0F) | 0x10

            # Store the parameters
            myHuffmanTables.update({componentId: HuffmanTable(dc=dcTable, ac=acTable)})
            myColorComponents.update({componentId: self.colorComponents[componentId]})

        # Get spectral selection and successive approximation
        if self.scanMode == "progressiveDct":
            spectralSelectionStart = data[
                dataHeader
            ]  # Index of the first values of the data unit
            spectralSelectionEnd = data[
                dataHeader + 1
            ]  # Index of the last values of the data unit
            bitPositionHigh = (
                data[dataHeader + 2] >> 4
            )  # The position of the last bit sent in the previous scan
            bitPositionLow = (
                data[dataHeader + 2] & 0x0F
            )  # The position of the bit sent in the current scan

            dataHeader += 3

        # Move the file header to the beginning of the entropy encoded segment
        self.fileHeader += dataSize

        # Define number of lines
        if self.imageHeight == 0:
            dnlIndex = self.__file[self.fileHeader :].find(DNL)
            if dnlIndex != -1:
                dnlIndex += self.fileHeader
                self.imageHeight = (self.__file[dnlIndex + 4] << 8) + self.__file[
                    dnlIndex + 6
                ]
            else:
                raise CorruptedJpeg("Error - Image height cannot be zero.")

        # Dimensions of the MCU (minimum coding unit)
        if componentsAmount > 1:
            self.mcuWidth: int = 8 * max(
                component.horizontalSampling
                for component in self.colorComponents.values()
            )
            self.mcuHeight: int = 8 * max(
                component.verticalSampling
                for component in self.colorComponents.values()
            )
            self.mcuShape = (self.mcuWidth, self.mcuHeight)
        else:
            self.mcuWidth: int = 8
            self.mcuHeight: int = 8
            self.mcuShape = (8, 8)

        # Amount of MCUs in the whole image (horizontal, vertical, and total)
        if componentsAmount > 1:
            self.mcuCount_h = (self.imageWidth // self.mcuWidth) + (
                0 if self.imageWidth % self.mcuWidth == 0 else 1
            )
            self.mcuCount_v = (self.imageHeight // self.mcuHeight) + (
                0 if self.imageHeight % self.mcuHeight == 0 else 1
            )
        else:
            component = myColorComponents[componentId]
            sampleRatio_h = self.sampleShape[0] / component.shape[0]
            sampleRatio_v = self.sampleShape[1] / component.shape[1]
            layerWidth = self.imageWidth / sampleRatio_h
            layerHeight = self.imageHeight / sampleRatio_v
            self.mcuCount_h = ceil(layerWidth / self.mcuWidth)
            self.mcuCount_v = ceil(layerHeight / self.mcuHeight)

        self.mcuCount = self.mcuCount_h * self.mcuCount_v

        # Create the image array (if one does not exist already)
        if self.imageArray is None:
            # 3-dimensional array to store the color values of each pixel on the image
            # array(x-coordinate, y-coordinate, RBG-color)
            count_h = (self.imageWidth // self.sampleShape[0]) + (
                0 if self.imageWidth % self.sampleShape[0] == 0 else 1
            )
            count_v = (self.imageHeight // self.sampleShape[1]) + (
                0 if self.imageHeight % self.sampleShape[1] == 0 else 1
            )
            self.arrayWidth = self.sampleShape[0] * count_h
            self.arrayHeight = self.sampleShape[1] * count_v
            self.arrayDepth = len(self.colorComponents)
            self.imageArray = np.zeros(
                shape=(self.arrayWidth, self.arrayHeight, self.arrayDepth),
                dtype="int16",
            )

        # Setup scan counter
        if self.scanCount == 0:
            self.scanAmount = self.__file[self.fileHeader :].count(SOS) + 1
            print(f"Number of scans: {self.scanAmount}")

        # Begin the scan of the entropy encoded segment
        if self.scanMode == "baselineDct":
            self.baselineDctScan(myHuffmanTables, myColorComponents)
        elif self.scanMode == "progressiveDct":
            self.progressiveDctScan(
                myHuffmanTables,
                myColorComponents,
                spectralSelectionStart,
                spectralSelectionEnd,
                bitPositionHigh,
                bitPositionLow,
            )
        else:
            raise UnsupportedJpeg(
                "Error - Encoding mode not supported. Only 'Baseline DCT' and 'Progressive DCT' are supported."
            )

    def bitsGenerator(self) -> Callable[[int, bool], str]:
        """Returns a function that fetches the bits values in order from the raw file."""
        bitQueue = deque()

        # This nested function "remembers" the contents of bitQueue between different calls
        def getBits(amount: int = 1, restart: bool = False) -> str:
            """Fetches a certain amount of bits from the raw file, and moves the file header
            when a new byte is reached.
            """
            nonlocal bitQueue

            # Should be set to 'True' when the restart interval is reached
            if restart:
                bitQueue.clear()  # Discard the remaining bits
                self.fileHeader += 2  # Jump over the restart marker

            # Fetch more bits if the queue has less than the requested amount
            while amount > len(bitQueue):
                next_byte = self.__file[self.fileHeader]
                self.fileHeader += 1

                if next_byte == 0xFF:
                    self.fileHeader += 1  # Jump over the stuffed byte

                bitQueue.extend(
                    np.unpackbits(
                        bytearray(
                            (next_byte,)
                        )  # Unpack the bits and add them to the end of the queue
                    )
                )

            # Return the bits sequence as a string
            return "".join(str(bitQueue.popleft()) for bit in range(amount))

        # Return the nested function
        return getBits

    def baselineDctScan(self, huffmanTablesId: dict, myColorComponents: dict) -> None:
        """Decode the image data from the entropy encoded segment.
        The file header should be at the beginning of said segment, and at
        the after the decoding the header will be moved to the end of the segment.
        """
        print(f"\nScan {self.scanCount+1} of {self.scanAmount}")
        print(
            f"Color components: {', '.join(component.name for component in myColorComponents.values())}"
        )
        print(f"MCU count: {self.mcuCount}")
        print(f"Decoding MCUs and performing IDCT...")

        # Function to read the bits from the file's bytes
        nextBits = self.bitsGenerator()

        # Function to decode the next Huffman value
        def nextHuffval() -> int:
            codeword = ""
            huffmanValue = None

            while huffmanValue is None:
                codeword += nextBits()
                if len(codeword) > 16:
                    raise CorruptedJpeg(
                        f"Failed to decode image ({currentMcu}/{self.mcuCount} MCUs decoded)."
                    )
                huffmanValue = huffmanTable.get(codeword)

            return huffmanValue

        # Function to perform the inverse discrete cosine transform (IDCT)
        idct = InverseDCT()

        # Function to resize a block of color values
        resize = ResizeGrid()

        # Number of color components in the scan
        componentsAmount = len(myColorComponents)

        # Decode all MCUs in the entropy encoded data
        currentMcu = 0
        previous_dc = np.zeros(componentsAmount, dtype="int16")
        while currentMcu < self.mcuCount:
            # (x, y) coordinates, on the image, for the current MCU
            mcu_y, mcu_x = divmod(currentMcu, self.mcuCount_h)

            # Loop through all color components
            for depth, (componentId, component) in enumerate(myColorComponents.items()):
                # Quantization table of the color component
                quantizationTable = self.quantizationTables[component.dqtId]

                # Minimum coding unit (MCU) of the component
                if componentsAmount > 1:
                    myMcu = np.zeros(shape=component.shape, dtype="int16")
                    repeat = component.repeat
                else:
                    myMcu = np.zeros(shape=(8, 8), dtype="int16")
                    repeat = 1

                for blockCount in range(repeat):
                    # Block of 8 x 8 pixels for the color component
                    block = np.zeros(64, dtype="int16")

                    # DC value of the block
                    tableId = huffmanTablesId[componentId].dc
                    huffmanTable: dict = self.huffmanTables[tableId]
                    huffmanValue = nextHuffval()

                    dcValue = (
                        binTwosComplement(nextBits(huffmanValue)) + previous_dc[depth]
                    )
                    previous_dc[depth] = dcValue
                    block[0] = dcValue

                    # AC values of the block
                    tableId = huffmanTablesId[componentId].ac
                    huffmanTable: dict = self.huffmanTables[tableId]
                    index = 1
                    while index < 64:
                        huffmanValue = nextHuffval()

                        # A huffmanValue of 0 means the 'end of block' (all remaining AC values are zero)
                        if huffmanValue == 0x00:
                            break

                        # Amount of zeroes before the next AC value
                        zeroRunLength = huffmanValue >> 4
                        index += zeroRunLength
                        if index >= 64:
                            break

                        # Get the AC value
                        ac_bitLength = huffmanValue & 0x0F

                        if ac_bitLength > 0:
                            acValue = binTwosComplement(nextBits(ac_bitLength))
                            block[index] = acValue

                        # Go to the next AC value
                        index += 1

                    # Undo the zigzag scan and apply dequantization
                    block = undoZigzag(block) * quantizationTable

                    # Apply the inverse discrete cosine transform (IDCT)
                    block = idct(block)

                    # Coordinates of the block on the current MCU
                    block_y, block_x = divmod(blockCount, component.horizontalSampling)
                    block_y, block_x = 8 * block_y, 8 * block_x

                    # Add the block to the MCU
                    myMcu[block_x : block_x + 8, block_y : block_y + 8] = block

                # Upsample the block if necessary
                if component.shape != self.sampleShape:
                    myMcu = resize(myMcu, self.sampleShape)

                # Add the MCU to the image
                x = self.mcuWidth * mcu_x
                y = self.mcuHeight * mcu_y
                self.imageArray[
                    x : x + self.mcuWidth, y : y + self.mcuHeight, component.order
                ] = myMcu

            # Go to the next MCU
            currentMcu += 1

            # Check for restart interval
            if (
                (self.restartInterval > 0)
                and (currentMcu % self.restartInterval == 0)
                and (currentMcu != self.mcuCount)
            ):
                nextBits(amount=0, restart=True)
                previous_dc[:] = 0
        self.scanCount += 1

    def progressiveDctScan(
        self,
        huffmanTablesId: dict,
        myColorComponents: dict,
        spectralSelectionStart: int,
        spectralSelectionEnd: int,
        bitPositionHigh: int,
        bitPositionLow: int,
    ) -> None:
        # Whether to the scan contains DC or AC values
        if (spectralSelectionStart == 0) and (spectralSelectionEnd == 0):
            values = "dc"
        elif (spectralSelectionStart > 0) and (
            spectralSelectionEnd >= spectralSelectionStart
        ):
            values = "ac"
        else:
            raise CorruptedJpeg(
                "Error - Progressive JPEG images cannot contain both DC and AC values in the same scan."
            )
        """NOTE
        In sequential JPEG both DC and AC values come in the same scan, however in progressive JPEG
        they must come in different scans.
        """

        # Whether this is a refining scan
        if bitPositionHigh == 0:
            refining = False
        elif (bitPositionHigh - bitPositionLow) == 1:
            refining = True
        else:
            raise CorruptedJpeg(
                "Error - Progressive JPEG images cannot contain more than 1 bit for each value on a refining scan."
            )

        print(f"\nScan {self.scanCount+1} of {self.scanAmount}")
        print(
            f"Color components: {', '.join(component.name for component in myColorComponents.values())}"
        )
        print(
            f"Spectral selection: {spectralSelectionStart}-{spectralSelectionEnd} ({values.upper()})"
        )
        print(
            f"Successive approximation: {bitPositionHigh}-{bitPositionLow} ({'refining' if refining else 'first'} scan)"
        )
        print(f"MCU count: {self.mcuCount}")
        print(f"Decoding MCUs...")

        # Function to read the bits from the file's bytes
        nextBits = self.bitsGenerator()

        # Function to decode the next Huffman value
        def nextHuffval() -> int:
            codeword = ""
            huffmanValue = None

            while huffmanValue is None:
                codeword += nextBits()
                if len(codeword) > 16:
                    raise CorruptedJpeg(
                        f"Error - Failed to decode image ({currentMcu}/{self.mcuCount} MCUs decoded)."
                    )
                huffmanValue = huffmanTable.get(codeword)

            return huffmanValue

        # Beginning of scan
        currentMcu = 0
        componentsAmount = len(myColorComponents)
        if (values == "ac") and (componentsAmount > 1):
            raise CorruptedJpeg(
                "Error - An AC progressive scan can only have a single color component."
            )

        # DC values scan
        if values == "dc":
            # First scan (DC)
            if not refining:
                # Previous DC values
                previous_dc = np.zeros(componentsAmount, dtype="int16")

            while currentMcu < self.mcuCount:
                # Loop through all color components
                for depth, (componentId, component) in enumerate(
                    myColorComponents.items()
                ):
                    # (x, y) coordinates, on the image, for the current MCU
                    x = (currentMcu % self.mcuCount_h) * component.shape[0]
                    y = (currentMcu // self.mcuCount_h) * component.shape[1]

                    # Minimum coding unit (MCU) of the component
                    if componentsAmount > 1:
                        repeat = component.repeat
                    else:
                        repeat = 1

                    # Blocks of 8 x 8 pixels for the color component
                    for blockCount in range(repeat):
                        # Coordinates of the block on the current MCU
                        block_y, block_x = divmod(
                            blockCount, component.horizontalSampling
                        )
                        delta_y, delta_x = 8 * block_y, 8 * block_x

                        # First scan of the DC values
                        if not refining:
                            # DC value of the block
                            tableId = huffmanTablesId[componentId].dc
                            huffmanTable: dict = self.huffmanTables[tableId]
                            huffmanValue = nextHuffval()

                            # Get the DC value (partial)
                            dcValue = (
                                binTwosComplement(nextBits(huffmanValue))
                                + previous_dc[depth]
                            )
                            previous_dc[depth] = dcValue

                            # Store the partial DC value on the image array
                            self.imageArray[
                                x + delta_x, y + delta_y, component.order
                            ] = (dcValue << bitPositionLow)

                        # Refining scan for the DC values
                        else:
                            new_bit = int(nextBits())
                            self.imageArray[
                                x + delta_x, y + delta_y, component.order
                            ] |= (new_bit << bitPositionLow)

                # Go to the next MCU
                currentMcu += 1

                # Check for restart interval
                if (
                    (self.restartInterval > 0)
                    and (currentMcu % self.restartInterval == 0)
                    and (currentMcu != self.mcuCount)
                ):
                    nextBits(amount=0, restart=True)
                    if not refining:
                        previous_dc[:] = 0

        # AC values scan
        elif values == "ac":
            # Spectral selection
            spectralSize = (spectralSelectionEnd + 1) - spectralSelectionStart

            # Color component
            ((componentId, component),) = myColorComponents.items()

            # Huffman table
            tableId = huffmanTablesId[componentId].ac
            huffmanTable: dict = self.huffmanTables[tableId]

            # End of band run length
            eobRun = 0

            # Zero run length
            zeroRun = 0

            # Refining function
            def refine_ac() -> None:
                """Perform the refinement of the AC values on a progressive scan"""
                nonlocal to_refine, nextBits, bitPositionLow, component

                # Fetch the bits that will be used to refine the AC values
                # (the bits come in the same order that the values to be refined were found)
                refineBits = nextBits(len(to_refine))

                # Refine the AC values
                ref_index = 0
                while to_refine:
                    ref_x, ref_y = to_refine.popleft()
                    new_bit = int(refineBits[ref_index], 2)
                    self.imageArray[ref_x, ref_y, component.order] |= (
                        new_bit << bitPositionLow
                    )
                    ref_index += 1

            # Queue of AC values that will be refined
            to_refine = deque()

            # Decode and refine the AC values
            currentMcu = 0
            while currentMcu < self.mcuCount:
                # Coordinates of the MCU on the image
                x = (currentMcu % self.mcuCount_h) * 8
                y = (currentMcu // self.mcuCount_h) * 8

                # Loop through the band
                index = spectralSelectionStart
                while (
                    index <= spectralSelectionEnd
                ):  # The element at the end of the band is included
                    # Get the next Huffman value from the encoded data
                    huffmanValue = nextHuffval()
                    runMagnitute = huffmanValue >> 4
                    acBitsLength = huffmanValue & 0x0F

                    # Determine the run length
                    if huffmanValue == 0:
                        # End of band run of 1
                        eobRun = 1
                        break
                    elif huffmanValue == 0xF0:
                        zeroRun = 16
                    elif acBitsLength == 0:
                        # End of band run (length determined by the next bits on the data)
                        # (amount of bands to skip)
                        eobBits = nextBits(runMagnitute)
                        eobRun = (1 << runMagnitute) + int(eobBits, 2)
                        break
                    else:
                        # Amount of zero values to skip
                        zeroRun = runMagnitute

                    # Perform the zero run
                    if not refining and zeroRun:  # First scan
                        index += zeroRun
                        zeroRun = 0
                        """NOTE
                        On the first scan, all AC values skipped are considered to be zero.
                        """
                    else:
                        while zeroRun > 0:  # Refining scan
                            xr, yr = zagzig[index]
                            currentValue = self.imageArray[
                                x + xr, y + yr, component.order
                            ]

                            if currentValue == 0:
                                zeroRun -= 1
                            else:
                                to_refine.append((x + xr, y + yr))

                            index += 1

                    # Decode the next AC value
                    if acBitsLength > 0:
                        acBits = nextBits(acBitsLength)
                        acValue = binTwosComplement(acBits)

                        # Store the AC value on the image array
                        # (the zig-zag scan order is undone to find the position of the value on the image)
                        ac_x, ac_y = zagzig[index]

                        # In order to create a new AC value, the decoder needs to be at a zero value
                        # (the index is moved until a zero is found, other values along the way will be refined)
                        if refining:
                            while (
                                self.imageArray[x + ac_x, y + ac_y, component.order]
                                != 0
                            ):
                                to_refine.append((x + ac_x, y + ac_y))
                                index += 1
                                ac_x, ac_y = zagzig[index]

                        # Create a new acValue
                        self.imageArray[x + ac_x, y + ac_y, component.order] = (
                            acValue << bitPositionLow
                        )

                        # Move to the next value
                        index += 1

                    # Refine AC values skipped by the zero run
                    if refining:
                        refine_ac()
                        """NOTE
                        Following the bits of the AC value (on the image data), come the bits of all
                        values enqueued to be refined. One bit per value, in the same order the values
                        were found. So if N values are going to be refined, then N bits will follow.
                        """

                # Move to the next band if we are at the end of a band
                if index > spectralSelectionEnd:
                    currentMcu += 1
                    if refining:
                        # Coordinates of the MCU on the image
                        x = (currentMcu % self.mcuCount_h) * 8
                        y = (currentMcu // self.mcuCount_h) * 8

                # Perform the end of band run
                if not refining:  # First scan
                    currentMcu += eobRun
                    eobRun = 0
                    """NOTE
                    In the first scan, all the skipped AC values are consideded to be zero.
                    If the EOB run is called when a band has been partially processed, then
                    only the remaining values on the band are considered zero (this band
                    stills count for the EOB run counter).
                    """

                else:  # Refining scan
                    while eobRun > 0:
                        xr, yr = zagzig[index]
                        currentValue = self.imageArray[x + xr, y + yr, component.order]

                        if currentValue != 0:
                            to_refine.append((x + xr, y + yr))

                        index += 1
                        if index > spectralSelectionEnd:
                            # Move to the next band
                            eobRun -= 1
                            currentMcu += 1
                            index = spectralSelectionStart

                            # Coordinates of the MCU on the image
                            x = (currentMcu % self.mcuCount_h) * 8
                            y = (currentMcu // self.mcuCount_h) * 8

                # Refine the AC values found during the EOB run
                if refining:
                    refine_ac()

                # Check for restart interval
                if (
                    (self.restartInterval > 0)
                    and (currentMcu % self.restartInterval == 0)
                    and (currentMcu != self.mcuCount)
                ):
                    nextBits(amount=0, restart=True)

        # Check if all scans have been performed and perform the IDCT
        self.scanCount += 1
        if self.scanCount == self.scanAmount:
            # Function to perform the inverse discrete cosine transform (IDCT)
            idct = InverseDCT()

            # Function to resize a block of color values
            resize = ResizeGrid()

            # Perform the IDCT once all scans have finished
            dctArray = self.imageArray.copy()
            print("\nPerforming IDCT on each color component...")
            for component in self.colorComponents.values():
                # Quantization table used by the component
                quantizationTable = self.quantizationTables[component.dqtId]

                # Subsampling ratio
                ratio_h = self.sampleShape[0] // component.shape[0]
                ratio_v = self.sampleShape[1] // component.shape[1]

                # Dimensions of the MCU of the component
                componentWidth = self.arrayWidth // ratio_h
                componentHeight = self.arrayHeight // ratio_v

                # Amount of MCUs
                mcuCount_h = componentWidth // 8
                mcuCount_v = componentHeight // 8
                mcuCount = mcuCount_h * mcuCount_v

                # Perform the inverse discrete cosine transform (IDCT)
                for currentMcu in range(mcuCount):
                    # Get coordinates of the block
                    x1 = (currentMcu % mcuCount_h) * 8
                    y1 = (currentMcu // mcuCount_h) * 8
                    x2 = x1 + 8
                    y2 = y1 + 8

                    # Undo quantization on the block
                    block = dctArray[x1:x2, y1:y2, component.order]
                    block *= quantizationTable

                    # Perform IDCT on the block to get the color values
                    block = idct(block.reshape(8, 8))

                    # Upsample the block if necessary
                    if component.shape != self.sampleShape:
                        block = resize(block, self.sampleShape)
                        x1 *= ratio_h
                        y1 *= ratio_v
                        x2 *= ratio_h
                        y2 *= ratio_v

                    # Store the color values of the image array
                    self.imageArray[x1:x2, y1:y2, component.order] = block

    def endImage(self, data: bytes) -> None:
        """Method called when the 'end of image' marker is reached.
        The file parsing is finished, the image is converted to RGB and displayed."""

        # Clip the image array to the image dimensions
        self.imageArray = self.imageArray[0 : self.imageWidth, 0 : self.imageHeight, :]

        # Convert image from YCbCr to RGB
        if self.arrayDepth == 3:
            self.imageArray = YCbCrToRgb(self.imageArray)
        elif self.arrayDepth == 1:
            np.clip(self.imageArray, a_min=0, a_max=255, out=self.imageArray)
            self.imageArray = self.imageArray[..., 0].astype("uint8")

        self.scanFinished = True
        print(f"Successfully decoded file : {self.filePath.name} ")
        del self.__file

    def writeOutput(self, extension="txt", binary=False):
        output_file = Path(
            f"{self.filePath.parents[0]}/decoded/{self.filePath.stem}.{extension}"
        )
        output_file.parent.mkdir(exist_ok=True, parents=True)

        img = np.swapaxes(self.imageArray, 0, 1)
        if not binary:
            with open(output_file, "w") as f:
                if extension == "ppm":
                    f.write("P3\n")
                    f.write(f"{self.imageWidth} {self.imageHeight}\n")
                    f.write("255\n")
                for i in range(self.imageHeight):
                    for j in range(self.imageWidth):
                        line = ""
                        for k in range(len(img[i, j])):
                            line += str(img[i, j, k]) + " "
                        f.write(line + "\n")
        elif binary:
            with open(output_file, "wb") as f:
                if extension == "ppm":
                    f.write(b"P6\n")
                    f.write(f"{self.imageWidth} {self.imageHeight}\n".encode())
                    f.write("255\n".encode())
                for i in range(self.imageHeight):
                    for j in range(self.imageWidth):
                        f.write(bytes(img[i, j]))

        print(
            "Output of the decoded image successfully written in the 'decoded' folder."
        )
        return


class InverseDCT:
    """Perform the inverse cosine discrete transform (IDCT) on a 8 x 8 matrix of DCT coefficients."""

    # Precalculate the constant values used on the IDCT function
    # (those values are cached, being calculated only the first time a instance of the class is created)
    idctTable = np.zeros(shape=(8, 8, 8, 8), dtype="float64")
    xyuv_coordinates = tuple(
        product(range(8), repeat=4)
    )  # All 4096 combinations of 4 values from 0 to 7 (each)
    xy_coordinates = tuple(
        product(range(8), repeat=2)
    )  # All 64 combinations of 2 values from 0 to 7 (each)
    for x, y, u, v in xyuv_coordinates:
        # Scaling factors
        Cu = 2 ** (-0.5) if u == 0 else 1.0  # Horizontal
        Cv = 2 ** (-0.5) if v == 0 else 1.0  # Vertical

        # Frequency component
        idctTable[x, y, u, v] = (
            0.25
            * Cu
            * Cv
            * cos((2 * x + 1) * pi * u / 16)
            * cos((2 * y + 1) * pi * v / 16)
        )

    """NOTE
    For an in-depth explanation on how the transform works, please refer to chapter 7 of this book:
    https://research-solution.com/uplode/book/book-26184.pdf
    Compressed Image File Formats, John Miano
    """

    def __call__(self, block: np.ndarray) -> np.ndarray:
        """Takes a 8 x 8 array of DCT coefficients, and performs the inverse discrete
        cosine transform in order to reconstruct the color values.
        """
        # Array to store the results
        output = np.zeros(shape=(8, 8), dtype="float64")

        # Summation of the frequencies components
        for x, y in self.xy_coordinates:
            output[x, y] = np.sum(block * self.idctTable[x, y, ...], dtype="float64")

        # Return the color values
        return np.round(output).astype(block.dtype) + 128


class ResizeGrid:
    """Resize a grid of color values, performing linear interpolation between of those values."""

    # Cache the meshes used for the interpolation
    meshCache = {}
    indicesCache = {}

    def __call__(self, block: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """Takes a 2-dimensional array and resizes it while performing
        linear interpolation between the points.
        """

        # Ratio of the resize
        oldWidth, oldHeight = block.shape
        newWidth, newHeight = new_shape
        key = ((oldWidth, oldHeight), (newWidth, newHeight))

        # Get the interpolation mesh from the cache
        new_xy = self.meshCache.get(key)
        if new_xy is None:
            # If the cache misses, then calculate and cache the mesh
            max_x = oldWidth - 1
            max_y = oldHeight - 1
            numPoints_x = newWidth * 1j
            numPoints_y = newHeight * 1j
            new_x, new_y = np.mgrid[0:max_x:numPoints_x, 0:max_y:numPoints_y]
            new_xy = (new_x, new_y)
            self.meshCache.update({key: new_xy})

        # Get, from the cache, the indices of the values on the original grid
        old_xy = self.indicesCache.get(key[0])
        if old_xy is None:
            # If the cache misses, calculate and cache the indices
            xx, yy = np.indices(block.shape)
            xx, yy = xx.flatten(), yy.flatten()
            old_xy = (xx, yy)
            self.indicesCache.update({key[0]: old_xy})

        # Resize the grid and perform linear interpolation
        resizedBlock = griddata(old_xy, block.ravel(), new_xy)

        return np.round(resizedBlock).astype(block.dtype)


# ----------------------------------------------------------------------------------------------------

# Decoder exceptions


class JpegError(Exception):
    """Parent of all other exceptions of this decoder."""


class NotJpeg(JpegError):
    """File is not a JPEG image."""


class CorruptedJpeg(JpegError):
    """Failed to parse the file headers."""


class UnsupportedJpeg(JpegError):
    """JPEG image is encoded in a way that our decoder does not support."""


# ---------------------------------------------------------------------------------------------------------------------

# Helper functions


def bytesToUint(bytes_obj: bytes):
    pass


def binTwosComplement(bits: str) -> int:
    """Convert a binary number to a signed integer using the two's complement."""
    if bits == "":
        return 0
    elif bits[0] == "1":
        return int(bits, 2)
    elif bits[0] == "0":
        bitLength = len(bits)
        return int(bits, 2) - (2**bitLength - 1)
    else:
        raise ValueError(f"'{bits}' is not a binary number.")


def YCbCrToRgb(imageArray: np.ndarray) -> np.ndarray:
    """Takes a 3-dimensional array representing an image in the YCbCr color
    space, and returns an array of the image in the RGB color space:
    array(width, heigth, YCbCr) -> array(width, heigth, RGB)
    """
    print("\nConverting colors from YCbCr to RGB...")
    Y = imageArray[..., 0].astype("float64")
    Cb = imageArray[..., 1].astype("float64")
    Cr = imageArray[..., 2].astype("float64")

    R = Y + 1.402 * (Cr - 128.0)
    G = Y - 0.34414 * (Cb - 128.0) - 0.71414 * (Cr - 128.0)
    B = Y + 1.772 * (Cb - 128.0)

    output = np.stack((R, G, B), axis=-1)
    np.clip(output, a_min=0.0, a_max=255.0, out=output)

    return np.round(output).astype("uint8")


def undoZigzag(block: np.ndarray) -> np.ndarray:
    """Takes an 1D array of 64 elements and undo the zig-zag scan of the JPEG
    encoding process. Returns a 2D array (8 x 8) that represents a block of pixels.
    """
    return np.array(
        [
            [
                block[0],
                block[1],
                block[5],
                block[6],
                block[14],
                block[15],
                block[27],
                block[28],
            ],
            [
                block[2],
                block[4],
                block[7],
                block[13],
                block[16],
                block[26],
                block[29],
                block[42],
            ],
            [
                block[3],
                block[8],
                block[12],
                block[17],
                block[25],
                block[30],
                block[41],
                block[43],
            ],
            [
                block[9],
                block[11],
                block[18],
                block[24],
                block[31],
                block[40],
                block[44],
                block[53],
            ],
            [
                block[10],
                block[19],
                block[23],
                block[32],
                block[39],
                block[45],
                block[52],
                block[54],
            ],
            [
                block[20],
                block[22],
                block[33],
                block[38],
                block[46],
                block[51],
                block[55],
                block[60],
            ],
            [
                block[21],
                block[34],
                block[37],
                block[47],
                block[50],
                block[56],
                block[59],
                block[61],
            ],
            [
                block[35],
                block[36],
                block[48],
                block[49],
                block[57],
                block[58],
                block[62],
                block[63],
            ],
        ],
        dtype=block.dtype,
    ).T  # <-- transposes the array


# List that undoes the zig-zag ordering for a single element in a band
# (the element index is used on the list, and it returns a (x, y) tuple
# for the coordinates on the data unit)
zagzig = (
    (0, 0),
    (1, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (1, 2),
    (0, 3),
    (0, 4),
    (1, 3),
    (2, 2),
    (3, 1),
    (4, 0),
    (5, 0),
    (4, 1),
    (3, 2),
    (2, 3),
    (1, 4),
    (0, 5),
    (0, 6),
    (1, 5),
    (2, 4),
    (3, 3),
    (4, 2),
    (5, 1),
    (6, 0),
    (7, 0),
    (6, 1),
    (5, 2),
    (4, 3),
    (3, 4),
    (2, 5),
    (1, 6),
    (0, 7),
    (1, 7),
    (2, 6),
    (3, 5),
    (4, 4),
    (5, 3),
    (6, 2),
    (7, 1),
    (7, 2),
    (6, 3),
    (5, 4),
    (4, 5),
    (3, 6),
    (2, 7),
    (3, 7),
    (4, 6),
    (5, 5),
    (6, 4),
    (7, 3),
    (7, 4),
    (6, 5),
    (5, 6),
    (4, 7),
    (5, 7),
    (6, 6),
    (7, 5),
    (7, 6),
    (6, 7),
    (7, 7),
)

# -------------------------------------------------------------------------------------------------------------

# Script :

if __name__ == "__main__":
    import sys

    # if len(sys.argv) < 2:
    #     print("Error - Invalid Argument")
    # else:
    #     try:
    #         for arg in sys.argv[1:]:
    #             decodedJPEG = JPEGReader(arg)
    #             print(decodedJPEG)
    #     except:
    #         print("Error - Something went wrong")
    decodedJPEG = JpegDecoder(
        "/Users/arsnm/Documents/cpge/mp/tipe_mp/code/encoder/data/villeLyonLossLess.jpg"
    )
    decodedJPEG.writeOutput("ppm", True)
