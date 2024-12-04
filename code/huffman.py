# Module computing Huffman compression


from collections import Counter, namedtuple
from heapq import heapify, heappop, heappush


# Node in a Huffman Tree
Node = namedtuple("Node", ["char", "freq"])

class HuffmanCompressor:
    """Huffman compression implementation"""
    def __init__(self):
        self.encoding_table = {}
        self.decoding_table = {}

    def build_tables(self, s: str):
        """create both the encodingn and decoding tables of a given string
    
    parameters
    ----------
        -s : string used to build the tables
    
    return
    ------
        - fill both the encoding and decoding table of the given class instance"""
    
        freq_table = Counter(s)
        
        # create a heap of the nodes in the tree
        heap = []
        for char, freq in freq_table.items():
            heap.append(Node(char, freq))
        heapify(heap)
        
        # create the Huffman tree
        while len(heap) > 1:
            left_node = heappop(heap)
            right_node = heappop(heap)
            combined_node = Node(None, left_node.freq + right_node.freq)
            heappush(heap, combined_node)
        
        def build_encoding_table(node, code=''):
            if node.char is not None:
                # if the node is a leaf, add it to the encoding table
                self.encoding_table[node.char] = code
                return
            # if the node is not a leaf, recursively build the encoding table
            build_encoding_table(node.left, code + '0')
            build_encoding_table(node.right, code + '1')
        
        build_encoding_table(heap[0])

        
        def build_decoding_table(node, code=''):
            if node.char is not None:
                # if the node is a leaf, add it to the decoding table
                self.decoding_table[code] = node.char
                return
            # if the node is not a leaf, recursively build the decoding table
            build_decoding_table(node.left, code + "0")
            build_decoding_table(node.right, code + "1")
        
        build_decoding_table(heap[0])
    
    def compress(self, s: str) -> str:
        """compress the inputed string
    
    parameters
    ----------
        -s : string to be compressed
    
    return
    ------
        - compressed string"""
        compressed = ""
        for char in s:
            compressed += self.encoding_table[char]
        return compressed
    
    def decompress(self, compressed: str) -> str:
        """decompress the inputed string
    
    parameters
    ----------
        -s : string to be compressed
    
    return
    ------
        - decompressed string"""
        decompressed = ""
        i = 0
        while i < len(compressed):
            for j in range(i+1, len(compressed)+1):
                if compressed[i:j] in self.decoding_table:
                    decompressed += self.decoding_table[compressed[i:j]]
                    i = j
                    break
        
        return decompressed


