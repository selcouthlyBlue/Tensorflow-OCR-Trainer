class EncoderDecoder:
    def __init__(self):
        self.charset = ""
        self.eos_token = '`'
        self.encode_map = {}
        self.decode_map = {}

    def initialize_encode_and_decode_maps_from(self, charset_string):
        for index, char in enumerate(list(charset_string)):
            self._add_to_encode_decode_maps(char, index)

    def _add_to_encode_decode_maps(self, char, index):
        self.encode_map[char] = index
        self.decode_map[index] = char

    def encode(self, string_to_encode):
        return [self.encode_map[c] for c in list(string_to_encode)]

    def decode(self, encoded_string_to_decode):
        return ''.join([self.decode_map[i] for i in encoded_string_to_decode])
