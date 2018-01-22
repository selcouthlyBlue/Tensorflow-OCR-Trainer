class EncoderDecoder:
    def __init__(self):
        self.charset = ""
        self.blank_label = ''
        self.eos_token = '`'
        self.encode_map = {}
        self.decode_map = {}
        self.number_of_characters_in_charset = 0

    def initialize_encode_and_decode_maps_from(self, charset_string):
        self.number_of_characters_in_charset = len(charset_string)
        for index, char in enumerate(list(charset_string)):
            self.encode_map[char] = index
            self.decode_map[index] = char
        self.encode_map[self.eos_token] = self.number_of_characters_in_charset
        self.decode_map[self.number_of_characters_in_charset] = self.eos_token
        self.number_of_characters_in_charset += 1
        self.encode_map[self.blank_label] = self.number_of_characters_in_charset
        self.decode_map[self.number_of_characters_in_charset] = self.blank_label
        self.number_of_characters_in_charset += 1

    def encode(self, string_to_encode):
        return [self.blank_label if string_to_encode == self.blank_label else
                self.encode_map[c] for c in list(string_to_encode)]

    def decode(self, encoded_string_to_decode):
        return ''.join([self.blank_label if not encoded_string_to_decode else
                        self.decode_map[i] for i in encoded_string_to_decode])