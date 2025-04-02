import sys

class CompressionContainer:
    line = 0
    og_code: str
    og_size: int


    is_RLE_start: bool = False
    is_RLE_end: bool = False
    RLE_code: str = "ERROR"

    BITMASK_size: int
    BITMASK_code: str
    is_BITMASK = False

    ONE_BIT_MISMATCH_size: int
    ONE_BIT_MISMATCH_code: str
    is_ONE_BIT_MISMATCH = False

    TWO_BIT_CONSEC_MISMATCH_size: int
    TWO_BIT_CONSEC_MISMATCH_code: str
    is_TWO_BIT_CONSEC_MISMATCH = False

    FOUR_BIT_CONSEC_MISMATCH_size: int
    FOUR_BIT_CONSEC_MISMATCH_code: str
    is_FOUR_BIT_CONSEC_MISMATCH = False

    TWO_BIT_ANYWHERE_MISMATCH_size: int
    TWO_BIT_ANYWHERE_MISMATCH_code: str
    is_TWO_BIT_ANYWHERE_MISMATCH = False

    DIRECT_MATCH_size: int
    DIRECT_MATCH_code: str
    is_DIRECT_MATCH = False


    def __init__(self, og_code: str, line: int):
        self.og_code = og_code
        self.og_size = len(og_code)
        self.line = line

    def hasNoCompression(self) -> bool:
        return not self.is_BITMASK and not self.is_ONE_BIT_MISMATCH and not self.is_TWO_BIT_CONSEC_MISMATCH and not self.is_FOUR_BIT_CONSEC_MISMATCH and not self.is_TWO_BIT_ANYWHERE_MISMATCH and not self.is_DIRECT_MATCH

    def getCompressedBinary(self, is_rle: bool = False):
        if self.is_RLE_start:
            return ''
        elif self.is_RLE_end:
            return self.RLE_code
        if is_rle:
            return ''
        elif self.is_DIRECT_MATCH:
            return self.DIRECT_MATCH_code
        elif self.is_ONE_BIT_MISMATCH:
            return self.ONE_BIT_MISMATCH_code
        elif self.is_TWO_BIT_CONSEC_MISMATCH:
            return self.TWO_BIT_CONSEC_MISMATCH_code
        elif self.is_FOUR_BIT_CONSEC_MISMATCH:
            return self.FOUR_BIT_CONSEC_MISMATCH_code
        elif self.is_BITMASK:
            return self.BITMASK_code
        elif self.is_TWO_BIT_ANYWHERE_MISMATCH:
            return self.TWO_BIT_ANYWHERE_MISMATCH_code
        else:
            return "000" + self.og_code



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

class DecompressionContainer:

    def __init__(self, tag, data):
        self.tag = tag
        self.data = data

        self.dict_index = None
        self.start_pos = None
        self.bitmask = None
        self.mismatch_pos = None
        self.mismatch_pos1 = None
        self.mismatch_pos2 = None
        self.repetitions = None
        self._parse_data()

    def _parse_data(self):
        if self.tag == "000":
            pass

        elif self.tag == "001":
            self.repetitions = int(self.data, 2) + 1

        elif self.tag == "010":
            self.start_pos = int(self.data[:5], 2)
            self.bitmask = self.data[5:9]
            self.dict_index = self.data[9:]

        elif self.tag == "011":
            self.mismatch_pos = int(self.data[:5], 2)
            self.dict_index = self.data[5:]

        elif self.tag == "100":
            self.start_pos = int(self.data[:5], 2)
            self.dict_index = self.data[5:]

        elif self.tag == "101":
            self.start_pos = int(self.data[:5], 2)
            self.dict_index = self.data[5:]

        elif self.tag == "110":
            self.mismatch_pos1 = int(self.data[:5], 2)
            self.mismatch_pos2 = int(self.data[5:10], 2)
            self.dict_index = self.data[10:]

        elif self.tag == "111":
            self.dict_index = self.data

    def apply_bitmask(self, dict_binary):
        binary_list = list(dict_binary)

        for i in range(4):
            if self.bitmask[i] == '1':
                binary_list[self.start_pos + i] = '1' if binary_list[self.start_pos + i] == '0' else '0'

        return ''.join(binary_list)

    def apply_one_bit_mismatch(self, dict_binary):
        binary_list = list(dict_binary)
        binary_list[self.mismatch_pos] = '1' if binary_list[self.mismatch_pos] == '0' else '0'
        return ''.join(binary_list)

    def apply_two_bit_consec_mismatch(self, dict_binary):
        binary_list = list(dict_binary)

        for i in range(2):
            binary_list[self.start_pos + i] = '1' if binary_list[self.start_pos + i] == '0' else '0'

        return ''.join(binary_list)

    def apply_four_bit_consec_mismatch(self, dict_binary):
        binary_list = list(dict_binary)

        for i in range(4):
            binary_list[self.start_pos + i] = '1' if binary_list[self.start_pos + i] == '0' else '0'

        return ''.join(binary_list)

    def apply_two_bit_anywhere_mismatch(self, dict_binary):
        binary_list = list(dict_binary)
        binary_list[self.mismatch_pos1] = '1' if binary_list[self.mismatch_pos1] == '0' else '0'
        binary_list[self.mismatch_pos2] = '1' if binary_list[self.mismatch_pos2] == '0' else '0'
        return ''.join(binary_list)

    def decompress(self, dict_mapping, previous_binary=None):
        if self.tag == "000":
            return self.data

        elif self.tag == "001":
            return previous_binary if previous_binary else None

        elif self.tag == "010":
            if self.dict_index in dict_mapping:
                return self.apply_bitmask(dict_mapping[self.dict_index])
            return None

        elif self.tag == "011":
            if self.dict_index in dict_mapping:
                return self.apply_one_bit_mismatch(dict_mapping[self.dict_index])
            return None

        elif self.tag == "100":
            if self.dict_index in dict_mapping:
                return self.apply_two_bit_consec_mismatch(dict_mapping[self.dict_index])
            return None

        elif self.tag == "101":
            if self.dict_index in dict_mapping:
                return self.apply_four_bit_consec_mismatch(dict_mapping[self.dict_index])
            return None

        elif self.tag == "110":
            if self.dict_index in dict_mapping:
                return self.apply_two_bit_anywhere_mismatch(dict_mapping[self.dict_index])
            return None

        elif self.tag == "111":
            return dict_mapping.get(self.dict_index, None)

        return None

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

OG_TAG = "000"
RLE_TAG = "001"
BITMASK_TAG = "010"
ONE_BIT_MISMATCH_TAG = "011"
TWO_BIT_CONSEC_MISMATCH_TAG = "100"
FOUR_BIT_CONSEC_MISMATCH_TAG = "101"
TWO_BIT_ANYWHERE_MISMATCH_TAG = "110"
DIRECT_MATCH_TAG = "111"

#“I have neither given nor received any unauthorized aid on this assignment”.

def direct_match(containers: list, direct_matches: list):
    for container_index, dict_index in direct_matches:
        containers[container_index].DIRECT_MATCH_code = DIRECT_MATCH_TAG + dict_index
        containers[container_index].DIRECT_MATCH_size = len(containers[container_index].DIRECT_MATCH_code)
        containers[container_index].is_DIRECT_MATCH = True

    return 0

def two_bit_anywhere_mismatch(containers: list, two_bit_anywhere_mismatches: list):
    for container_index, dict_index, mismatch_positions in two_bit_anywhere_mismatches:
        containers[container_index].TWO_BIT_ANYWHERE_MISMATCH_code = TWO_BIT_ANYWHERE_MISMATCH_TAG + format(mismatch_positions[0], '05b') + format(mismatch_positions[1], '05b') + dict_index
        containers[container_index].TWO_BIT_ANYWHERE_MISMATCH_size = len(containers[container_index].TWO_BIT_ANYWHERE_MISMATCH_code)
        containers[container_index].is_TWO_BIT_ANYWHERE_MISMATCH = True

    return 0

def one_bit_mismatch(containers: list, one_bit_mismatches: list):
    for container_index, dict_index, mismatch_position in one_bit_mismatches:
        containers[container_index].ONE_BIT_MISMATCH_code = ONE_BIT_MISMATCH_TAG + format(mismatch_position, '05b') + dict_index
        containers[container_index].ONE_BIT_MISMATCH_size = len(containers[container_index].ONE_BIT_MISMATCH_code)
        containers[container_index].is_ONE_BIT_MISMATCH = True

    return 0

def two_bit_consec_mismatch(containers: list, two_bit_consec_mismatches: list):
    for container_index, dict_index, start_pos in two_bit_consec_mismatches:
        containers[container_index].TWO_BIT_CONSEC_MISMATCH_code = TWO_BIT_CONSEC_MISMATCH_TAG + format(start_pos, '05b') + dict_index
        containers[container_index].TWO_BIT_CONSEC_MISMATCH_size = len(containers[container_index].TWO_BIT_CONSEC_MISMATCH_code)
        containers[container_index].is_TWO_BIT_CONSEC_MISMATCH = True

    return 0

def four_bit_consec_mismatch(containers: list, four_bit_consec_mismatches: list):
    for container_index, dict_index, start_pos in four_bit_consec_mismatches:
        containers[container_index].FOUR_BIT_CONSEC_MISMATCH_code = FOUR_BIT_CONSEC_MISMATCH_TAG + format(start_pos, '05b') + dict_index
        containers[container_index].FOUR_BIT_CONSEC_MISMATCH_size = len(containers[container_index].FOUR_BIT_CONSEC_MISMATCH_code)
        containers[container_index].is_FOUR_BIT_CONSEC_MISMATCH = True

    return 0

def bitmask(containers: list, valid_bitmasks: list):
    for container_index, dict_index, start_pos, bitmask in valid_bitmasks:
        containers[container_index].BITMASK_code = BITMASK_TAG + format(start_pos, '05b') + bitmask + dict_index
        containers[container_index].BITMASK_size = len(containers[container_index].BITMASK_code)
        containers[container_index].is_BITMASK = True
    return 0

def RLE(containers: list, dictionary: dict):
    rle_pairs = dictionary.get('all_pairs', [])

    for start, end in rle_pairs:
        start_idx = start - 1
        end_idx = end - 1

        containers[start_idx + 1].is_RLE_start = True
        repetitions = end_idx - start_idx
        rle_count = format(repetitions - 1, '03b')

        containers[start_idx].RLE_code = "001" + rle_count
        containers[end_idx].RLE_code = "001" + rle_count

        containers[end_idx].is_RLE_end = True

    return 0

def debug_print(containers: list, rle_dictionary: dict, freq_dictionary: dict, valid_bistmasks: list, reverse_dict: dict, one_bit_mismatches: list, direct_matches: list, two_bit_anywhere_mismatches: list):
    counter = 1
    for container in containers:
        print(f"{counter}. {container.og_code}")

        if container.is_RLE_start:
            print(f"   RLE START: {container.RLE_code}")

        if container.is_RLE_end:
            print(f"   RLE END")

        if container.is_BITMASK:
            print(f"   BITMASK: {container.BITMASK_code}")

        if container.is_ONE_BIT_MISMATCH:
            print(f"   ONE BIT MISMATCH: {container.ONE_BIT_MISMATCH_code}")

        if container.is_TWO_BIT_CONSEC_MISMATCH:
            print(f"   TWO BIT CONSEC MISMATCH: {container.TWO_BIT_CONSEC_MISMATCH_code}")

        if container.is_FOUR_BIT_CONSEC_MISMATCH:
            print(f"   FOUR BIT CONSEC MISMATCH: {container.FOUR_BIT_CONSEC_MISMATCH_code}")

        if container.is_TWO_BIT_ANYWHERE_MISMATCH:
            print(f"   TWO BIT ANYWHERE MISMATCH: {container.TWO_BIT_ANYWHERE_MISMATCH_code}")

        if container.is_DIRECT_MATCH:
            print(f"   DIRECT MATCH: {container.is_DIRECT_MATCH}")

        if container.hasNoCompression():
            print(f"   NO COMPRESSION 000 + {container.og_code}")

        counter += 1
        print()

    print("\nRLE Dictionary Contents:")
    for key, value in rle_dictionary.items():
        print(f'Key: {key}: Value: {value}')

    print("\nFrequency Dictionary Contents:")
    for key, value in freq_dictionary.items():
        print(f'Key: {key}: Value: {value}')

    print("\nValid Bitmask Options:")
    print("container_index, dict_index, start_pos, bitmask")
    for container_index, dict_index, start_pos, bitmask in valid_bistmasks:
        print(f"Container {container_index + 1}: Dictionary {dict_index}, Start {start_pos}, Bitmask {bitmask}, Dict Binary {reverse_dict[dict_index]}")

    print("\nOne Bit Mismatches:")
    print("container_index, dict_index, mismatch_position")
    for container_index, dict_index, mismatch_position in one_bit_mismatches:
        print(f"Container {container_index + 1}: Dictionary {dict_index}, Mismatch Position {mismatch_position}, Dict Binary {reverse_dict[dict_index]}")

    print("\nDirect Matches:")
    print("container_index, dict_index")
    for container_index, dict_index in direct_matches:
        print(f"Container {container_index + 1}: Dictionary {dict_index}, Binary {containers[container_index].og_code}, Dict Binary {reverse_dict[dict_index]}")

    print("\nTwo Bit Anywhere Mismatches:")
    print("container_index, dict_index, mismatch_positions")
    for container_index, dict_index, mismatch_positions in two_bit_anywhere_mismatches:
        print(f"Container {container_index + 1}: Dictionary {dict_index}, Mismatch Positions {mismatch_positions}, Binary {containers[container_index].og_code}, Dict Binary {reverse_dict[dict_index]}")

def load_two_bit_anywhere_mismatch(containers: list, freq_dictionary: dict, reverse_dict: dict) -> list:
    two_bit_anywhere_mismatches = []

    for container_index, container in enumerate(containers):
        og_code = container.og_code

        for dict_binary, dict_index in freq_dictionary.items():
            mismatches = 0
            mismatch_positions = []

            for i in range(len(og_code)):
                if og_code[i] != dict_binary[i]:
                    mismatches += 1
                    mismatch_positions.append(i)
                    if mismatches > 2:
                        break

            if mismatches == 2:
                two_bit_anywhere_mismatches.append((container_index, dict_index, mismatch_positions))

    return two_bit_anywhere_mismatches

def load_direct_match(containers: list, freq_dictionary: dict, reverse_dict: dict) -> list:
    direct_matches = []

    for container_index, container in enumerate(containers):
        og_code = container.og_code

        if og_code in freq_dictionary:
            direct_matches.append((container_index, freq_dictionary[og_code]))

    return direct_matches

def load_one_bit_mismatch(containers: list, freq_dictionary: dict, reverse_dict: dict) -> list:
    one_bit_mismatches = []

    for container_index, container in enumerate(containers):
        og_code = container.og_code

        for dict_binary, dict_index in freq_dictionary.items():
            mismatches = 0
            mismatch_position = -1

            for i in range(len(og_code)):
                if og_code[i] != dict_binary[i]:
                    mismatches += 1
                    mismatch_position = i
                    if mismatches > 1:
                        break

            if mismatches == 1:
                total_mismatches = sum(1 for i in range(len(og_code)) if og_code[i] != dict_binary[i])
                if total_mismatches == 1:
                    one_bit_mismatches.append((container_index, dict_index, mismatch_position))

    return one_bit_mismatches

def load_two_bit_consec_mismatch(containers: list, freq_dictionary: dict, reverse_dict: dict) -> list:
    two_bit_consec_mismatches = []

    for container_index, container in enumerate(containers):
        og_code = container.og_code.strip()

        for dict_binary, dict_index in freq_dictionary.items():
            dict_binary = dict_binary.strip()

            for start_pos in range(31):
                mismatches = 0

                if start_pos + 1 < len(og_code) and start_pos + 1 < len(dict_binary):
                    if og_code[start_pos] != dict_binary[start_pos]:
                        mismatches += 1
                    if og_code[start_pos + 1] != dict_binary[start_pos + 1]:
                        mismatches += 1

                    if mismatches == 2:
                        total_mismatches = sum(1 for i in range(len(og_code))
                                               if i < len(dict_binary) and og_code[i] != dict_binary[i])

                        if total_mismatches == 2:
                            two_bit_consec_mismatches.append((container_index, dict_index, start_pos))

    return two_bit_consec_mismatches

def load_four_bit_consec_mismatch(containers: list, freq_dictionary: dict, reverse_dict: dict) -> list:
    four_bit_consec_mismatches = []

    for container_index, container in enumerate(containers):
        og_code = container.og_code.strip()

        for dict_binary, dict_index in freq_dictionary.items():
            dict_binary = dict_binary.strip()

            for start_pos in range(29):
                mismatches = 0

                if start_pos + 3 < len(og_code) and start_pos + 3 < len(dict_binary):
                    for i in range(4):
                        if og_code[start_pos + i] != dict_binary[start_pos + i]:
                            mismatches += 1

                    if mismatches == 4:
                        total_mismatches = sum(1 for i in range(len(og_code))
                                               if i < len(dict_binary) and og_code[i] != dict_binary[i])

                        if total_mismatches == 4:
                            four_bit_consec_mismatches.append((container_index, dict_index, start_pos))

    return four_bit_consec_mismatches

def load_valid_bitmasks(containers: list, freq_dictionary: dict) -> list:
    valid_bitmasks = []

    sorted_dict = sorted(freq_dictionary.items(), key=lambda x: int(x[1], 2))

    for container_index, container in enumerate(containers):
        og_code = container.og_code.strip()
        bitmask_found = False

        for dict_binary, dict_index in sorted_dict:
            if bitmask_found:
                break

            mismatches = 0
            mismatch_positions = []

            for i in range(len(og_code)):
                if i < len(dict_binary) and og_code[i] != dict_binary[i]:
                    mismatches += 1
                    mismatch_positions.append(i)
                    if mismatches > 4:
                        break

            if 1 <= mismatches <= 4:
                mismatch_positions.sort()

                for start_pos in range(29):
                    window = list(range(start_pos, start_pos + 4))
                    mismatches_in_window = [pos for pos in mismatch_positions if pos in window]

                    if (len(mismatches_in_window) == len(mismatch_positions) and
                            start_pos in mismatch_positions):

                        bitmask = ['0', '0', '0', '0']
                        for pos in mismatch_positions:
                            if pos in window:
                                bitmask[pos - start_pos] = '1'

                        if bitmask[0] == '1':
                            valid_bitmasks.append((container_index, dict_index, start_pos, ''.join(bitmask)))
                            bitmask_found = True
                            break

    return valid_bitmasks

def load_containers() -> list:
    containers = []
    count = 1
    with open("original.txt") as file:
        for line in file:
            container = CompressionContainer(line, count)
            containers.append(container)
            count += 1
    return containers

def load_rle_dictionary(containers: list) -> dict:
    dictionary = {}
    pairs = []

    if not containers:
        return dictionary

    i = 0
    while i < len(containers):
        curr_line = containers[i].og_code
        start = i

        j = i + 1
        while j < len(containers) and containers[j].og_code == curr_line:
            j += 1

        seq_length = j - start

        if seq_length > 1:
            remaining = seq_length
            pos = start

            while remaining > 0:
                chunk_size = min(9, remaining)

                if chunk_size > 1:
                    pairs.append((pos, pos + chunk_size - 1))

                pos += chunk_size
                remaining -= chunk_size

        i = j

    dictionary['all_pairs'] = [(start + 1, end + 1) for start, end in pairs]

    return dictionary

def write_compression_output(containers: list, freq_dictionary: dict):
    file = open("cout.txt", "w")
    is_in_RLE = False
    output_buffer = ""

    for i, container in enumerate(containers):
        if container.is_RLE_end:
            is_in_RLE = False

            con_code = container.og_code
            con_code = con_code.replace("\n", "")

            compressed_binary = container.getCompressedBinary(is_in_RLE)
            compressed_binary = compressed_binary.replace("\n", "")
            output_buffer += compressed_binary

            while len(output_buffer) >= 32:
                file.write(output_buffer[:32] + "\n")
                output_buffer = output_buffer[32:]

            continue

        if container.is_RLE_start:
            is_in_RLE = True

        if is_in_RLE:
            continue

        con_code = container.og_code
        con_code = con_code.replace("\n", "")

        compressed_binary = container.getCompressedBinary(is_in_RLE)
        compressed_binary = compressed_binary.replace("\n", "")
        output_buffer += compressed_binary

        while len(output_buffer) >= 32:
            file.write(output_buffer[:32] + "\n")
            output_buffer = output_buffer[32:]

    if output_buffer:
        file.write(output_buffer.ljust(32, '0') + "\n")

    file.write("xxxx" + "\n")
    for key, value in freq_dictionary.items():
        if key == list(freq_dictionary.keys())[-1]:
            key = key.replace("\n", "")
            file.write(key)
        else:
            file.write(key)


    file.close()

def load_freq_dictionary_read_from_file() -> dict:
    dictionary = {}
    hasPassedSeparator = False

    with open("compressed.txt") as file:
        for line in file:
            if not hasPassedSeparator:
                if line == "xxxx\n":
                    hasPassedSeparator = True
                continue

            # Strip newline characters
            clean_line = line.strip()
            clean_line = clean_line + "\n"
            dictionary[clean_line] = format(len(dictionary), '04b')

    print(len(dictionary), "read")
    for key, value in dictionary.items():
        print(f"Key: {key}: Value: {value}")
        print(len(key))
        for i in range(len(key)):
            print(key[i])

    return dictionary

def load_freq_dictionary_generated(containers: list) -> dict:
    dictionary = {}

    og_code_counter = {}
    for container in containers:
        og_code = container.og_code
        if og_code in og_code_counter:
            og_code_counter[og_code] += 1
        else:
            og_code_counter[og_code] = 1

    top_16 = []
    for i in range(16):
        max_count = -1
        max_code = None
        for code, count in og_code_counter.items():
            if count > max_count and code not in top_16:
                max_count = count
                max_code = code
        if max_code is not None:
            top_16.append(max_code)

    for index, code in enumerate(top_16):
        dictionary[code] = format(index, '04b')

    print(len(dictionary), "custom")
    for key, value in dictionary.items():
        print(f"Key: {key}: Value: {value}")
        print(len(key))
        for i in range(len(key)):
            print(key[i])

    return dictionary

def compress():
    containers = load_containers()
    rle_dictionary = load_rle_dictionary(containers)
    freq_dictionary = load_freq_dictionary_read_from_file()

    reverse_dict = {}
    for binary, index in freq_dictionary.items():
        reverse_dict[index] = binary

    valid_bitmasks = load_valid_bitmasks(containers, freq_dictionary)
    one_bit_mismatches = load_one_bit_mismatch(containers, freq_dictionary, reverse_dict)
    direct_matches = load_direct_match(containers, freq_dictionary, reverse_dict)
    two_bit_anywhere_mismatches = load_two_bit_anywhere_mismatch(containers, freq_dictionary, reverse_dict)
    two_bit_consec_mismatches = load_two_bit_consec_mismatch(containers, freq_dictionary, reverse_dict)
    four_bit_consec_mismatches = load_four_bit_consec_mismatch(containers, freq_dictionary, reverse_dict)


    RLE(containers, rle_dictionary)
    bitmask(containers, valid_bitmasks)
    two_bit_anywhere_mismatch(containers, two_bit_anywhere_mismatches)
    direct_match(containers, direct_matches)
    one_bit_mismatch(containers, one_bit_mismatches)
    two_bit_consec_mismatch(containers, two_bit_consec_mismatches)
    four_bit_consec_mismatch(containers, four_bit_consec_mismatches)

    write_compression_output(containers, freq_dictionary)

    #debug_print(containers, rle_dictionary, freq_dictionary, valid_bitmasks, reverse_dict, one_bit_mismatches, direct_matches, two_bit_anywhere_mismatches)
#118 delete

def read_compressed_file():
    compressed_data = ""
    dictionary = []
    separator_found = False

    with open("compressed.txt", "r") as file:
        for line in file:
            line = line.strip()
            if line == "xxxx":
                separator_found = True
                continue

            if separator_found:
                dictionary.append(line)
            else:
                compressed_data += line

    return compressed_data, dictionary

def create_dict_mapping(dictionary):
    dict_mapping = {}
    for i, entry in enumerate(dictionary):
        dict_mapping[format(i, '04b')] = entry
    return dict_mapping

def parse_compressed_data(compressed_data):
    tag_to_length_dict = {
        OG_TAG: 32,
        RLE_TAG: 3,
        BITMASK_TAG: 13,
        ONE_BIT_MISMATCH_TAG: 9,
        TWO_BIT_CONSEC_MISMATCH_TAG: 9,
        FOUR_BIT_CONSEC_MISMATCH_TAG: 9,
        TWO_BIT_ANYWHERE_MISMATCH_TAG: 14,
        DIRECT_MATCH_TAG: 4
    }

    containers = []
    position = 0

    while position < len(compressed_data):

        if position + 3 > len(compressed_data):
            break

        tag = compressed_data[position:position + 3]
        position += 3

        data_length = tag_to_length_dict.get(tag, 0)

        if position + data_length > len(compressed_data):
            break

        data = compressed_data[position:position + data_length]
        position += data_length

        container = DecompressionContainer(tag, data)
        containers.append(container)

    return containers

def decompress_containers(containers, dict_mapping):
    decompressed_binaries = []
    previous_binary = None
    i = 0

    while i < len(containers):
        container = containers[i]

        if container.tag == RLE_TAG:
            if previous_binary is None:
                i += 1
                continue

            for _ in range(container.repetitions):
                decompressed_binaries.append(previous_binary)

            i += 1
            continue

        binary = container.decompress(dict_mapping, previous_binary)

        if binary:
            decompressed_binaries.append(binary)
            previous_binary = binary

        i += 1

    return decompressed_binaries

def write_decompression_output(decompressed_binaries):
    with open("dout.txt", "w") as file:
        for i, binary in enumerate(decompressed_binaries):
            if i == len(decompressed_binaries) - 1:
                file.write(binary)
            else:
                file.write(binary + "\n")

def decompress():
    compressed_data, dictionary = read_compressed_file()
    dict_mapping = create_dict_mapping(dictionary)
    containers = parse_compressed_data(compressed_data)
    decompressed_binaries = decompress_containers(containers, dict_mapping)
    write_decompression_output(decompressed_binaries)


def main():
    try:
        if sys.argv[1] == "1":
            compress()

        if sys.argv[1] == "2":
            decompress()
    except IndexError:
        print("Please enter 1 for compression or 2 for decompression")
        a = input()
        if a == "1":
            compress()
        if a == "2":
            decompress()

if __name__ == "__main__":
    main()