import os
import sentencepiece as spm


class Tokenizer:
    def __init__(
        self,
        sentencepiece_path,
        max_len=128,
        vocab_size=1024,
        max_sentencepiece_length=7,
        pad_token="<pad>",
        unk_token="<unk>",
        start_token="<s>",
        end_token="</s>",
        # mask_token="<mask>",
        # cls_token="<cls>"
    ):
        self.sentencepiece_path = sentencepiece_path
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        # self.mask_token = mask_token
        # self.cls_token = cls_token
        self.max_sentencepiece_length = max_sentencepiece_length

    def load(self):
        self.initialize()
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(f"{self.sentencepiece_path}.model")

    def initialize(self):
        try:
            vocab_file = open(f"{self.sentencepiece_path}.vocab", "r")
        except Exception as e:
            print(
                f"{e}: {self.sentencepiece_path}.vocab file not found.\
                   Train your model first (-_-)"
            )
            return

        self.vocab = [i.split("\t")[0].strip() for i in vocab_file.readlines()]
        self.char_to_idx = dict(zip(self.vocab, range(self.vocab_size)))

        assert self.vocab_size == len(self.char_to_idx), (
            "vocab size must be equal to char_to_index,"
            + "you should retrain your model"
            + " with appropriate vocab size."
        )

    def train(
        self, txt_file_path: os.path, model_type: str = "bpe", replace: bool = False
    ):
        """Train BPE based sentencepiece model

        Args:
            txt_file_path (os.path): Text file path.
            model_type (str, optional): Byte-Pair-Encoding. Default: 'bpe'
            replace (bool, optional): to replace previous model. Default: False
        """

        if os.path.exists(f"{self.sentencepiece_path}.model"):
            print("Model already exists. Enable replace flag to retrain.")
            if not replace:
                return

        cmd = (
            f"--input={txt_file_path}"
            + " --max_sentence_length=8000"
            + " --train_extremely_large_corpus=true"
            + f" --vocab_size={self.vocab_size}"
            + f" --model_prefix={self.sentencepiece_path}.{str(self.vocab_size)}"
            + " --pad_id=0"
            + " --unk_id=1"
            + " --bos_id=2"
            + " --eos_id=3"
            # + " --mask_id=5"
            # + " --cls_id==6"
            + f" --pad_piece={self.pad_token}"
            + f" --unk_piece={self.unk_token}"
            + f" --bos_piece={self.start_token}"
            + f" --eos_piece={self.end_token}"
            # + f" --mask_piece={self.mask_token}"
            # + f" --cls_piece={self.cls_token}"
            + f" --max_sentencepiece_length={self.max_sentencepiece_length}"
            + f" --model_type={model_type}"
        )
        spm.SentencePieceTrainer.train(cmd)

    def __call__(self, text, padding: bool = False):
        tokenized_text = self.tokenizer.encode_as_ids(text)
        tokenized_text = tokenized_text[: self.max_len - 2]
        pad_len = self.max_len - len(tokenized_text) - 2

        padded_text = (
            [self.tokenizer.bos_id()]
            + tokenized_text
            + [self.tokenizer.eos_id()]
            + [self.tokenizer.pad_id()] * pad_len * int(padding)
        )

        return padded_text, 2 + len(tokenized_text)

    def tokenize(self, text):
        return self.encode(text)

    def encode(self, text, *args, **karws):
        text = self.tokenizer.encode_as_ids(text)
        return text

    @property
    def pad_token_id(self):
        return self.char_to_idx[self.pad_token]

    @property
    def start_token_id(self):
        return self.char_to_idx[self.start_token]

    @property
    def end_token_id(self):
        return self.char_to_idx[self.end_token]

    @property
    def unk_token_id(self):
        return self.char_to_idx[self.unk_token]

    def get_id(self, token):
        return self.char_to_idx.get(token, self.unk_token_id)

    @property
    def cls_token_id(self):
        return self.pad_token_id


""" Reference for training a tokenizer

0. https://github.com/google/sentencepiece

1. Sentencepiece Tokenizer With Offsets For T5, ALBERT, XLM-RoBERTa And Many More
   https://www.youtube.com/watch?v=U51ranzJBpY

2. Understanding SentencePiece ([Under][Standing][_Sentence][Piece])
   https://jacky2wong.medium.com/understanding-sentencepiece-under-standing-sentence-piece-ac8da59f6b08

3. Sentencepiece python module example - Google Colab
   https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb

   
CMD Interface:
   --input (comma separated list of input sentences)  type: std::string default: ""
   --input_format (Input format. Supported format is `text` or `tsv`.)  type: std::string default: ""
   --sentencepiece_path (output model prefix)  type: std::string default: ""
   --model_type (model algorithm: unigram, bpe, word or char)  type: std::string default: "unigram"
   --vocab_size (vocabulary size)  type: int32 default: 8000
   --accept_language (comma-separated list of languages this model can accept)  type: std::string default: ""
   --self_test_sample_size (the size of self test samples)  type: int32 default: 0
   --character_coverage (character coverage to determine the minimum symbols)  type: double default: 0.9995
   --input_sentence_size (maximum size of sentences the trainer loads)  type: std::uint64_t default: 0
   --shuffle_input_sentence (Randomly sample input sentences in advance. Valid when --input_sentence_size > 0)  type: bool default: true
   --seed_sentencepiece_size (the size of seed sentencepieces)  type: int32 default: 1000000
   --shrinking_factor (Keeps top shrinking_factor pieces with respect to the loss)  type: double default: 0.75
   --num_threads (number of threads for training)  type: int32 default: 16
   --num_sub_iterations (number of EM sub-iterations)  type: int32 default: 2
   --max_sentencepiece_length (maximum length of sentence piece)  type: int32 default: 16
   --max_sentence_length (maximum length of sentence in byte)  type: int32 default: 4192
   --split_by_unicode_script (use Unicode script to split sentence pieces)  type: bool default: true
   --split_by_number (split tokens by numbers (0-9))  type: bool default: true
   --split_by_whitespace (use a white space to split sentence pieces)  type: bool default: true
   --split_digits (split all digits (0-9) into separate pieces)  type: bool default: false
   --treat_whitespace_as_suffix (treat whitespace marker as suffix instead of prefix.)  type: bool default: false
   --allow_whitespace_only_pieces (allow pieces that only contain (consecutive) whitespace tokens)  type: bool default: false
   --control_symbols (comma separated list of control symbols)  type: std::string default: ""
   --control_symbols_file (load control_symbols from file.)  type: std::string default: ""
   --user_defined_symbols (comma separated list of user defined symbols)  type: std::string default: ""
   --user_defined_symbols_file (load user_defined_symbols from file.)  type: std::string default: ""
   --required_chars (UTF8 characters in this flag are always used in the character set regardless of --character_coverage)  type: std::string default: ""
   --required_chars_file (load required_chars from file.)  type: std::string default: ""
   --byte_fallback (decompose unknown pieces into UTF-8 byte pieces)  type: bool default: false
   --vocabulary_output_piece_score (Define score in vocab file)  type: bool default: true
   --normalization_rule_name (Normalization rule name. Choose from nfkc or identity)  type: std::string default: "nmt_nfkc"
   --normalization_rule_tsv (Normalization rule TSV file. )  type: std::string default: ""
   --denormalization_rule_tsv (Denormalization rule TSV file.)  type: std::string default: ""
   --add_dummy_prefix (Add dummy whitespace at the beginning of text)  type: bool default: true
   --remove_extra_whitespaces (Removes leading, trailing, and duplicate internal whitespace)  type: bool default: true
   --hard_vocab_limit (If set to false, --vocab_size is considered as a soft limit.)  type: bool default: true
   --use_all_vocab (If set to true, use all tokens as vocab. Valid for word/char models.)  type: bool default: false
   --unk_id (Override UNK (<unk>) id.)  type: int32 default: 0
   --bos_id (Override BOS (<s>) id. Set -1 to disable BOS.)  type: int32 default: 1
   --eos_id (Override EOS (</s>) id. Set -1 to disable EOS.)  type: int32 default: 2
   --pad_id (Override PAD (<pad>) id. Set -1 to disable PAD.)  type: int32 default: -1
   --unk_piece (Override UNK (<unk>) piece.)  type: std::string default: "<unk>"
   --bos_piece (Override BOS (<s>) piece.)  type: std::string default: "<s>"
   --eos_piece (Override EOS (</s>) piece.)  type: std::string default: "</s>"
   --pad_piece (Override PAD (<pad>) piece.)  type: std::string default: "<pad>"
   --unk_surface (Dummy surface string for <unk>. In decoding <unk> is decoded to `unk_surface`.)  type: std::string default: " ⁇ "
   --train_extremely_large_corpus (Increase bit depth for unigram tokenization.)  type: bool default: false
   --random_seed (Seed value for random generator.)  type: uint32 default: 4294967295
   --enable_differential_privacy (Whether to add DP while training. Currently supported only by UNIGRAM model.)  type: bool default: false
   --differential_privacy_noise_level (Amount of noise to add for DP)  type: float default: 0
   --differential_privacy_clipping_threshold (Threshold for clipping the counts for DP)  type: std::uint64_t default: 0
   --help (show help)  type: bool default: false
   --version (show version)  type: bool default: false
   --minloglevel (Messages logged at a lower level than this don't actually get logged anywhere)  type: int default: 0
"""
