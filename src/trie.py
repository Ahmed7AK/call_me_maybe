from functools import lru_cache

from .vocab import get_vocab_bytes


# One node per byte position, token_id is only set when a path ends here. So interior nodes stay None
class TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children: dict[int, "TrieNode"] = {}
        self.token_id: int | None = None


# A Trie is a prefix index over the vocab. It allows us to see what tokens are reachable from what bytes 
class Trie:
    def __init__(self, tokens):
        self.root = TrieNode()
        for token_id, data in enumerate(tokens):
            self.insert(data, token_id)

    # Reads through the token's bytes from the root creating a node per byte where one is missing. Then adds the token_id on the final node
    def insert(self, data, token_id):
        node = self.root
        for byte in data:
            child = node.children.get(byte)
            if child is None:
                child = TrieNode()
                node.children[byte] = child
            node = child
        node.token_id = token_id

    # Advances by one node
    def step(self, node, byte):
        return node.children.get(byte)

    # Looks for and finds a node
    def find(self, data):
        node = self.root
        for byte in data:
            node = node.children.get(byte)
            if node is None:
                return None
        return node

    # Returns every token that is a prefix of the target 
    def prefixes_of(self, target):
        node = self.root
        for length, byte in enumerate(target, start=1):
            node = node.children.get(byte)
            if node is None:
                return
            if node.token_id is not None:
                yield node.token_id, length


@lru_cache(maxsize=1)
def get_trie():
    return Trie(get_vocab_bytes())
