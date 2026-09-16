from collections.abc import Iterator, Sequence
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field

from .vocab import get_vocab_bytes


class TrieNode(BaseModel):
    """
    One node per byte position, token_id is only set
    when a path ends here. So interior nodes stay None
    """
    model_config = ConfigDict(extra="forbid")

    children: dict[int, "TrieNode"] = Field(default_factory=dict)
    token_id: int | None = None


class Trie(BaseModel):
    """
    A Trie is a prefix index over the vocab. It allows us to see what
    tokens are reachable from what bytes. Instead of having to traverse
    the entire 150k vocab, we only have to traverse nodes that lead to
    the target.
    """
    model_config = ConfigDict(extra="forbid")

    root: TrieNode = Field(default_factory=TrieNode)

    @classmethod
    def from_tokens(cls, tokens: Sequence[bytes]) -> "Trie":
        """Builds a Trie holding every token, keyed by its position"""
        trie = cls()
        for token_id, data in enumerate(tokens):
            trie.insert(data, token_id)
        return trie

    def insert(self, data: bytes, token_id: int) -> None:
        """Reads through the token's bytes from the root
        creating a node per byte where one is missing.
        Then adds the token_id on the final node.
        """
        node = self.root
        for byte in data:
            child = node.children.get(byte)
            if child is None:
                child = TrieNode()
                node.children[byte] = child
            node = child
        node.token_id = token_id

    def step(self, node: TrieNode, byte: int) -> TrieNode | None:
        """Advances by one node"""
        return node.children.get(byte)

    def find(self, data: bytes) -> TrieNode | None:
        """Looks for and finds a node"""
        node: TrieNode | None = self.root
        for byte in data:
            if node is None:
                return None
            node = node.children.get(byte)
        return node

    def prefixes_of(self, target: bytes) -> Iterator[tuple[int, int]]:
        """Returns every token that is a prefix of the target"""
        node = self.root
        for length, byte in enumerate(target, start=1):
            child = node.children.get(byte)
            if child is None:
                return
            node = child
            if node.token_id is not None:
                yield node.token_id, length


@lru_cache(maxsize=1)
def get_trie() -> Trie:
    """Returns a cached Trie"""
    return Trie.from_tokens(get_vocab_bytes())
