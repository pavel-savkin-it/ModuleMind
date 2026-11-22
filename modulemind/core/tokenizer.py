"""Посимвольный токенизатор: минимальная зависимость для крошечного ядра."""


class CharTokenizer:
    """Текст <-> список id. Неизвестные символы -> UNK («?» при декодировании)."""

    def __init__(self, alphabet: str, unk: str = "?"):
        if unk in alphabet:
            raise ValueError(f"символ UNK «{unk}» не должен входить в алфавит")
        self.unk = unk
        self.chars = sorted(set(alphabet))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.unk_id = len(self.chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 1  # +1 на UNK

    def encode(self, text: str) -> list:
        return [self.stoi.get(c, self.unk_id) for c in text]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            out.append(self.itos[i] if i in self.itos else self.unk)
        return "".join(out)
