# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from typing import Optional, Union

from cryptography.fernet import Fernet, InvalidToken

from datumaro.components.errors import DatumaroError
from datumaro.util.deprecation import deprecated


@deprecated(deprecated_version="1.11", removed_version="1.12")
class Crypter:
    # Prefix (datum-) = 6 and Fernet = 44, 6 + 44 = 50
    FERNET_KEY_LEN = 50
    KEY_PREFIX = b"datum-"
    KEY_PREFIX_LEN = len(KEY_PREFIX)

    def __init__(self, key: Union[str, bytes]) -> None:
        if isinstance(key, str):
            key = key.encode()
        if len(key) != self.FERNET_KEY_LEN:
            raise DatumaroError(
                f"Key length should be {self.FERNET_KEY_LEN}, "
                f"but your key length is {len(key)} (key={key})."
            )
        self._key = key[self.KEY_PREFIX_LEN :]
        self._fernet = Fernet(self._key)

    @property
    def key(self) -> bytes:
        return self.KEY_PREFIX + self._key

    def decrypt(self, msg: bytes) -> bytes:
        return self._fernet.decrypt(msg)

    def encrypt(self, msg: bytes) -> bytes:
        return self._fernet.encrypt(msg)

    def handshake(self, key: bytes) -> bool:
        try:
            return self.decrypt(key) == self.key
        except InvalidToken as e:
            log.debug(e)
            return False

    @classmethod
    def gen_key(cls, key: Optional[bytes] = None) -> bytes:
        """If "key" is not None, return the different key with "key"."""
        _key = cls.KEY_PREFIX + Fernet.generate_key()
        while _key == key:
            _key = cls.KEY_PREFIX + Fernet.generate_key()
        return _key

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Crypter):
            return False

        return self._key == __o._key

    @property
    def is_null_crypter(self):
        return self == NULL_CRYPTER


class NullCrypter(Crypter):
    def __init__(self) -> None:
        self._key = None
        self._fernet = None

    def decrypt(self, msg: bytes) -> bytes:
        return msg

    def encrypt(self, msg: bytes) -> bytes:
        return msg

    def handshake(self, key: bytes) -> bool:
        return self.decrypt(key) == b""

    @property
    def key(self) -> None:
        return None


NULL_CRYPTER = NullCrypter()
