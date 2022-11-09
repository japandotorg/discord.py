from typing import Literal, Union

ThreadType = Literal[10, 11, 12]
ChannelTypeWithoutThread = Literal[0, 1, 2, 3, 4, 5, 6, 13, 15]

ChannelType = Union[ChannelTypeWithoutThread, ThreadType]
