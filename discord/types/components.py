from __future__ import annotations

from typing import List, Literal, TypedDict, Union

from typing_extensions import NotRequired

from .channel import ChannelType
from .emoji import PartialEmoji

ComponentType = Literal[1, 2, 3, 4]
ButtonStyle = Literal[1, 2, 3, 4, 5]
TextStyle = Literal[1, 2]


class ActionRow(TypedDict):
    type: Literal[1]
    components: List[ActionRowChildComponent]


class ButtonComponent(TypedDict):
    type: Literal[2]
    style: ButtonStyle
    custom_id: NotRequired[str]
    url: NotRequired[str]
    disabled: NotRequired[bool]
    emoji: NotRequired[PartialEmoji]
    label: NotRequired[str]


class SelectOption(TypedDict):
    label: str
    value: str
    default: bool
    description: NotRequired[str]
    emoji: NotRequired[PartialEmoji]


class SelectComponent(TypedDict):
    custom_id: str
    placeholder: NotRequired[str]
    min_values: NotRequired[int]
    max_values: NotRequired[int]
    disabled: NotRequired[bool]


class StringSelectComponent(SelectComponent):
    type: Literal[3]
    options: NotRequired[List[SelectOption]]


class UserSelectComponent(SelectComponent):
    type: Literal[5]


class RoleSelectComponent(SelectComponent):
    type: Literal[6]


class MentionableSelectComponent(SelectComponent):
    type: Literal[7]


class ChannelSelectComponent(SelectComponent):
    type: Literal[8]
    channel_types: NotRequired[List[ChannelType]]


class TextInput(TypedDict):
    type: Literal[4]
    custom_id: str
    style: TextStyle
    label: str
    placeholder: NotRequired[str]
    value: NotRequired[str]
    required: NotRequired[bool]
    min_length: NotRequired[int]
    max_length: NotRequired[int]


class SelectMenu(SelectComponent):
    type: Literal[3, 5, 6, 7, 8]
    options: NotRequired[List[SelectOption]]
    channel_types: NotRequired[List[ChannelType]]


ActionRowChildComponent = Union[ButtonComponent, SelectMenu, TextInput]
Component = Union[ActionRow, ActionRowChildComponent]
