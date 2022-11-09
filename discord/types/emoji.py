from typing import Optional, TypedDict


class PartialEmoji(TypedDict):
    id: Optional[Snowflake]
    name: Optional[str]


class Emoji(PartialEmoji, total=False):
    roles: SnowflakeList
    user: User
    require_colons: bool
    managed: bool
    animated: bool
    available: bool


class EditEmoji(TypedDict):
    name: str
    roles: Optional[SnowflakeList]
