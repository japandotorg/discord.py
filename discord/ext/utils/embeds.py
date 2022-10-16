import re
import datetime
from typing import (
    MutableMapping,
    TypedDict,
    Sequence,
    Optional,
    Union,
    Any,
)

import discord

from . import regex

EMBED_TOP_LEVEL_ATTRIBUTES_MASK_DICT = {
    "provider": None,
    "type": None,
    "title": None,
    "description": None,
    "url": None,
    "color": None,
    "timestamp": None,
    "footer": None,
    "thumbnail": None,
    "image": None,
    "author": None,
    "fields": None,
}

EMBED_TOP_LEVEL_ATTRIBUTES_SET = {
    "provider",
    "type",
    "title",
    "description",
    "url",
    "color",
    "timestamp",
    "footer",
    "thumbnail",
    "image",
    "author",
    "fields",
}

EMBED_SYSTEM_ATTRIBUTES_MASK_DICT = {
    "provider": {
        "name": None,
        "url": None,
    },
    "type": None,
    "footer": {
        "proxy_icon_url": None,
    },
    "thumbnail": {
        "proxy_url": None,
        "width": None,
        "height": None,
    },
    "image": {
        "proxy_url": None,
        "width": None,
        "height": None,
    },
    "author": {
        "proxy_icon_url": None,
    },
}

EMBED_SYSTEM_ATTRIBUTES = {
    "provider",
    "proxy_url",
    "proxy_icon_url",
    "width",
    "height",
    "type",
}

EMBED_NON_SYSTEM_ATTRIBUTES = {
    "name",
    "value",
    "inline",
    "url",
    "image",
    "thumbnail",
    "title",
    "description",
    "color",
    "timestamp",
    "footer",
    "text",
    "icon_url",
    "author",
    "fields",
}

EMBED_ATTRIBUTES_SET = {
    "provider",
    "name",
    "value",
    "inline",
    "url",
    "image",
    "thumbnail",
    "proxy_url",
    "type",
    "title",
    "description",
    "color",
    "timestamp",
    "footer",
    "text",
    "icon_url",
    "proxy_icon_url",
    "author",
    "fields",
}

EMBED_ATTRIBUTES_WITH_SUB_ATTRIBUTES_SET = {
    "author",
    "thumbnail",
    "image",
    "fields",
    "footer",
    "provider",
}

EMBED_TOTAL_CHAR_LIMIT = 6000

EMBED_FIELDS_LIMIT = 25

EMBED_CHAR_LIMITS = {
    "author.name": 256,
    "title": 256,
    "description": 4096,
    "fields": 25,
    "field.name": 256,
    "field.value": 1024,
    "footer.text": 2048,
}


class FlattenedEmbedDict(TypedDict):
    author_name: Optional[str]
    author_url: Optional[str]
    author_icon_url: Optional[str]
    title: Optional[str]
    url: Optional[str]
    thumbnail_url: Optional[str]
    description: Optional[str]
    image_url: Optional[str]
    color: int
    fields: Optional[Sequence[dict[str, Union[str, bool]]]]
    footer_text: Optional[str]
    footer_icon_url: Optional[str]
    timestamp: Optional[Union[str, datetime.datetime]]


EMBED_MASK_DICT_HINT = dict[
    str,
    Union[
        str,
        int,
        dict[str, Union[str, bool]],
        list[dict[str, Union[str, bool]]],
        datetime.datetime,
    ],
]


def copy_embed_dict(embed_dict: MutableMapping[str, Any]) -> dict:
    """
    Make a shallow copy of the given embed dictionary, and embed fields (if present).

    Args:
        embed_dict (dict): The target embed dictionary.

    Returns:
        dict: The copy.
    """

    copied_embed_dict = {
        k: v.copy() if isinstance(v, dict) else v for k, v in embed_dict.items()
    }

    if "fields" in embed_dict:
        copied_embed_dict["fields"] = [
            dict(field_dict) for field_dict in embed_dict["fields"]
        ]

    return copied_embed_dict


def create_embed_mask_dit(
    attributes: str = "",
    allow_system_attributes: bool = False,
    fields_as_field_dict: bool = False,
) -> EMBED_MASK_DICT_HINT:
    """
    Create an embed mask dictionary based on the given attributes in the given string.
    This is mostly used for interal purposes relating to comparing and modifying embed dictionaries.
    All embed attributes are set to `None` by default. Which will be ignored by `discord.Embed`.

    Args:
        attributes (str, optional): The attribute string. Defaults to "", which will return
            all valid attributes of an embed.
        allow_system_attributes (bool, optional): Whether to include embed attributes that
            can not be manually set by bot users. Defaults to false.
        fields_as_field_dict (bool, optional): Whether the embed `fields` attribute returned
            in the output dictionary of this function should be a dictionary that maps stringized indices
            to embed field dictionaries. Defaults to False.

    Raises:
        ValueError: Invalid embed attribute string.

    Returns:
        dict: The egenrated embed with the specified attributes set to None.
    """

    embed_top_level_attrib_dict = EMBED_TOP_LEVEL_ATTRIBUTES_MASK_DICT
    embed_top_level_attrib_dict = {
        k: v.copy() if isinstance(v, dict) else v
        for k, v in embed_top_level_attrib_dict.items()
    }

    system_attribs_dict = EMBED_SYSTEM_ATTRIBUTES_MASK_DICT
    system_attribs_dict = {
        k: v.copy() if isinstance(v, dict) else v
        for k, v in system_attribs_dict.items()
    }

    all_system_attribs_set = EMBED_SYSTEM_ATTRIBUTES

    embed_mask_dict = {}

    attribs = attributes

    attribs_tuple = tuple(
        attr_str.split(sep=".") if "." in attr_str else attr_str
        for attr_str in attribs.split()
    )

    all_attribs_set = EMBED_ATTRIBUTES_SET | set(str(i) for i in range(25))
    attribs_with_sub_attribs = EMBED_ATTRIBUTES_WITH_SUB_ATTRIBUTES_SET

    for attr in attribs_tuple:
        if isinstance(attr, list):
            if len(attr) > 3:
                raise ValueError(
                    "Invalid embed attribute filter string! "
                    "Sub-attributes do not propagate beyond 3 levels."
                )
            bottom_dict = {}
            for i in range(len(attr)):
                if attr[i] not in all_attribs_set:
                    if i == 1:
                        if (
                            attr[i - 1] == "fields"
                            and "(" not in attr[i]
                            and ")" not in attr[i]
                        ):
                            raise ValueError(
                                f"`{attr[i]}` is not a valid embed (sub-)attribute "
                                "name!"
                            )
                    else:
                        raise ValueError(
                            f"`{attr[i]}` is not a valid embed (sub-)attribute name!"
                        )

                elif attr[i] in all_system_attribs_set and not allow_system_attributes:
                    raise ValueError(
                        f"The given attribute `{attr[i]}` cannot be retrieved when "
                        "`system_attributes=` is set to `False`."
                    )
                if not i:
                    if attribs_tuple.count(attr[i]):
                        raise ValueError(
                            "Invalid embed attribute filter string! "
                            f"Top level embed attribute `{attr[i]}` conflicts with "
                            "its preceding instances."
                        )
                    elif attr[i] not in attribs_with_sub_attribs:
                        raise ValueError(
                            "Invalid embed attribute filter string! "
                            f"The embed attribute `{attr[i]}` does not have any "
                            "sub-attributes!"
                        )

                    if attr[i] not in embed_mask_dict:
                        embed_mask_dict[attr[i]] = bottom_dict
                    else:
                        bottom_dict = embed_mask_dict[attr[i]]

                elif i == 1 and attr[i - 1] == "fields" and not attr[i].isnumeric():
                    if attr[i].startswith("(") and attr[i].endswith(")"):
                        if not attr[i].startswith("(") and not attr[i].endswith(")"):
                            raise ValueError(
                                "Invalid embed attribute filter string! "
                                "Embed field ranges should only contain integers "
                                "and should be structured like this: "
                                "`fields.(start, stop[, step]).attribute`"
                            )
                        field_str_range_list = [v for v in attr[i][1:][:-1].split(",")]
                        field_range_list = []

                        for j in range(len(field_str_range_list)):
                            if (
                                field_str_range_list[j].isnumeric()
                                or len(field_str_range_list[j]) > 1
                                and field_str_range_list[j][1:].isnumeric()
                            ):
                                field_range_list.append(int(field_str_range_list[j]))
                            else:
                                raise ValueError(
                                    "Invalid embed attribute filter string! "
                                    "Embed field ranges should only contain integers "
                                    "and should be structured like this: "
                                    "`fields.(start, stop[, step]).attribute`"
                                )

                        sub_attrs = []
                        if attr[i] == attr[-1]:
                            sub_attrs.extend(("name", "value", "inline"))

                        elif attr[-1] in ("name", "value", "inline"):
                            sub_attrs.append(attr[-1])

                        else:
                            raise ValueError(
                                f"`{attr[-1]}` is not a valid embed (sub-)attribute name!",
                            )

                        field_range = range(*field_range_list)
                        if not field_range:
                            raise ValueError(
                                "Invalid embed attribute filter string! "
                                "Empty field range!"
                            )
                        for j in range(*field_range_list):
                            str_idx = str(j)
                            if str_idx not in embed_mask_dict["fields"]:
                                embed_mask_dict["fields"][str_idx] = {
                                    sub_attr: None for sub_attr in sub_attrs
                                }
                            else:
                                for sub_attr in sub_attrs:
                                    embed_mask_dict["fields"][str_idx][sub_attr] = None

                        break

                    elif attr[i] in ("name", "value", "inline"):
                        for sub_attr in ("name", "value", "inline"):
                            if attr[i] == sub_attr:
                                for j in range(25):
                                    str_idx = str(j)
                                    if str_idx not in embed_mask_dict["fields"]:
                                        embed_mask_dict["fields"][str_idx] = {
                                            sub_attr: None
                                        }
                                    else:
                                        embed_mask_dict["fields"][str_idx][
                                            sub_attr
                                        ] = None
                                break
                        else:
                            raise ValueError(
                                "Invalid embed attribute filter string! "
                                f"The given attribute `{attr[i]}` is not an "
                                "attribute of an embed field!"
                            )
                        break
                    else:
                        raise ValueError(
                            "Invalid embed attribute filter string! "
                            "Embed field attibutes must be either structutred like"
                            "`fields.0`, `fields.0.attribute`, `fields.attribute` or "
                            "`fields.(start,stop[,step]).attribute`. Note that embed "
                            "field ranges cannot contain whitespace."
                        )

                elif i == len(attr) - 1:
                    if attr[i] not in bottom_dict:
                        bottom_dict[attr[i]] = None
                else:
                    if attr[i] not in embed_mask_dict[attr[i - 1]]:
                        bottom_dict = {}
                        embed_mask_dict[attr[i - 1]][attr[i]] = bottom_dict
                    else:
                        bottom_dict = embed_mask_dict[attr[i - 1]][attr[i]]

        elif attr in embed_top_level_attrib_dict:
            if attribs_tuple.count(attr) > 1:
                raise ValueError(
                    "Invalid embed attribute filter string! "
                    "Do not specify top level embed attributes "
                    f"twice when not using the `.` operator: `{attr}`",
                )
            elif attr in all_system_attribs_set and not allow_system_attributes:
                raise ValueError(
                    f"The given attribute `{attr}` cannot be retrieved when "
                    "`system_attributes=` is set to `False`.",
                )

            if attr not in embed_mask_dict:
                embed_mask_dict[attr] = None
            else:
                raise ValueError(
                    "Invalid embed attribute filter string! "
                    "Do not specify upper level embed attributes twice!",
                )

        else:
            raise ValueError(
                f"Invalid top level embed attribute name `{attr}`!",
            )

    if not fields_as_field_dict and "fields" in embed_mask_dict:
        embed_mask_dict["fields"] = [
            embed_mask_dict["fields"][i]
            for i in sorted(embed_mask_dict["fields"].keys())
        ]

    return embed_mask_dict


def split_embed_dict(
    embed_dict: MutableMapping[str, Any], divide_code_blocks: bool = True
):
    """
    Split an embed dictionary into multiple valid embed dictionaries based on embed text
    attribute character limits and the total character limit of a single embed in a single
    message. This function will not correct invalid embed attributes or add any missing required
    ones.

    Args:
        embed_dict (dict): The target embed.
        divide_code_blocks (bool, optional): Whether to divide code blocks into two valid ones,
            if they contain a division point of an embed text attribute.
            Defaults to True.

    Returns:
        list[dict]: A list of newly generated embed dictionaries.
    """

    embed_dict = copy_embed_dict(embed_dict)
    embed_dicts = [embed_dict]
    updated = True

    while updated:
        updated = False
        for i in range(len(embed_dicts)):
            embed_dict = embed_dicts[i]
            if "author" in embed_dict and "name" in embed_dict["author"]:
                author_name = embed_dict["author"]["name"]
                if len(author_name) > EMBED_CHAR_LIMITS["author.name"]:
                    if "title" not in embed_dict:
                        embed_dict["title"] = ""

                    normal_split = True

                    if (
                        (
                            url_matches := tuple(
                                re.finditer(regex.URL, author_name)
                            )
                        )
                        and (url_match := url_matches[-1]).start()
                        < EMBED_CHAR_LIMITS["author.name"] - 1
                        and url_match.end() > EMBED_CHAR_LIMITS["author.name"]
                    ):
                        if (
                            ((match_span := url_match.span())[1] - match_span[0] + 1)
                        ) <= EMBED_CHAR_LIMITS[
                            "title"
                        ]:
                            embed_dict["author"]["name"] = author_name[
                                : url_match.start()
                            ]
                            embed_dict["title"] = (
                                author_name[url_match.start() :]
                                + f' {embed_dict["title"]}'
                            ).strip()

                            normal_split = False

                    if normal_split:
                        embed_dict["author"]["name"] = author_name[
                            : EMBED_CHAR_LIMITS["author.name"] - 1
                        ]
                        embed_dict["title"] = (
                            author_name[EMBED_CHAR_LIMITS["author.name"] - 1 :]
                            + f' {embed_dict["title"]}'
                        ).strip()

                    if not embed_dict["title"]:
                        del embed_dict["title"]

                    updated = True

            if "title" in embed_dict:
                title = embed_dict["title"]
                if len(title) > EMBED_CHAR_LIMITS["title"]:
                    if "description" not in embed_dict:
                        embed_dict["description"] = ""

                    normal_split = True

                    if (
                        (
                            inline_code_matches := tuple(
                                re.finditer(regex.INLINE_CODE_BLOCK, title)
                            )
                        )
                        and (inline_code_match := inline_code_matches[-1]).start()
                        < EMBED_CHAR_LIMITS["title"] - 1
                        and inline_code_match.end() > EMBED_CHAR_LIMITS["title"] - 1
                    ):

                        if divide_code_blocks:
                            embed_dict[
                                "title"
                            ] = f'{title[: EMBED_CHAR_LIMITS["title"] - 1]}`'

                            embed_dict["description"] = (
                                f'`{title[EMBED_CHAR_LIMITS["title"] - 1 :]}'
                                f' {embed_dict["description"]}'
                            ).strip()
                            normal_split = False
                        elif (
                            (
                                (match_span := inline_code_match.span())[1]
                                - match_span[0]
                                + 1
                            )
                        ) <= EMBED_CHAR_LIMITS[
                            "description"
                        ]:
                            embed_dict["title"] = title[: inline_code_match.start()]
                            embed_dict["description"] = (
                                title[inline_code_match.start() :]
                                + f' {embed_dict["description"]}'
                            ).strip()

                            normal_split = False

                    elif (
                        (url_matches := tuple(re.finditer(regex.URL, title)))
                        and (url_match := url_matches[-1]).start()
                        < EMBED_CHAR_LIMITS["title"] - 1
                        and url_match.end() > EMBED_CHAR_LIMITS["title"]
                    ):
                        if (
                            ((match_span := url_match.span())[1] - match_span[0] + 1)
                        ) <= EMBED_CHAR_LIMITS[
                            "description"
                        ]:
                            embed_dict["title"] = title[: url_match.start()]
                            embed_dict["description"] = (
                                title[url_match.start() :]
                                + f' {embed_dict["description"]}'
                            ).strip()
                            normal_split = False

                    if normal_split:
                        embed_dict["title"] = title[: EMBED_CHAR_LIMITS["title"] - 1]
                        embed_dict["description"] = (
                            title[EMBED_CHAR_LIMITS["title"] - 1 :]
                            + f' {embed_dict["description"]}'
                        )

                    if not embed_dict["description"]:
                        del embed_dict["description"]

                    updated = True

            if "description" in embed_dict:
                description = embed_dict["description"]
                if len(description) > EMBED_CHAR_LIMITS["description"]:
                    next_embed_dict = {
                        attr: embed_dict.pop(attr)
                        for attr in ("color", "fields", "image", "footer")
                        if attr in embed_dict
                    }
                    next_embed_dict["description"] = ""
                    if "color" in next_embed_dict:
                        embed_dict["color"] = next_embed_dict["color"]

                    normal_split = True

                    if (
                        (
                            code_matches := tuple(
                                re.finditer(regex.CODE_BLOCK, description)
                            )
                        )
                        and (code_match := code_matches[-1]).start()
                        < EMBED_CHAR_LIMITS["description"] - 1
                        and code_match.end() > EMBED_CHAR_LIMITS["description"] - 1
                    ):

                        if (
                            divide_code_blocks
                            and code_match.start() + code_match.group().find("\n")
                            < EMBED_CHAR_LIMITS["description"] - 1
                        ):
                            embed_dict[
                                "description"
                            ] = f'{description[: EMBED_CHAR_LIMITS["description"] - 3]}```'

                            next_embed_dict["description"] = (
                                f'```{code_match.group(1)}\n{description[EMBED_CHAR_LIMITS["description"] - 1 :]}'  # group 1 is the code language
                                + f' {next_embed_dict["description"]}'
                            )
                            normal_split = False
                        elif (
                            ((match_span := code_match.span())[1] - match_span[0] + 1)
                        ) <= EMBED_CHAR_LIMITS["description"]:
                            embed_dict["description"] = description[
                                : code_match.start()
                            ]
                            next_embed_dict["description"] = (
                                description[code_match.start() :]
                                + f' {next_embed_dict["description"]}'
                            )
                            normal_split = False

                    elif (
                        (
                            inline_code_matches := tuple(
                                re.finditer(regex.CODE_BLOCK, description)
                            )
                        )
                        and (inline_code_match := inline_code_matches[-1]).start()
                        < EMBED_CHAR_LIMITS["description"] - 1
                        and inline_code_match.end()
                        > EMBED_CHAR_LIMITS["description"] - 1
                    ):

                        if divide_code_blocks:
                            embed_dict[
                                "description"
                            ] = f'{description[: EMBED_CHAR_LIMITS["description"] - 1]}`'

                            next_embed_dict["description"] = (
                                f'`{description[EMBED_CHAR_LIMITS["description"] - 1 :]}'
                                + f' {next_embed_dict["description"]}'
                            ).strip()
                            normal_split = False
                        elif (
                            (
                                (match_span := inline_code_match.span())[1]
                                - match_span[0]
                                + 1
                            )
                        ) <= EMBED_CHAR_LIMITS[
                            "description"
                        ]:
                            embed_dict["description"] = description[
                                : inline_code_match.start()
                            ]
                            next_embed_dict["description"] = (
                                description[inline_code_match.start() :]
                                + f' {next_embed_dict["description"]}'
                            ).strip()
                            normal_split = False

                    elif (
                        (
                            url_matches := tuple(
                                re.finditer(regex.URL, description)
                            )
                        )
                        and (url_match := url_matches[-1]).start()
                        < EMBED_CHAR_LIMITS["description"] - 1
                        and url_match.end() > EMBED_CHAR_LIMITS["description"] - 1
                    ):
                        if (
                            ((match_span := url_match.span())[1] - match_span[0] + 1)
                        ) <= EMBED_CHAR_LIMITS[
                            "description"
                        ]:

                            embed_dict["description"] = description[: url_match.start()]
                            next_embed_dict["description"] = (
                                description[url_match.start() :]
                                + f' {next_embed_dict["description"]}'
                            ).strip()

                            normal_split = False

                    if normal_split:
                        embed_dict["description"] = description[
                            : EMBED_CHAR_LIMITS["description"] - 1
                        ]
                        next_embed_dict["description"] = (
                            description[EMBED_CHAR_LIMITS["description"] - 1 :]
                            + f' {next_embed_dict["description"]}'
                        ).strip()

                    if not next_embed_dict["description"]:
                        del next_embed_dict["description"]

                    if next_embed_dict and not (
                        len(next_embed_dict) == 1 and "color" in next_embed_dict
                    ):
                        embed_dicts.insert(i + 1, next_embed_dict)

                    updated = True

            current_len = (
                len(embed_dict.get("author", {}).get("name", ""))
                + len(embed_dict.get("title", ""))
                + len(embed_dict.get("description", ""))
            )

            if "fields" in embed_dict:
                fields = embed_dict["fields"]
                for j in range(len(fields)):
                    field = fields[j]
                    if "name" in field:
                        field_name = field["name"]
                        if len(field_name) > EMBED_CHAR_LIMITS["field.name"]:
                            if "value" not in field:
                                field["value"] = ""

                            normal_split = True

                            if (
                                inline_code_matches := tuple(
                                    re.finditer(
                                        regex.INLINE_CODE_BLOCK, field_name
                                    )
                                )
                            ) and (
                                inline_code_match := inline_code_matches[-1]
                            ).end() > EMBED_CHAR_LIMITS[
                                "field.name"
                            ] - 1:

                                if divide_code_blocks:
                                    field[
                                        "name"
                                    ] = f'{field_name[: EMBED_CHAR_LIMITS["field.name"] - 1]}`'

                                    field["value"] = (
                                        f'`{field_name[EMBED_CHAR_LIMITS["field.name"] - 1 :]}'
                                        f' {field["value"]}'
                                    ).strip()
                                elif (
                                    (
                                        (match_span := inline_code_match.span())[1]
                                        - match_span[0]
                                        + 1
                                    )
                                ) <= EMBED_CHAR_LIMITS[
                                    "field.value"
                                ]:
                                    field["name"] = field_name[
                                        : inline_code_match.start()
                                    ]
                                    field["value"] = (
                                        field_name[inline_code_match.start() :]
                                        + f' {field["value"]}'
                                    ).strip()
                                    normal_split = False

                            elif (
                                (
                                    url_matches := tuple(
                                        re.finditer(regex.URL, field_name)
                                    )
                                )
                                and (url_match := url_matches[-1]).start()
                                < EMBED_CHAR_LIMITS["field.name"] - 1
                                and url_match.end() > EMBED_CHAR_LIMITS["field.name"]
                            ):
                                if (
                                    (
                                        (match_span := url_match.span())[1]
                                        - match_span[0]
                                        + 1
                                    )
                                ) <= EMBED_CHAR_LIMITS[
                                    "field.name"
                                ]:
                                    field["name"] = field_name[: url_match.start()]
                                    field["value"] = (
                                        field_name[url_match.start() :]
                                        + f' {field["value"]}'
                                    ).strip()

                                    normal_split = False

                            if normal_split:
                                field["name"] = field_name[
                                    : EMBED_CHAR_LIMITS["field.name"] - 1
                                ]
                                field["value"] = (
                                    field_name[EMBED_CHAR_LIMITS["field.name"] - 1 :]
                                    + f' {field["value"]}'
                                ).strip()

                            if not field["value"]:
                                del field["value"]

                            updated = True

                    if "value" in field:
                        field_value = field["value"]
                        if len(field_value) > EMBED_CHAR_LIMITS["field.value"]:
                            next_field = {}
                            next_field["name"] = "\u200b"

                            if "inline" in field:
                                next_field["inline"] = field["inline"]

                            normal_split = True

                            if (
                                (
                                    code_matches := tuple(
                                        re.finditer(
                                            regex.CODE_BLOCK, field_value
                                        )
                                    )
                                )
                                and (code_match := code_matches[-1]).start()
                                < EMBED_CHAR_LIMITS["field.value"] - 1
                                and code_match.end()
                                > EMBED_CHAR_LIMITS["field.value"] - 1
                            ):

                                if (
                                    divide_code_blocks
                                    and code_match.start()
                                    + code_match.group().find("\n")
                                    < EMBED_CHAR_LIMITS["field.value"] - 1
                                ):
                                    field[
                                        "value"
                                    ] = f'{field_value[: EMBED_CHAR_LIMITS["field.value"] - 3]}```'

                                    next_field["value"] = (
                                        f'```{code_match.group(1)}\n{field_value[EMBED_CHAR_LIMITS["field.value"] - 1 :]}'  # group 1 is the code language
                                        f' {next_field["field.value"]}'
                                    ).split()
                                    normal_split = False
                                elif (
                                    (
                                        (match_span := code_match.span())[1]
                                        - match_span[0]
                                        + 1
                                    )
                                ) <= EMBED_CHAR_LIMITS["field.value"]:
                                    field["value"] = field_value[: code_match.start()]
                                    next_field["value"] = (
                                        field_value[code_match.start() :]
                                        + f' {next_field["value"]}'
                                    ).strip()
                                    normal_split = False

                            elif (
                                (
                                    inline_code_matches := tuple(
                                        re.finditer(
                                            regex.CODE_BLOCK, field_value
                                        )
                                    )
                                )
                                and (
                                    inline_code_match := inline_code_matches[-1]
                                ).start()
                                < EMBED_CHAR_LIMITS["field.value"] - 1
                                and inline_code_match.end()
                                > EMBED_CHAR_LIMITS["field.value"] - 1
                            ):

                                if divide_code_blocks:
                                    field[
                                        "value"
                                    ] = f'{field_value[: EMBED_CHAR_LIMITS["field.value"] - 1]}`'

                                    next_field["value"] = (
                                        f'`{field_value[EMBED_CHAR_LIMITS["field.value"] - 1 :]}'
                                        f' {next_field["value"]}'
                                    ).strip()
                                    normal_split = False
                                elif (
                                    (
                                        (match_span := inline_code_match.span())[1]
                                        - match_span[0]
                                        + 1
                                    )
                                ) <= EMBED_CHAR_LIMITS[
                                    "field.value"
                                ]:
                                    field["value"] = field_value[
                                        : inline_code_match.start()
                                    ]
                                    next_field["value"] = (
                                        field_value[inline_code_match.start() :]
                                        + f' {next_field["value"]}'
                                    ).strip()
                                    normal_split = False

                            elif (
                                (
                                    url_matches := tuple(
                                        re.finditer(regex.URL, field_value)
                                    )
                                )
                                and (url_match := url_matches[-1]).start()
                                < EMBED_CHAR_LIMITS["field.value"] - 1
                                and url_match.end() > EMBED_CHAR_LIMITS["field.value"]
                            ):
                                if (
                                    (
                                        (match_span := url_match.span())[1]
                                        - match_span[0]
                                        + 1
                                    )
                                ) <= EMBED_CHAR_LIMITS[
                                    "field.value"
                                ]:
                                    field["value"] = field_value[: url_match.start()]
                                    next_field["value"] = (
                                        field_value[url_match.start() :]
                                        + f' {next_field["value"]}'
                                    ).strip()

                                    normal_split = False

                            if normal_split:
                                field["value"] = field_value[
                                    : EMBED_CHAR_LIMITS["field.value"] - 1
                                ]
                                next_field["value"] = (
                                    field["value"][
                                        EMBED_CHAR_LIMITS["field.value"] - 1 :
                                    ]
                                    + f' {next_field["value"]}'
                                ).strip()

                            if not next_field["value"]:
                                del next_field["value"]

                            if next_field:
                                fields.insert(j + 1, next_field)

                            updated = True

                for j in range(len(fields)):
                    field = fields[j]
                    field_char_count = len(field.get("name", "")) + len(
                        field.get("value", "")
                    )
                    if (
                        current_len + field_char_count > EMBED_TOTAL_CHAR_LIMIT
                        or j > 24
                    ):
                        next_embed_dict = {
                            attr: embed_dict.pop(attr)
                            for attr in ("color", "image", "footer")
                            if attr in embed_dict
                        }
                        if "color" in next_embed_dict:
                            embed_dict["color"] = next_embed_dict["color"]

                        embed_dict["fields"] = fields[:j]
                        next_embed_dict["fields"] = fields[j:]
                        embed_dicts.insert(i + 1, next_embed_dict)

                        updated = True
                        break

                    current_len += field_char_count

            if "footer" in embed_dict and "text" in embed_dict["footer"]:
                footer_text = ""
                for _ in range(2):
                    footer_text = embed_dict["footer"]["text"]
                    footer_text_len = len(footer_text)
                    if (
                        footer_text_len > EMBED_CHAR_LIMITS["footer.text"]
                        or current_len + footer_text_len > EMBED_TOTAL_CHAR_LIMIT
                    ):
                        if i + 1 < len(embed_dicts):
                            next_embed_dict = embed_dicts[i + 1]
                        else:
                            next_embed_dict = {
                                "footer": {
                                    attr: embed_dict["footer"].pop(attr)
                                    for attr in ("icon_url", "proxy_icon_url")
                                    if attr in embed_dict["footer"]
                                }
                            }
                            if "color" in embed_dict:
                                next_embed_dict["color"] = embed_dict["color"]

                            embed_dicts.insert(i + 1, next_embed_dict)

                        if footer_text_len > EMBED_CHAR_LIMITS["footer.text"]:
                            split_index = EMBED_CHAR_LIMITS["footer.text"] - 1
                        else:
                            split_index = (
                                footer_text_len
                                - (
                                    current_len
                                    + footer_text_len
                                    - EMBED_TOTAL_CHAR_LIMIT
                                )
                                - 1
                            )

                        normal_split = True

                        if (
                            (
                                url_matches := tuple(
                                    re.finditer(regex.URL, footer_text)
                                )
                            )
                            and (url_match := url_matches[-1]).start() < split_index
                            and url_match.end() > split_index
                        ):
                            if (
                                (
                                    (match_span := url_match.span())[1]
                                    - match_span[0]
                                    + 1
                                )
                            ) <= EMBED_CHAR_LIMITS["footer.text"]:
                                embed_dict["footer"]["text"] = footer_text[
                                    : url_match.start()
                                ]
                                next_embed_dict["footer"]["text"] = (
                                    footer_text[url_match.start() :]
                                    + f' {next_embed_dict["footer"]["text"]}'
                                ).strip()
                                normal_split = False

                        if normal_split:
                            embed_dict["footer"]["text"] = footer_text[:split_index]
                            next_embed_dict["footer"]["text"] = (
                                footer_text[split_index:]
                                + f' {next_embed_dict["footer"]["text"]}'
                            ).strip()

                        if not embed_dict["footer"]["text"]:
                            del embed_dict["footer"]["text"]

                        if next_embed_dict["footer"]:
                            embed_dicts.insert(i + 1, next_embed_dict)

                        updated = True

                    current_len += len(footer_text)

    return embed_dicts

def create_embed_as_dict(
    author_name: Optional[str] = None,
    author_url: Optional[str] = None,
    author_icon_url: Optional[str] = None,
    title: Optional[str] = None,
    url: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
    description: Optional[str] = None,
    image_url: Optional[str] = None,
    color: int = 0,
    fields: Optional[
        Sequence[
            Union[list[Union[
                str, bool
            ]], tuple[str, str], tuple[
                str, str, bool
            ]]
        ]
    ] = None,
    footer_text: Optional[str] = None,
    footer_icon_url: Optional[str] = None,
    timestamp: Optional[Union[str, datetime.datetime]] = None,
) -> dict:
    
    embed_dict = {}
    
    if author_name:
        embed_dict["author"] = {"name": author_name}
        if author_url:
            embed_dict["author"]["url"] = author_url
        if author_icon_url:
            embed_dict["author"]["icon_url"] = author_icon_url

    if footer_text:
        embed_dict["footer"] = {"text": footer_text}
        if footer_icon_url:
            embed_dict["footer"]["icon_url"] = footer_icon_url

    if title:
        embed_dict["title"] = title

    if url:
        embed_dict["url"] = url

    if description:
        embed_dict["description"] = description
        
    embed_dict["color"] = int(color) if 0 <= color <= 0x2F3136 else 0
    
    if timestamp:
        if isinstance(timestamp, str):
            try:
                datetime.datetime.fromisoformat(timestamp.removesuffix("Z"))
                embed_dict["timestamp"] = timestamp
            except ValueError:
                pass
        elif isinstance(timestamp, datetime.datetime):
            embed_dict["timestamp"] = timestamp.isoformat()
            
    if image_url:
        embed_dict["image"] = {"url": image_url}
        
    if thumbnail_url:
        embed_dict["thumbnail"] = {"url": thumbnail_url}
        
    if fields:
        fields_list = []
        embed_dict["fields"] = fields_list
        for i, field in enumerate(fields):
            name, value, inline = _read_embed_field_dict(field_dict=field, index=i) # type: ignore
            fields_list.append({ "name": name, "value": value, "inline": inline })
            
    return embed_dict

def _read_embed_field_dict(
    field_dict: Sequence[
        Union[list[Union[str, bool]], tuple[str, str], tuple[str, str, bool]]
    ],
    allow_incomplete: bool = False,
    index: Optional[int] = None,
) -> tuple[Optional[str], Optional[str], Optional[bool]]:
    err_str = f" at `fields[{index}]`" if index is not None else ""
    name = value = inline = None
    
    if isinstance(field_dict, dict):
        name, value, inline = (
            field_dict.get("name"),
            field_dict.get("value"),
            field_dict.get("inline"),
        )
    else:
        raise TypeError(
            f"invalid embed field type{err_str}: an embed "
            "field must be a dictionary of the structure "
            "`{'name': '...', 'value': '...'[, 'inline': True/False]}`"
        )
        
    if (
        not any((name, value, inline))
        or (
            not isinstance(name, str)
            or not isinstance(value, str)
            or not isinstance(inline, bool)
        )
        and not allow_incomplete
    ):
        raise ValueError(
            f"Invalid embed field{err_str}: an embed "
            "field must be a dictionary of the structure "
            "`{'name': '...', 'value': '...'[, 'inline': True/False]}`"
        )
        
    return name, value, inline

def create_embed(
    *,
    author_name: Optional[str] = None,
    author_url: Optional[str] = None,
    author_icon_url: Optional[str] = None,
    title: Optional[str] = None,
    url: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
    description: Optional[str] = None,
    image_url: Optional[str] = None,
    color: int = 0,
    fields: Optional[
        Sequence[Union[list[Union[str, bool]], tuple[str, str], tuple[str, str, bool]]]
    ] = None,
    footer_text: Optional[str] = None,
    footer_icon_url: Optional[str] = None,
    timestamp: Optional[Union[str, datetime.datetime]] = None,
) -> discord.Embed:
    """
    Create an embed using the specified arguments.
    
    Args:
        author_name (Optional[str], optional): The value for `author.name`.
            Defaults to None.
        author_url (Optional[str], optional): The value for `author.url`.
            Defaults to None.
        author_icon_url (Optional[str], optional): The value for `author.icon_url`.
            Defaults to None.
        title (Optional[str], optional): The value for `title`.
            Defaults to None.
        url (Optional[str], optional): The value for `url`.
            Defaults to None.
        thumbnail_url (Optional[str], optional): The value for `thumbnail_url`.
            Defaults to None.
        description (Optional[str], optional): The value for `description`.
            Defaults to None.
        image_url (Optional[str], optional): The value for `image.url`.
            Defaults to None.
        color (int, optional): The value for `color`.
            Defaults to 0.
        fields (Optional[
            Sequence[Union[list[Union[str, bool]], tuple[str, str, bool]]]
        ], optional): The value for `fields`, which must be a sequence of embed fields.
            Those can be an embed field dictionary or a 3-item list/tuple of values for
            `fields.N.name`, `fields.N.value`, and `fields.N.inline`.
            Defaults to None.
        footer_text (Optional[str], optional): The value for `footer.text`.
            Defaults to None.
        footer_icon_url (Optional[str], optional): The value for `footer.icon_url`.
            Defaults to None.
        timestamp (Optional[Union[str, datetime.datetime]], optional): The value for `timestamp`.
            Defaults to None.
            
    Returns:
        Embed: The created embed.
        
    Raises:
        TypeError: Invalid argument types.
    """
    embed: discord.Embed = discord.Embed(
        title=title,
        url=url,
        description=description,
        color=color if 0 <= color <= 0x2F3136 else 0,
    )
    
    if timestamp:
        if isinstance(timestamp, str):
            try:
                embed.timestamp = datetime.datetime.fromisoformat(
                    timestamp.removesuffix("Z")
                )
            except ValueError:
                pass
        elif isinstance(timestamp, datetime.datetime):
            embed.timestamp = timestamp
        else:
            raise TypeError(
                "Argument 'timestamp' must be None or string or a datetime object."
            )
            
    if author_name:
        embed.set_author(name=author_name, url=author_url, icon_url=author_icon_url)
        
    if thumbnail_url:
        embed.set_thumbnail(url=thumbnail_url)
        
    if image_url:
        embed.set_image(url=image_url)
        
    if fields:
        for i, field in enumerate(fields):
            name, value, inline = _read_embed_field_dict(field_dict=field, index=i) # type: ignore
            embed.add_field(
                name=name,
                value=value,
                inline=inline or False,
            )
            
    if footer_text:
        embed.set_footer(text=footer_text, icon_url=footer_icon_url)
        
    return embed
        