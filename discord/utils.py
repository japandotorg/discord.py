# -*- coding: utf-8 -*-

"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import time
import array
import asyncio
import unicodedata
from base64 import b64encode, b64decode
from bisect import bisect_left
import datetime
import functools
from inspect import isawaitable as _isawaitable, signature as _signature
from operator import attrgetter
import json
import re
import warnings
import uuid as uuid_

from typing import (
    AsyncIterator,
    NamedTuple,
    Set,
    Literal,
    List,
    Tuple,
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
    TYPE_CHECKING,
)
from typing_extensions import ParamSpec, Self

from .errors import InvalidArgument, HTTPException
from .permissions import Permissions

# from .template import Template

DISCORD_EPOCH = 1420070400000
MAX_ASYNCIO_SECONDS = 3456000


class _MissingSentinel:
    def __eq__(self, other):
        return False

    def __bool__(self):
        return False

    def __hash__(self):
        return 0

    def __repr__(self):
        return "..."


MISSING: Any = _MissingSentinel()


class cached_property:
    def __init__(self, function):
        self.function = function
        self.__doc__ = getattr(function, "__doc__")

    def __get__(self, instance, owner):
        if instance is None:
            return self

        value = self.function(instance)
        setattr(instance, self.function.__name__, value)

        return value


T = TypeVar("T")
P = ParamSpec("P")
T_co = TypeVar("T_co", covariant=True)
_Iter = Union[Iterable[T], AsyncIterable[T]]
Coro = Coroutine[Any, Any, T]
MaybeAwaitable = Union[T, Awaitable[T]]
MaybeAwaitableFunc = Callable[P, "MaybeAwaitable[T]"]

_SnowflakeListBase = array.array


class _RequestLike(Protocol):
    headers: Mapping[str, Any]


class CachedSlotProperty(Generic[T, T_co]):
    def __init__(self, name, function: Callable[[T], T_co]):
        self.name = name
        self.function = function
        self.__doc__ = getattr(function, "__doc__")

    def __get__(self, instance, owner):
        if instance is None:
            return self

        try:
            return getattr(instance, self.name)
        except AttributeError:
            value = self.function(instance)
            setattr(instance, self.name, value)
            return value


class classproperty(Generic[T_co]):
    def __init__(self, fget: Callable[[Any], T_co]) -> None:
        self.fget = fget

    def __get__(self, instance: Optional[Any], owner: Type[Any]) -> T_co:
        return self.fget(owner)

    def __set__(self, instance: Optional[Any], value: Any) -> None:
        raise AttributeError("Cannot set attribute.")


def cached_slot_property(
    name: str,
) -> Callable[[Callable[[T], T_co]], CachedSlotProperty[T, T_co]]:
    def decorator(func: Callable[[T], T_co]) -> CachedSlotProperty[T, T_co]:
        return CachedSlotProperty(name, func)

    return decorator


class SequenceProxy(Sequence[T_co]):
    """Read-only proxy of a Sequence."""

    def __init__(self, proxied: Sequence[T_co]):
        self.__proxied = proxied

    def __getitem__(self, idx: int) -> T_co:
        return self.__proxied[idx]

    def __len__(self) -> int:
        return len(self.__proxied)

    def __contains__(self, item: Any) -> bool:
        return item in self.__proxied

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.__proxied)

    def __reversed__(self) -> Iterator[T_co]:
        return reversed(self.__proxied)

    def index(self, value: Any, *args: Any, **kwargs: Any) -> int:
        return self.__proxied.index(value, *args, **kwargs)

    def count(self, value: Any) -> int:
        return self.__proxied.count(value)


@overload
def parse_time(timestamp: None) -> None:
    ...


@overload
def parse_time(timestamp: str) -> datetime.datetime:
    ...


@overload
def parse_time(timestamp: Optional[str]) -> Optional[datetime.datetime]:
    ...


def parse_time(timestamp: Optional[str]) -> Optional[datetime.datetime]:
    if timestamp:
        return datetime.datetime.fromisoformat(timestamp)
        # return datetime.datetime(*map(int, re.split(r'[^\d]', timestamp.replace('+00:00', ''))))
    return None


def copy_doc(original: Callable) -> Callable[[T], T]:
    def decorator(overriden: T) -> T:
        overriden.__doc__ = original.__doc__
        overriden.__signature__ = _signature(original)  # type: ignore
        return overriden

    return decorator


def warn_deprecated(
    name: str,
    instead: Union[str, None] = None,
    since: Union[str, None] = None,
    removed: Union[str, None] = None,
    reference: Union[str, None] = None,
) -> None:
    """
    Warn about a deprecated function, with the ability to specify details about the deprecation. Emits a
    DeprecationWarning.

    Parameters
    ----------
    name: str
        The name of the deprecated function.
    instead: Optional[:class:`str`]
        A recommended alternative to the function.
    since: Optional[:class:`str`]
        The version in which the function was deprecated. This should be in the format ``major.minor(.patch)``, where
        the patch version is optional.
    removed: Optional[:class:`str`]
        The version in which the function is planned to be removed. This should be in the format
        ``major.minor(.patch)``, where the patch version is optional.
    reference: Optional[:class:`str`]
        A reference that explains the deprecation, typically a URL to a page such as a changelog entry or a GitHub
        issue/PR.
    """
    warnings.simplefilter("always", DeprecationWarning)  # turn off filter

    message = f"{name} is deprecated"

    if since:
        message += f" since version {since}"
    if removed:
        message += f" and will be removed in version {removed}"
    if instead:
        message += f", consider using {instead} instead"
    message += "."
    if reference:
        message += f" See {reference} for more information."

    warnings.warn(message, stacklevel=3, category=DeprecationWarning)
    warnings.simplefilter("default", DeprecationWarning)


def deprecated(
    instead: Optional[str] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def actual_decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def decorated(*args: P.args, **kwargs: P.kwargs) -> T:
            warnings.simplefilter("always", DeprecationWarning)  # turn off filter
            if instead:
                fmt = "{0.__name__} is deprecated, use {1} instead."
            else:
                fmt = "{0.__name__} is deprecated."

            warnings.warn(
                fmt.format(func, instead), stacklevel=3, category=DeprecationWarning
            )
            warnings.simplefilter("default", DeprecationWarning)  # reset filter
            return func(*args, **kwargs)

        return decorated

    return actual_decorator


def oauth_url(
    client_id: Union[int, str],
    permissions: Permissions = MISSING,
    guild=None,
    redirect_uri: str = MISSING,
    scopes: Iterable[str] = MISSING,
    disable_guild_select: bool = False,
):
    """A helper function that returns the OAuth2 URL for inviting the bot
    into guilds.

    Parameters
    -----------
    client_id: :class:`str`
        The client ID for your bot.
    permissions: :class:`~discord.Permissions`
        The permissions you're requesting. If not given then you won't be requesting any
        permissions.
    guild: :class:`~discord.Guild`
        The guild to pre-select in the authorization screen, if available.
    redirect_uri: :class:`str`
        An optional valid redirect URI.
    scopes: Iterable[:class:`str`]
        An optional valid list of scopes. Defaults to ``('bot',)``.

        .. versionadded:: 1.7

    Returns
    --------
    :class:`str`
        The OAuth2 URL for inviting the bot into guilds.
    """
    url = "https://discord.com/oauth2/authorize?client_id={}".format(client_id)
    url += "&scope=" + "+".join(scopes or ("bot", "applications.commands"))
    if permissions is not MISSING:
        # url = url + '&permissions=' + str(permissions.value)
        url += f"&permissions={permissions.value}"
    if guild is not None:
        # url = url + "&guild_id=" + str(guild.id)
        url += f"&guild_id={guild.id}"
    if redirect_uri is not MISSING:
        from urllib.parse import urlencode

        url += "&response_type=code&" + urlencode({"redirect_uri": redirect_uri})
    if disable_guild_select:
        url += "&disable_guild_select=true"
    return url


def snowflake_time(id: int) -> datetime.datetime:
    """
    Parameters
    -----------
    id: :class:`int`
        The snowflake ID.

    Returns
    --------
    :class:`datetime.datetime`
        The creation date in UTC of a Discord snowflake ID."""
    return datetime.datetime.utcfromtimestamp(((id >> 22) + DISCORD_EPOCH) / 1000)


def time_snowflake(datetime_obj: datetime.datetime, high: bool = False) -> int:
    """Returns a numeric snowflake pretending to be created at the given date.

    When using as the lower end of a range, use ``time_snowflake(high=False) - 1`` to be inclusive, ``high=True`` to be exclusive
    When using as the higher end of a range, use ``time_snowflake(high=True)`` + 1 to be inclusive, ``high=False`` to be exclusive

    Parameters
    -----------
    datetime_obj: :class:`datetime.datetime`
        A timezone-naive datetime object representing UTC time.
    high: :class:`bool`
        Whether or not to set the lower 22 bit to high or low.
    """
    # unix_seconds = (datetime_obj - type(datetime_obj)(1970, 1, 1)).total_seconds()
    discord_millis = int(datetime_obj.timestamp() * 1000 - DISCORD_EPOCH)

    return (discord_millis << 22) + (2**22 - 1 if high else 0)


def _find(predicate: Callable[[T], Any], iterable: Iterable[T], /) -> Optional[T]:
    return next((element for element in iterable if predicate(element)), None)


async def _afind(
    predicate: Callable[[T], Any], iterable: AsyncIterable[T], /
) -> Optional[T]:
    async for element in iterable:
        if predicate(element):
            return element

    return None


@overload
def __find(predicate: Callable[[T], Any], iterable: Iterable[T], /) -> Optional[T]:
    ...


@overload
def __find(
    predicate: Callable[[T], Any], iterable: AsyncIterable[T], /
) -> Coro[Optional[T]]:
    ...


def __find(
    predicate: Callable[[T], Any], iterable: _Iter[T], /
) -> Union[Optional[T], Coro[Optional[T]]]:
    """A helper to return the first element found in the sequence
    that meets the predicate. For example: ::

        member = discord.utils.find(lambda m: m.name == 'Mighty', channel.guild.members)

    would find the first :class:`~discord.Member` whose name is 'Mighty' and return it.
    If an entry is not found, then ``None`` is returned.

    This is different from :func:`py:filter` due to the fact it stops the moment it finds
    a valid entry.

    Parameters
    -----------
    predicate
        A function that returns a boolean-like result.
    seq: iterable
        The iterable to search through.
    """

    return (
        _find(predicate, iterable)  # type: ignore
        if hasattr(iterable, "__iter__")
        else _afind(predicate, iterable)  # type: ignore
    )


def find(predicate, seq):
    """A helper to return the first element found in the sequence
    that meets the predicate. For example: ::

        member = discord.utils.find(lambda m: m.name == 'Mighty', channel.guild.members)

    would find the first :class:`~discord.Member` whose name is 'Mighty' and return it.
    If an entry is not found, then ``None`` is returned.

    This is different from :func:`py:filter` due to the fact it stops the moment it finds
    a valid entry.

    Parameters
    -----------
    predicate
        A function that returns a boolean-like result.
    seq: iterable
        The iterable to search through.
    """

    for element in seq:
        if predicate(element):
            return element
    return None


def _get(iterable: Iterable[T], /, **attrs: Any) -> Optional[T]:
    _all = all
    attrget = attrgetter

    if len(attrs) == 1:
        k, v = attrs.popitem()
        pred = attrget(k.replace("__", "."))
        return next((elem for elem in iterable if pred(elem) == v), None)

    converted = [
        (attrget(attr.replace("__", ".")), value) for attr, value in attrs.items()
    ]

    for elem in iterable:
        if _all(pred(elem) == value for pred, value in converted):
            return elem

    return None


async def _aget(iterable: AsyncIterable[T], /, **attrs: Any) -> Optional[T]:
    _all = all
    attrget = attrgetter

    if len(attrs) == 1:
        k, v = attrs.popitem()
        pred = attrget(k.replace("__", "."))
        async for elem in iterable:
            if pred(elem) == v:
                return elem

        return None

    converted = [
        (attrget(attr.replace("__", ".")), value) for attr, value in attrs.items()
    ]

    async for elem in iterable:
        if _all(pred(elem) == value for pred, value in converted):
            return elem

    return None


@overload
def __get(iterable: Iterable[T], /, **attrs: Any) -> Optional[T]:
    ...


@overload
def __get(iterable: AsyncIterable[T], /, **attrs: Any) -> Coro[Optional[T]]:
    ...


def __get(iterable: _Iter[T], /, **attrs: Any) -> Union[Optional[T], Coro[Optional[T]]]:
    r"""A helper that returns the first element in the iterable that meets
    all the traits passed in ``attrs``. This is an alternative for
    :func:`~discord.utils.find`.

    When multiple attributes are specified, they are checked using
    logical AND, not logical OR. Meaning they have to meet every
    attribute passed in and not one of them.

    To have a nested attribute search (i.e. search by ``x.y``) then
    pass in ``x__y`` as the keyword argument.

    If nothing is found that matches the attributes passed, then
    ``None`` is returned.

    .. versionchanged:: 1.7.69

        The ``iterable`` parameter is now positional-only.

    .. versionchanged:: 1.7.69

        The ``iterable`` parameter supports :term:`asynchronous iterable`\s.

    Examples
    ---------

    Basic usage:

    .. code-block:: python3

        member = discord.utils.get(message.guild.members, name='Foo')

    Multiple attribute matching:

    .. code-block:: python3

        channel = discord.utils.get(guild.voice_channels, name='Foo', bitrate=64000)

    Nested attribute matching:

    .. code-block:: python3

        channel = discord.utils.get(client.get_all_channels(), guild__name='Cool', name='general')

    Async iterables:

    .. code-block:: python3

        msg = await discord.utils.get(channel.history(), author__name='Dave')

    Parameters
    -----------
    iterable: Union[:class:`collections.abc.Iterable`, :class:`collections.abc.AsyncIterable`]
        The iterable to search through. Using a :class:`collections.abc.AsyncIterable`,
        makes this function return a :term:`coroutine`.
    \*\*attrs
        Keyword arguments that denote attributes to search with.
    """

    return (
        _get(iterable, **attrs)  # type: ignore
        if hasattr(
            iterable, "__iter__"
        )  # isinstance(iterable, collections.abc.Iterable) is too slow
        else _aget(iterable, **attrs)  # type: ignore
    )


def get(iterable, **attrs):
    r"""A helper that returns the first element in the iterable that meets
    all the traits passed in ``attrs``. This is an alternative for
    :func:`~discord.utils.find`.

    When multiple attributes are specified, they are checked using
    logical AND, not logical OR. Meaning they have to meet every
    attribute passed in and not one of them.

    To have a nested attribute search (i.e. search by ``x.y``) then
    pass in ``x__y`` as the keyword argument.

    If nothing is found that matches the attributes passed, then
    ``None`` is returned.

    Examples
    ---------

    Basic usage:

    .. code-block:: python3

        member = discord.utils.get(message.guild.members, name='Foo')

    Multiple attribute matching:

    .. code-block:: python3

        channel = discord.utils.get(guild.voice_channels, name='Foo', bitrate=64000)

    Nested attribute matching:

    .. code-block:: python3

        channel = discord.utils.get(client.get_all_channels(), guild__name='Cool', name='general')

    Parameters
    -----------
    iterable
        An iterable to search through.
    \*\*attrs
        Keyword arguments that denote attributes to search with.
    """

    # global -> local
    _all = all
    attrget = attrgetter

    # Special case the single element call
    if len(attrs) == 1:
        k, v = attrs.popitem()
        pred = attrget(k.replace("__", "."))
        for elem in iterable:
            if pred(elem) == v:
                return elem
        return None

    converted = [
        (attrget(attr.replace("__", ".")), value) for attr, value in attrs.items()
    ]

    for elem in iterable:
        if _all(pred(elem) == value for pred, value in converted):
            return elem
    return None


async def get_or_fetch(obj, attr: str, id: int, *, default: Any = MISSING) -> Any:
    """|coro|

    Attempts to get an attribute from the object in cache. If it fails, it will attempt to fetch it.
    If the fetch also fails, an error will be raised.

    Parameters
    ----------
    obj: Any
        The object to use the get or fetch methods in
    attr: :class:`str`
        The attribute to get or fetch. Note the object must have both a ``get_`` and ``fetch_`` method for this attribute.
    id: :class:`int`
        The ID of the object
    default: Any
        The default value to return if the object is not found, instead of raising an error.

    Returns
    -------
    Any
        The object found or the default value.

    Raises
    ------
    :exc:`AttributeError`
        The object is missing a ``get_`` or ``fetch_`` method
    :exc:`NotFound`
        Invalid ID for the object
    :exc:`HTTPException`
        An error occurred fetching the object
    :exc:`Forbidden`
        You do not have permission to fetch the object

    Examples
    --------
    Getting a guild from a guild ID: ::
        guild = await utils.get_or_fetch(client, 'guild', guild_id)
    Getting a channel from the guild. If the channel is not found, return None: ::
        channel = await utils.get_or_fetch(guild, 'channel', channel_id, default=None)
    """
    getter = getattr(obj, f"get_{attr}")(id)

    if getter is None:
        try:
            getter = await getattr(obj, f"fetch_{attr}")(id)
        except AttributeError:
            getter = await getattr(obj, f"_fetch_{attr}")(id)
            if getter is None:
                raise ValueError(f"Could not find {attr} with id {id} on {obj}")
        except (HTTPException, ValueError):
            if default is not MISSING:
                return default
            else:
                raise

    return getter


def _unique(iterable: Iterable[T]) -> List[T]:
    # seen = set()
    # adder = seen.add
    # return [x for x in iterable if not (x in seen or adder(x))]
    return [x for x in dict.fromkeys(iterable)]


def _get_as_snowflake(data: Any, key: str) -> Optional[int]:
    try:
        value = data[key]
    except KeyError:
        return None
    else:
        return value and int(value)


def _get_mime_type_for_image(data: bytes):
    if data.startswith(b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A"):
        return "image/png"
    elif data[0:3] == b"\xff\xd8\xff" or data[6:10] in (b"JFIF", b"Exif"):
        return "image/jpeg"
    elif data.startswith((b"\x47\x49\x46\x38\x37\x61", b"\x47\x49\x46\x38\x39\x61")):
        return "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    else:
        raise InvalidArgument("Unsupported image type given")


def _bytes_to_base64_data(data: bytes):
    fmt = "data:{mime};base64,{data}"
    mime = _get_mime_type_for_image(data)
    b64 = b64encode(data).decode("ascii")
    return fmt.format(mime=mime, data=b64)


def to_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)


from_json = json.loads


def _parse_ratelimit_header(request: Any, *, use_clock: bool = False) -> float:
    reset_after = request.headers.get("X-Ratelimit-Reset-After")
    if use_clock or not reset_after:
        utc = datetime.timezone.utc
        now = datetime.datetime.now(utc)
        reset = datetime.datetime.fromtimestamp(
            float(request.headers["X-Ratelimit-Reset"]), utc
        )
        return (reset - now).total_seconds()
    else:
        return float(reset_after)


# async def maybe_coroutine(
#     f: MaybeAwaitableFunc[P, T], *args: P.args, **kwargs: P.kwargs
# ) -> T:
async def maybe_coroutine(
    f: Callable[P, Union[Awaitable[T], T]], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    value = f(*args, **kwargs)
    if _isawaitable(value):
        return await value
    else:
        return value  # type: ignore


async def async_all(
    gen: Iterable[Awaitable[T]], *, check: Callable[[T], bool] = _isawaitable
) -> bool:
    for elem in gen:
        if check(elem):  # type: ignore
            elem = await elem
        if not elem:
            return False
    return True


async def sane_wait_for(
    futures: Iterable[Awaitable[T]], *, timeout: Optional[float]
) -> Set[asyncio.Task[T]]:
    ensured = [asyncio.ensure_future(fut) for fut in futures]
    done, pending = await asyncio.wait(
        ensured, timeout=timeout, return_when=asyncio.ALL_COMPLETED
    )

    if len(pending) != 0:
        raise asyncio.TimeoutError()

    return done


def get_slots(cls: Type[Any]) -> Iterator[str]:
    for mro in reversed(cls.__mro__):
        try:
            yield from mro.__slots__  # type: ignore
        except AttributeError:
            continue


def compute_timedelta(dt: datetime.datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.astimezone()
    now = datetime.datetime.now(datetime.timezone.utc)
    return max((dt - now).total_seconds(), 0)


async def sleep_until(
    when: datetime.datetime, result: Optional[T] = None
) -> Optional[T]:
    """|coro|

    Sleep until a specified time.

    If the time supplied is in the past this function will yield instantly.

    .. versionadded:: 1.3

    Parameters
    -----------
    when: :class:`datetime.datetime`
        The timestamp in which to sleep until. If the datetime is naive then
        it is assumed to be in UTC.
    result: Any
        If provided is returned to the caller when the coroutine completes.
    """
    # if when.tzinfo is None:
    #     when = when.replace(tzinfo=datetime.timezone.utc)
    # now = datetime.datetime.now(datetime.timezone.utc)
    # delta = (when - now).total_seconds()
    # while delta > MAX_ASYNCIO_SECONDS:
    #     await asyncio.sleep(MAX_ASYNCIO_SECONDS)
    #     delta -= MAX_ASYNCIO_SECONDS
    # return await asyncio.sleep(max(delta, 0), result)
    delta = compute_timedelta(when)
    return await asyncio.sleep(delta, result)


def utcnow() -> datetime.datetime:
    """
    A helper function to return an aware UTC datetime representing the current time.

    This should be preferred to :meth:`datetime.datetime.utcnow` since it is an aware
    datetime, compared to the naive datetime in the standard library.

    .. versionadded:: 1.7.69

    Returns
    --------
    :class:`datetime.datetime`
        The current aware datetime in UTC.
    """
    return datetime.datetime.now(datetime.timezone.utc)


def local_time() -> datetime.datetime:
    """
    A helper function to return the current datetime for the system's timezone.

    .. versionadded:: 1.7.69

    Returns
    -------
    :class:`datetime.datetime`
        The current datetime for the system's timezone.
    """
    return utcnow().astimezone()


def monotonic() -> int:
    """
    Performance counter for benchmarking.
    """
    return time.perf_counter()  # type: ignore


def monotonic_ns() -> int:
    """
    Performance counter for benchmarking as nanoseconds.
    """
    return time.perf_counter_ns()


def uuid() -> str:
    """
    Generates an unique UUID (1ns precision).
    """
    return uuid_.uuid1(None, monotonic_ns()).hex


def valid_icon_size(size: int) -> bool:
    """Icons must be power of 2 within [16, 4096]."""
    return not size & (size - 1) and 4096 >= size >= 16


class SnowflakeList(array.array):
    """Internal data storage class to efficiently store a list of snowflakes.

    This should have the following characteristics:

    - Low memory usage
    - O(n) iteration (obviously)
    - O(n log n) initial creation if data is unsorted
    - O(log n) search and indexing
    - O(n) insertion
    """

    __slots__ = ()

    if TYPE_CHECKING:

        def __init__(self, data: Iterable[int], *, is_sorted: bool = False):
            ...

    def __new__(cls, data: Iterable[int], *, is_sorted: bool = False) -> Self:
        return array.array.__new__(cls, "Q", data if is_sorted else sorted(data))  # type: ignore

    def add(self, element: int) -> None:
        i = bisect_left(self, element)
        self.insert(i, element)

    def get(self, element: int) -> Optional[int]:
        i = bisect_left(self, element)
        return self[i] if i != len(self) and self[i] == element else None

    def has(self, element: int) -> bool:
        i = bisect_left(self, element)
        return i != len(self) and self[i] == element


_IS_ASCII = re.compile(r"^[\x00-\x7f]+$")


def _string_width(string: str, *, _IS_ASCII=_IS_ASCII) -> int:
    """Returns string's width."""
    match = _IS_ASCII.match(string)
    if match:
        return match.endpos

    UNICODE_WIDE_CHAR_TYPE = "WFA"
    func = unicodedata.east_asian_width
    return sum(2 if func(char) in UNICODE_WIDE_CHAR_TYPE else 1 for char in string)


class ResolvedInvite(NamedTuple):
    code: str
    event: Optional[int]


def resolve_invite(invite):
    """
    Resolves an invite from a :class:`~discord.Invite`, URL or code.

    Parameters
    -----------
    invite: Union[:class:`~discord.Invite`, :class:`str`]
        The invite.

    Returns
    --------
    :class:`str`
        The invite code.
    """
    from .invite import Invite  # circular import

    if isinstance(invite, Invite):
        return invite.code
    else:
        rx = r"(?:https?\:\/\/)?discord(?:\.gg|(?:app)?\.com\/invite)\/(.+)"
        m = re.match(rx, invite)
        if m:
            return m.group(1)
    return invite


def resolve_template(code):
    """
    Resolves a template code from a :class:`~discord.Template`, URL or code.

    .. versionadded:: 1.4

    Parameters
    -----------
    code: Union[:class:`~discord.Template`, :class:`str`]
        The code.

    Returns
    --------
    :class:`str`
        The template code.
    """
    from .template import Template  # circular import

    if isinstance(code, Template):
        return code.code
    else:
        rx = r"(?:https?\:\/\/)?discord(?:\.new|(?:app)?\.com\/template)\/(.+)"
        m = re.match(rx, code)
        if m:
            return m.group(1)
    return code


_MARKDOWN_ESCAPE_SUBREGEX = "|".join(
    r"\{0}(?=([\s\S]*((?<!\{0})\{0})))".format(c) for c in ("*", "`", "_", "~", "|")
)

_MARKDOWN_ESCAPE_COMMON = r"^>(?:>>)?\s|\[.+\]\(.+\)"

_MARKDOWN_ESCAPE_REGEX = re.compile(
    r"(?P<markdown>%s|%s)" % (_MARKDOWN_ESCAPE_SUBREGEX, _MARKDOWN_ESCAPE_COMMON),
    re.MULTILINE,
)

_URL_REGEX = r"(?P<url><[^: >]+:\/[^ >]+>|(?:https?|steam):\/\/[^\s<]+[^<.,:;\"\'\]\s])"

_MARKDOWN_STOCK_REGEX = r"(?P<markdown>[_\\~|\*`]|%s)" % _MARKDOWN_ESCAPE_COMMON


def remove_markdown(text: str, *, ignore_links: bool = True) -> str:
    """A helper function that removes markdown characters.

    .. versionadded:: 1.7

    .. note::
            This function is not markdown aware and may remove meaning from the original text. For example,
            if the input contains ``10 * 5`` then it will be converted into ``10  5``.

    Parameters
    -----------
    text: :class:`str`
        The text to remove markdown from.
    ignore_links: :class:`bool`
        Whether to leave links alone when removing markdown. For example,
        if a URL in the text contains characters such as ``_`` then it will
        be left alone. Defaults to ``True``.

    Returns
    --------
    :class:`str`
        The text with the markdown special characters removed.
    """

    def replacement(match):
        groupdict = match.groupdict()
        return groupdict.get("url", "")

    regex = _MARKDOWN_STOCK_REGEX
    if ignore_links:
        regex = "(?:%s|%s)" % (_URL_REGEX, regex)
    return re.sub(regex, replacement, text, 0, re.MULTILINE)


def escape_markdown(
    text: str, *, as_needed: bool = False, ignore_links: bool = True
) -> str:
    r"""A helper function that escapes Discord's markdown.

    Parameters
    -----------
    text: :class:`str`
        The text to escape markdown from.
    as_needed: :class:`bool`
        Whether to escape the markdown characters as needed. This
        means that it does not escape extraneous characters if it's
        not necessary, e.g. ``**hello**`` is escaped into ``\*\*hello**``
        instead of ``\*\*hello\*\*``. Note however that this can open
        you up to some clever syntax abuse. Defaults to ``False``.
    ignore_links: :class:`bool`
        Whether to leave links alone when escaping markdown. For example,
        if a URL in the text contains characters such as ``_`` then it will
        be left alone. This option is not supported with ``as_needed``.
        Defaults to ``True``.

    Returns
    --------
    :class:`str`
        The text with the markdown special characters escaped with a slash.
    """

    if not as_needed:

        def replacement(match):
            groupdict = match.groupdict()
            is_url = groupdict.get("url")
            if is_url:
                return is_url
            return "\\" + groupdict["markdown"]

        regex = _MARKDOWN_STOCK_REGEX
        if ignore_links:
            regex = "(?:%s|%s)" % (_URL_REGEX, regex)
        return re.sub(regex, replacement, text, 0, re.MULTILINE)
    else:
        text = re.sub(r"\\", r"\\\\", text)
        return _MARKDOWN_ESCAPE_REGEX.sub(r"\\\1", text)


def escape_mentions(text: str) -> str:
    """A helper function that escapes everyone, here, role, and user mentions.

    .. note::

        This does not include channel mentions.

    .. note::

        For more granular control over what mentions should be escaped
        within messages, refer to the :class:`~discord.AllowedMentions`
        class.

    Parameters
    -----------
    text: :class:`str`
        The text to escape mentions from.

    Returns
    --------
    :class:`str`
        The text with the mentions removed.
    """
    return re.sub(r"@(everyone|here|[!&]?[0-9]{17,20})", "@\u200b\\1", text)


def parse_token(token: str) -> Tuple[int, datetime.datetime, bytes]:
    """
    Parse a token into its parts

    Parameters
    ----------
    token: :class:`str`
        The bot token

    Returns
    -------
    Tuple[:class:`int`, :class:`datetime.datetime`, :class:`bytes`]
        The bot's ID, the time when the token was generated and the hmac.
    """
    parts = token.split(".")

    user_id = int(b64decode(parts[0]))

    timestamp = int.from_bytes(b64decode(parts[1] + "=="), "big")
    created_at = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)

    hmac = b64decode(parts[2] + "==")

    return user_id, created_at, hmac


def parse_raw_mentions(text: str) -> List[int]:
    """
    A helper function that parses mentions from a string as an array of
    :class:`~discord.User` IDs matched with the syntax of ``<@user_id>`` or ``<@!user_id>``.

    .. note::

        This does not include role or channel mentions. See :func:`parse_raw_role_mentions`
        and :func:`parse_raw_channel_mentions` for those.

    .. versionadded:: 1.7.69

    Parameters
    ----------
    text: :class:`str`
        The text to parse mentions from.

    Returns
    -------
    List[:class:`int`]
        A list of user IDs that were mentioned.
    """
    return [int(x) for x in re.findall(r"<@!?(\d{15,20})>", text)]


def parse_raw_role_mentions(text: str) -> List[int]:
    """
    A helper function that parses mentions from a string as an array of
    :class:`~discord.Role` IDs matched with the syntax of ``<@&role_id>``.

    .. versionadded:: 1.7.69

    Parameters
    ----------
    text: :class:`str`
        The text to parse mentions from.

    Returns
    -------
    List[:class:`int`]
        A list of role IDs that were mentioned.
    """
    return [int(x) for x in re.findall(r"<@&(\d{15,20})>", text)]


def parse_raw_channel_mentions(text: str) -> List[int]:
    """
    A helper function that parses mentions from a string as an array of
    :class:`~discord.abc.GuildChannel` IDs matched with the syntax of ``<#channel_id>``.

    .. versionadded:: 1.7.69

    Parameters
    ----------
    text: :class:`str`
        The text to parse mentions from.

    Returns
    -------
    List[:class:`int`]
        A list of channel IDs that were mentioned.
    """
    return [int(x) for x in re.findall(r"<#(\d{15,20})>", text)]


def _chunk(iterator: Iterable[T], max_size: int) -> Iterator[List[T]]:
    ret = []
    n = 0
    for item in iterator:
        ret.append(item)
        n += 1
        if n == max_size:
            yield ret
            ret = []
            n = 0

    if ret:
        yield ret


async def _achunk(iterator: AsyncIterable[T], max_size: int) -> AsyncIterator[List[T]]:
    ret = []
    n = 0
    async for item in iterator:
        ret.append(item)
        n += 1
        if n == max_size:
            yield ret
            ret = []
            n = 0
    if ret:
        yield ret


@overload
def as_chunks(iterator: Iterable[T], max_size: int) -> Iterator[List[T]]:
    ...


@overload
def as_chunks(iterator: AsyncIterable[T], max_size: int) -> AsyncIterator[List[T]]:
    ...


def as_chunks(iterator: _Iter[T], max_size: int) -> _Iter[List[T]]:
    """
    A helper function that collects an iterator into chunks of a given size.

    .. versionadded:: 1.7.69

    Parameters
    ----------
    iterator: Union[:class:`collections.abc.Iterable`, :class:`collections.abc.AsyncIterable`]
        The iterator to chunk, can be sync or async.
    max_size: :class:`int`
        The maximum chunk size.

    .. warning::

        The last chunk collected may not be as large as ``max_size``.

    Returns
    --------
    Union[:class:`Iterator`, :class:`AsyncIterator`]
        A new iterator which yields chunks of a given size.
    """
    if max_size <= 0:
        raise ValueError("Chunk sizes must be greater than 0.")

    if isinstance(iterator, AsyncIterable):
        return _achunk(iterator, max_size)
    return _chunk(iterator, max_size)


def is_inside_class(func: Callable[..., Any]) -> bool:
    if func.__qualname__ == func.__name__:
        return False
    (remaining, _, _) = func.__qualname__.rpartition(".")
    return not remaining.endswith("<locals>")


TimestampStyle = Literal["f", "F", "d", "D", "t", "T", "R"]


def format_dt(dt: datetime.datetime, /, style: Optional[TimestampStyle] = None) -> str:
    """
    A helper function to format a :class:`datetime.datetime` for presentation within Discord.

    This allows for a locale-independent way of presenting data using Discord specific Markdown.

    +-------------+----------------------------+-----------------+
    |    Style    |       Example Output       |   Description   |
    +=============+============================+=================+
    | t           | 22:57                      | Short Time      |
    +-------------+----------------------------+-----------------+
    | T           | 22:57:58                   | Long Time       |
    +-------------+----------------------------+-----------------+
    | d           | 17/05/2016                 | Short Date      |
    +-------------+----------------------------+-----------------+
    | D           | 17 May 2016                | Long Date       |
    +-------------+----------------------------+-----------------+
    | f (default) | 17 May 2016 22:57          | Short Date Time |
    +-------------+----------------------------+-----------------+
    | F           | Tuesday, 17 May 2016 22:57 | Long Date Time  |
    +-------------+----------------------------+-----------------+
    | R           | 5 years ago                | Relative Time   |
    +-------------+----------------------------+-----------------+

    Note that the exact output depends on the user's locale setting in the client. The example output
    presented is using the ``en-GB`` locale.

    .. versionadded:: 1.7.69

    Parameters
    -----------
    dt: :class:`datetime.datetime`
        The datetime to format.
    style: :class:`str`
        The style to format the datetime with.

    Returns
    --------
    :class:`str`
        The formatted string.
    """
    if style is None:
        return f"<t:{int(dt.timestamp())}>"
    return f"<t:{int(dt.timestamp())}:{style}>"


def generate_snowflake(dt: Union[datetime.datetime, None] = None) -> int:
    """
    Returns a numeric snowflake pretending to be created at the given date but more accurate and random
    than :func:`time_snowflake`. If dt is not passed, it makes one from the current time using utcnow.

    Parameters
    ----------
    dt: :class:`datetime.datetime`
        A datetime object to convert to a snowflake.
        If naive, the timezone is assumed to be local time.

    Returns
    -------
    :class:`int`
        The snowflake representing the time given.
    """
    dt = dt or utcnow()
    return int(dt.timestamp() * 1000 - DISCORD_EPOCH) << 22 | 0x3FFFFF
