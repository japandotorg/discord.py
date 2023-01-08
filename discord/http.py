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

import asyncio
import json
import logging
import sys
import weakref
import datetime
from urllib.parse import quote as _uriquote
from typing import (
    Any,
    ClassVar,
    Coroutine,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Type,
    TypeVar,
    Union,
)

import aiohttp

from .file import File
from .errors import HTTPException, Forbidden, NotFound, LoginFailure, DiscordServerError, GatewayNotFound, RateLimited
from .gateway import DiscordClientWebSocketResponse
from . import __version__, utils, message
from .mentions import AllowedMentions
from .utils import MISSING

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing_extensions import Self
    from types import TracebackType

    from .embeds import Embed
    from .message import Attachment
    from .enums import AuditLogAction


    T = TypeVar('T')
    BE = TypeVar('BE', bound=BaseException)
    Response = Coroutine[Any, Any, T]

async def json_or_text(response: aiohttp.ClientResponse) -> Union[Dict[str, Any], str]:
    text = await response.text(encoding='utf-8')
    try:
        if response.headers['content-type'] == 'application/json':
            return json.loads(text)
    except KeyError:
        # Thanks Cloudflare
        pass

    return text

class MultipartParameters(NamedTuple):
    payload: Optional[Dict[str, Any]]
    multipart: Optional[List[Dict[str, Any]]]
    files: Optional[Sequence[File]]

    def __enter__(self) -> Self:
        return self

    def __exit__(
            self,
            exc_type: Optional[Type[BE]],
            exc: Optional[BE],
            traceback: Optional[TracebackType],
    ) -> None:
        if self.files:
            for file in self.files:
                file.close()


def handle_message_parameters(
    content: Optional[str] = MISSING,
    *,
    username: str = MISSING,
    avatar_url: Any = MISSING,
    tts: bool = False,
    nonce: Optional[Union[int, str]] = None,
    file: File = MISSING,
    files: Sequence[File] = MISSING,
    embed: Optional[Embed] = MISSING,
    embeds: Sequence[Embed] = MISSING,
    attachments: Sequence[Union[Attachment, File]] = MISSING,
    allowed_mentions: Optional[AllowedMentions] = MISSING,
    message_reference: Optional[message.MessageReference] = MISSING,
    previous_allowede_mentions: Optional[AllowedMentions] = MISSING,
    mention_author: Optional[bool] = None,
    channel_payload: Dict[str, Any] = MISSING,
 ) -> MultipartParameters:
    if files is not MISSING and file is not MISSING:
        raise TypeError('Cannot mix file and files keyword argument.')

    if embeds is not MISSING and embed is not MISSING:
        raise TypeError('Cannot mis embed and embeds keyword argument.')

    if file is not MISSING:
        files = [file]

    if attachments is not MISSING and files is not MISSING:
        raise TypeError('Cannot mix attachments and files keyword arguments.')

    payload = {}
    if embeds is not MISSING:
        if len(embeds) > 10:
            raise ValueError('embeds has maximum of 10 elements')
        payload['embeds'] = [e.to_dict() for e in embeds]

    if embed is not MISSING:
        if embed is None:
            payload['embeds'] = []
        else:
            payload['embeds'] = [embed.to_dict()]

    if content is not MISSING:
        if content is not None:
            payload['content'] = str(content)
        else:
            payload['content'] = None

    if nonce is not None:
        payload['nonce'] = str(nonce)

    if message_reference is not MISSING:
        payload['message_reference'] = message_reference

    payload['tts'] = tts

    if avatar_url:
        payload['avatar_url'] = str(avatar_url)

    if username:
        payload['username'] = username

    if allowed_mentions:
        if previous_allowede_mentions is not None:
            payload['allowed_mentions'] = previous_allowede_mentions.merge(allowed_mentions)
        else:
            payload['allowed_mentions'] = allowed_mentions.to_dict()
    elif previous_allowede_mentions is not None:
        payload['allowed_mentions'] = previous_allowede_mentions.to_dict()

    if mention_author is not None:
        if 'allowed_mentions' not in payload:
            payload['allowed_mentions'] = AllowedMentions().to_dict()
        payload['allowed_mentions']['replied_user'] = mention_author

    if attachments is MISSING:
        attachments = files
    else:
        files = [a for a in attachments if isinstance(a, File)]

    if attachments is not MISSING:
        file_index = 0
        attachments_payload = []
        for attachment in attachments:
            if isinstance(attachment, File):
                attachments_payload.append(attachment.to_dict(file_index))
                file_index += 1
            else:
                attachments_payload.append(attachment.to_dict())

        payload['attachments'] = attachments_payload

    if channel_payload is not MISSING:
        payload = {
            'message': payload,
        }
        payload.update(channel_payload)

    multipart = []
    if files:
        multipart.append({
            'name': 'payload_json',
            'value': utils._to_json(payload)
        })
        payload = None
        for index, file in enumerate(files):
            multipart.append(
                {
                    'name': f'files[{index}]',
                    'value': file.fp,
                    'filename': file.filename,
                    'content_type': 'application/octet-stream',
                }
            )

    return MultipartParameters(payload=payload, multipart=multipart, files=files)



INTERNAL_API_BASE: str = "https://discord.com/api"
INTERNAL_API_VERSION: int = 7


def _set_api_version(value: int):
    api_list = (
        6, 7, 8, 9, 10,
    )

    global INTERNAL_API_VERSION

    if not isinstance(value, int):
        raise TypeError(f'expected int not {value.__class__.__name__}')

    if value not in api_list:
        raise ValueError(f'expected {api_list} not {value}')

    INTERNAL_API_VERSION = value
    Route.BASE = f"{INTERNAL_API_BASE}/v{value}"

def _set_api_base(value: str):
    global INTERNAL_API_BASE

    if not isinstance(value, str):
        raise TypeError(f'expected str not {value.__class__.__name__}')

    INTERNAL_API_BASE = value
    Route.BASE = f"{value}/v{INTERNAL_API_VERSION}"


class Route:
    BASE: ClassVar[str] = 'https://discord.com/api/v7'

    def __init__(self, method: str, path: str, *, metadata: Optional[str] = None, **parameters: Any) -> None:
        self.path: str = path
        self.method: str = method
        # Metadata is a special string used to differentiate between known sub rate limits
        # Since these can't be handled generically, this is the next best way to do so.
        self.metadata: Optional[str] = metadata
        url = (self.BASE + self.path)
        if parameters:
            self.url = url.format(**{k: _uriquote(v) if isinstance(v, str) else v for k, v in parameters.items()})
        else:
            self.url: str = url

        # major parameters:
        self.channel_id: Optional[int] = parameters.get('channel_id')
        self.guild_id: Optional[int] = parameters.get('guild_id')
        self.webhook_id: Optional[int] = parameters.get('webhook_id')
        self.webhook_token: Optional[int] = parameters.get('webhook_token')

    @property
    def key(self) -> str:
        """
        The bucket key is used to represent the route in various mappings.
        """
        if self.metadata:
            return f"{self.method} {self.path}:{self.metadata}"
        return f"{self.method} {self.path}"

    @property
    def major_parameters(self) -> str:
        """
        Returns the major parameters formatted a string.

        This needs to be appended to a bucket hash to constitute as a full rate limit key.
        """
        return '+'.join(
    str(k) for k in (self.channel_id, self.guild_id, self.webhook_id, self.webhook_token) if k is not None
    )

    @property
    def bucket(self):
        # the bucket is just method + path w/ major parameters
        return '{0.channel_id}:{0.guild_id}:{0.path}'.format(self)


class MaybeUnlock:
    def __init__(self, lock):
        self.lock = lock
        self._unlock = True

    def __enter__(self):
        return self

    def defer(self):
        self._unlock = False

    def __exit__(self, type, value, traceback):
        if self._unlock:
            self.lock.release()


class Ratelimit:
    """
    Represents a Discord rate limit.
    """

    __slots__ = (
        'limit',
                'remaining',
                'outgoing',
                'reset_after',
                'expires',
                'dirty',
                '_last_request',
                '_max_ratelimit_timeout',
                '_loop',
                '_pending_requests',
                '_sleeping',
    )

    def __init__(
            self,
            max_ratelimit_timeout: Optional[float]
    ) -> None:
        self.limit: int = 1
        self.remaining: int = self.limit
        self.outgoing: int = 0
        self.reset_after: float = 0.0
        self.expires: Optional[float] = None
        self.dirty: bool = False
        self._max_ratelimit_timeout: Optional[float] = max_ratelimit_timeout
        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._pending_requests: deque[asyncio.Future[Any]] = deque()

        self._sleeping: asyncio.Lock = asyncio.Lock()
        self._last_request: float = self._loop.time()

    def __repr__(self) -> str:
        return (
            f'<RatelimitBucket limit={self.limit}> remaining={self.remaining} pending_requests={len(self._pending_requests)}'
        )

    def reset(self):
        self.remaining = self.limit - self.outgoing
        self.expires = None
        self.reset_after = 0.0
        self.dirty = False

    def update(self, response: aiohttp.ClientResponse, *, use_clock: bool = False) -> None:
        headers = response.headers
        self.limit = int(headers.get('X-Ratelimit-Limit', 1))

        if self.dirty:
            self.remaining = min(int(headers.get('X-Ratelimit-Remaining', 0)), self.limit - self.outgoing)
        else:
            self.remaining = int(headers.get('X-Ratelimit-Remaining', 0))
            self.dirty = True

        reset_after = headers.get('X-Ratelimit-Reset-After')
        if use_clock or not reset_after:
            utc = datetime.timezone.utc
            now = datetime.datetime.now(utc)
            reset = datetime.datetime.fromtimestamp(float(headers['X-Ratelimit-Reset']), utc)
            self.reset_after = (reset - now).total_seconds()
        else:
            self.reset_after = float(reset_after)

        self.expires = self._loop.time() + self.reset_after

    def _wake_next(self) -> None:
        while self._pending_requests:
            future = self._pending_requests.popleft()
            if not future.done():
                future.set_result(None)
                break

    def _wake(self, count: int = 1, *, exception: Optional[RateLimited] = None) -> None:
        awaken = 0
        while self._pending_requests:
            future = self._pending_requests.popleft()
            if not future.done():
                if exception:
                    future.set_exception(exception)
                else:
                    future.set_result(None)
                awaken += 1

            if awaken >= count:
                break

    async def _refresh(self) -> None:
        error = self._max_ratelimit_timeout and self.reset_after > self._max_ratelimit_timeout
        exception = RateLimited(self.reset_after) if error else None
        async with self._sleeping:
            if not error:
                await asyncio.sleep(self.reset_after)

        self.reset()
        self._wake(self.remaining, exception=exception)

    def is_expired(self) -> bool:
        return self.expires is not None and self._loop.time() > self.expires

    def is_inactive(self) -> bool:
        delta = self._loop.time() - self._last_request
        return delta >= 300 and self.outgoing == 0 and len(self._pending_requests) == 0

    async def acquire(self) -> None:
        self._last_request = self._loop.time()
        if self.is_expired():
            self.reset()

        if self._max_ratelimit_timeout is not None and self.expires is not None:
            current_reset_after = self.expires - self._loop.time()
            if current_reset_after > self._max_ratelimit_timeout:
                raise RateLimited(current_reset_after)

        while self.remaining <= 0:
            future = self._loop.create_future()
            self._pending_requests.append(future)
            try:
                await future
            except:
                future.cancel()
                if self.remaining >= 0 and not future.cancelled():
                    self._wake_next()
                raise

        self.remaining -= 1
        self.outgoing += 1

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Type[BE], exc_val: BE, exc_tb: TracebackType) -> None:
        self.outgoing -= 1
        tokens = self.remaining - self.outgoing
        if not self._sleeping.locked():
            if tokens <= 0:
                await self._refresh()
            elif self._pending_requests:
                exception = (
                    RateLimited(self.reset_after)
                    if self._max_ratelimit_timeout and self.reset_after > self._max_ratelimit_timeout
                    else None
                )
                self._wake(tokens, exception=exception)


# For some reason, the Discord voice websocket expects this header to be
# completely lowercase while aiohttp respects spec and does it as case-insensitive
aiohttp.hdrs.WEBSOCKET = 'websocket'


class HTTPClient:
    """Represents an HTTP client sending HTTP requests to the Discord API."""

    SUCCESS_LOG = '{method} {url} has received {text}'
    REQUEST_LOG = '{method} {url} with {json} has returned {status}'

    def __init__(self,
        connector: Optional[aiohttp.BaseConnector] = None,
        *,
        proxy: Optional[str] = None,
        proxy_auth: Optional[aiohttp.BasicAuth] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        unsync_clock: bool = True,
        http_trace: Optional[aiohttp.TraceConfig] = None,
        max_ratelimit_timeout: Optional[float] = None,
    ) -> None:
        self.loop: asyncio.AbstractEventLoop = asyncio.get_event_loop() if loop is None else loop
        self.connector: aiohttp.BaseConnector = connector or MISSING
        self.__session: aiohttp.ClientSession = MISSING # filled in static_login
        # Route key -> Bucket hash
        self._bucket_hashes: Dict[str, str] = {}
        # Bucket Hash + Major Parameters -> Rate Limit
        # or
        # Route key + Major Paramters -> Rate Limit
        # When the key is the latter, it is used for temporary
        # one shot requests that don't have a bucket hash
        # When this reaches 26 elements, it will try to evict based off of expiry
        self._buckets: Dict[str, Ratelimit] = {}
        self._locks = weakref.WeakValueDictionary()
        self._global_over: asyncio.Event = asyncio.Event()
        self._global_over.set()
        self.token: Optional[str] = None
        self.bot_token: bool = False
        self.proxy: Optional[str] = proxy
        self.proxy_auth: Optional[aiohttp.BasicAuth] = proxy_auth
        self.use_clock: bool = not unsync_clock
        self.max_ratelimit_timeout: Optional[float] = max(30.0, max_ratelimit_timeout) if max_ratelimit_timeout else None
        self.http_trace: Optional[aiohttp.TraceConfig] = http_trace

        user_agent = 'DiscordBot (https://github.com/Rapptz/discord.py {0}) Python/{1[0]}.{1[1]} aiohttp/{2}'
        self.user_agent = user_agent.format(__version__, sys.version_info, aiohttp.__version__)
        
        if not INTERNAL_API_BASE.startswith("https://discord.com/api"):
            self.request = self.request_without_ratelimiter
    

    def recreate(self):
        if self.__session.closed:
            self.__session = aiohttp.ClientSession(connector=self.connector, ws_response_class=DiscordClientWebSocketResponse)

    async def ws_connect(self, url, *, compress=0):
        kwargs = {
            'proxy_auth': self.proxy_auth,
            'proxy': self.proxy,
            'max_msg_size': 0,
            'timeout': 30.0,
            'autoclose': False,
            'headers': {
                'User-Agent': self.user_agent,
            },
            'compress': compress
        }

        return await self.__session.ws_connect(url, **kwargs)
    
    def _try_clear_expired_ratelimits(self) -> None:
        if len(self._buckets) < 256:
            return
        
        keys = [key for key, bucket in self._buckets.items() if bucket.is_inactive()]
        for key in keys:
            del self._buckets[key]
            
    def get_ratelimit(self, key: str) -> Ratelimit:
        try:
            value = self._buckets[key]
        except KeyError:
            self._buckets[key] = value = Ratelimit(self.max_ratelimit_timeout)
            self._try_clear_expired_ratelimits()
        return value

    async def request(
        self, 
        route: Route, 
        *, 
        files: Optional[Sequence[File]] = None, 
        form: Optional[Iterable[Dict[str, Any]]] = None, 
        **kwargs: Any,
    ) -> Any:
        bucket = route.bucket
        method = route.method
        url = route.url

        lock = self._locks.get(bucket)
        if lock is None:
            lock = asyncio.Lock()
            if bucket is not None:
                self._locks[bucket] = lock

        # header creation
        headers = {
            'User-Agent': self.user_agent,
            'X-Ratelimit-Precision': 'millisecond',
        }

        if self.token is not None:
            headers['Authorization'] = 'Bot ' + self.token if self.bot_token else self.token
        # some checking if it's a JSON request
        if 'json' in kwargs:
            headers['Content-Type'] = 'application/json'
            kwargs['data'] = utils.to_json(kwargs.pop('json'))

        try:
            reason = kwargs.pop('reason')
        except KeyError:
            pass
        else:
            if reason:
                headers['X-Audit-Log-Reason'] = _uriquote(reason, safe='/ ')

        kwargs['headers'] = headers

        # Proxy support
        if self.proxy is not None:
            kwargs['proxy'] = self.proxy
        if self.proxy_auth is not None:
            kwargs['proxy_auth'] = self.proxy_auth

        if not self._global_over.is_set():
            # wait until the global lock is complete
            await self._global_over.wait()

        await lock.acquire()
        with MaybeUnlock(lock) as maybe_lock:
            for tries in range(5):
                if files:
                    for f in files:
                        f.reset(seek=tries)

                if form:
                    form_data = aiohttp.FormData()
                    for params in form:
                        form_data.add_field(**params)
                    kwargs['data'] = form_data

                try:
                    async with self.__session.request(method, url, **kwargs) as r:
                        log.debug('%s %s with %s has returned %s', method, url, kwargs.get('data'), r.status)

                        # even errors have text involved in them so this is safe to call
                        data = await json_or_text(r)

                        # check if we have rate limit header information
                        remaining = r.headers.get('X-Ratelimit-Remaining')
                        if remaining == '0' and r.status != 429:
                            # we've depleted our current bucket
                            delta = utils._parse_ratelimit_header(r, use_clock=self.use_clock)
                            log.debug('A rate limit bucket has been exhausted (bucket: %s, retry: %s).', bucket, delta)
                            maybe_lock.defer()
                            self.loop.call_later(delta, lock.release)

                        # the request was successful so just return the text/json
                        if 300 > r.status >= 200:
                            log.debug('%s %s has received %s', method, url, data)
                            return data

                        # we are being rate limited
                        if r.status == 429:
                            if not r.headers.get('Via'):
                                # Banned by Cloudflare more than likely.
                                raise HTTPException(r, data)

                            fmt = 'We are being rate limited. Retrying in %.2f seconds. Handled under the bucket "%s"'

                            # sleep a bit
                            retry_after = data['retry_after'] / 1000.0
                            log.warning(fmt, retry_after, bucket)

                            # check if it's a global rate limit
                            is_global = data.get('global', False)
                            if is_global:
                                log.warning('Global rate limit has been hit. Retrying in %.2f seconds.', retry_after)
                                self._global_over.clear()

                            await asyncio.sleep(retry_after)
                            log.debug('Done sleeping for the rate limit. Retrying...')

                            # release the global lock now that the
                            # global rate limit has passed
                            if is_global:
                                self._global_over.set()
                                log.debug('Global rate limit is now over.')

                            continue

                        # we've received a 500 or 502, unconditional retry
                        if r.status in {500, 502}:
                            await asyncio.sleep(1 + tries * 2)
                            continue

                        # the usual error cases
                        if r.status == 403:
                            raise Forbidden(r, data)
                        elif r.status == 404:
                            raise NotFound(r, data)
                        elif r.status == 503:
                            raise DiscordServerError(r, data)
                        else:
                            raise HTTPException(r, data)

                # This is handling exceptions from the request
                except OSError as e:
                    # Connection reset by peer
                    if tries < 4 and e.errno in (54, 10054):
                        continue
                    raise

            # We've run out of retries, raise.
            if r.status >= 500:
                raise DiscordServerError(r, data)

            raise HTTPException(r, data)
        
    async def request_without_ratelimiter(
        self,
        route: Route,
        *,
        files: Optional[Sequence[File]] = None,
        form: Optional[Iterable[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        method = route.method
        url = route.url
        
        headers: Dict[str, str] = {
            'User-Agent': self.user_agent,
        }
        
        if 'json' in kwargs:
            headers['Content-Type'] = 'application/json'
            kwargs['data'] = utils._to_json(kwargs.pop('json'))
            
        try:
            reason = kwargs.pop('reason')
        except KeyError:
            pass
        else:
            if reason:
                headers['X-Audit-Log-Reason'] = _uriquote(reason, safe='/ ')
                
                kwargs['headers'] = headers
                
                if self.proxy is not None:
                    kwargs['proxy'] = self.proxy
                if self.proxy_auth is not None:
                    kwargs['proxy_auth'] = self.proxy_auth
                    
                response: Optional[aiohttp.ClientResponse] = None
                data: Optional[Union[Dict[str, Any], str]] = None
                for tries in range(5):
                    if files:
                        for f in files:
                            f.reset(seek=tries)
                            
                    if form:
                        form_data = aiohttp.FormData(quote_fields=False)
                        for params in form:
                            form_data.add_field(**params)
                        kwargs['data'] = form_data
                        
                    try:
                        async with self.__session.request(method, url, **kwargs) as response:
                            log.debug('%s %s with %s has returned %s', method, url, kwargs.get('data'), response.status)
                            
                            data = await json_or_text(response)
                            
                            if 300 > response.status >= 200:
                                log.debug('%s %s has recieved %s', method, url, data)
                                return data
                            
                            if response.status == 429:
                                if not response.headers.get('Via') or isinstance(data, str):
                                    raise HTTPException(response, data)
                                
                                if data.get('code'):
                                    await asyncio.sleep(data['retry_after'])
                                    
                                continue
                            
                            if response.status == 403:
                                raise Forbidden(response, data)
                            elif response.status = 404:
                                raise NotFound(response, data)
                            elif response.status >= 500:
                                raise DiscordServerError(response, data)
                            else:
                                raise HTTPException(response, data)
                            
                    except OSError as e:
                        if tries < 4 and e.errno in (54, 10054):
                            await asyncio.sleep(1 + tries * 2)
                            continue
                        raise
                    
                if response is not None:
                    if response.status >= 500:
                        raise DiscordServerError(response, data)
                    
                    raise HTTPException(response, data)
                
                raise RuntimeError('Unreachable code in HTTP handling.')

    async def get_from_cdn(self, url):
        async with self.__session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()
            elif resp.status == 404:
                raise NotFound(resp, 'asset not found')
            elif resp.status == 403:
                raise Forbidden(resp, 'cannot retrieve asset')
            else:
                raise HTTPException(resp, 'failed to get asset')

    # state management

    async def close(self):
        if self.__session:
            await self.__session.close()

    def _token(self, token, *, bot=True):
        self.token = token
        self.bot_token = bot
        self._ack_token = None

    # login management

    async def static_login(self, token: str, *, bot):
        # Necessary to get aiohttp to stop complaining about session creation
        if self.connector is MISSING:
            self.connector = aiohttp.TCPConnector(limit=0)
            
        self.__session = aiohttp.ClientSession(
            connector=self.connector, 
            ws_response_class=DiscordClientWebSocketResponse,
            trace_configs=None if self.http_trace is None else [self.http_trace]
        )
        old_token, old_bot = self.token, self.bot_token
        self._token(token, bot=bot)

        try:
            data = await self.request(Route('GET', '/users/@me'))
        except HTTPException as exc:
            self._token(old_token, bot=old_bot)
            if exc.response.status == 401:
                raise LoginFailure('Improper token has been passed.') from exc
            raise

        return data

    def logout(self):
        return self.request(Route('POST', '/auth/logout'))

    # Group functionality

    def start_group(self, user_id, recipients: List[int]):
        payload = {
            'recipients': recipients
        }

        return self.request(Route('POST', '/users/{user_id}/channels', user_id=user_id), json=payload)

    def leave_group(self, channel_id) -> Response[None]:
        return self.request(Route('DELETE', '/channels/{channel_id}', channel_id=channel_id))

    def add_group_recipient(self, channel_id, user_id):
        r = Route('PUT', '/channels/{channel_id}/recipients/{user_id}', channel_id=channel_id, user_id=user_id)
        return self.request(r)

    def remove_group_recipient(self, channel_id, user_id):
        r = Route('DELETE', '/channels/{channel_id}/recipients/{user_id}', channel_id=channel_id, user_id=user_id)
        return self.request(r)

    def edit_group(self, channel_id, **options):
        valid_keys = ('name', 'icon')
        payload = {
            k: v for k, v in options.items() if k in valid_keys
        }

        return self.request(Route('PATCH', '/channels/{channel_id}', channel_id=channel_id), json=payload)

    def convert_group(self, channel_id):
        return self.request(Route('POST', '/channels/{channel_id}/convert', channel_id=channel_id))

    # Message management

    def start_private_message(self, user_id):
        payload = {
            'recipient_id': user_id
        }

        return self.request(Route('POST', '/users/@me/channels'), json=payload)

    def send_message(self, channel_id, content, *, tts=False, embed=None, nonce=None, allowed_mentions=None, message_reference=None):
        r = Route('POST', '/channels/{channel_id}/messages', channel_id=channel_id)
        payload = {}

        if content:
            payload['content'] = content

        if tts:
            payload['tts'] = True

        if embed:
            payload['embed'] = embed

        if nonce:
            payload['nonce'] = nonce

        if allowed_mentions:
            payload['allowed_mentions'] = allowed_mentions

        if message_reference:
            payload['message_reference'] = message_reference

        return self.request(r, json=payload)

    def send_typing(self, channel_id) -> Response[None]:
        return self.request(Route('POST', '/channels/{channel_id}/typing', channel_id=channel_id))

    def send_files(self, channel_id, *, files, content=None, tts=False, embed=None, nonce=None, allowed_mentions=None, message_reference=None):
        r = Route('POST', '/channels/{channel_id}/messages', channel_id=channel_id)
        form = []

        payload = {'tts': tts}
        if content:
            payload['content'] = content
        if embed:
            payload['embed'] = embed
        if nonce:
            payload['nonce'] = nonce
        if allowed_mentions:
            payload['allowed_mentions'] = allowed_mentions
        if message_reference:
            payload['message_reference'] = message_reference

        form.append({'name': 'payload_json', 'value': utils.to_json(payload)})
        if len(files) == 1:
            file = files[0]
            form.append({
                'name': 'file',
                'value': file.fp,
                'filename': file.filename,
                'content_type': 'application/octet-stream'
            })
        else:
            for index, file in enumerate(files):
                form.append({
                    'name': 'file%s' % index,
                    'value': file.fp,
                    'filename': file.filename,
                    'content_type': 'application/octet-stream'
                })

        return self.request(r, form=form, files=files)

    async def ack_message(self, channel_id, message_id):
        r = Route('POST', '/channels/{channel_id}/messages/{message_id}/ack', channel_id=channel_id, message_id=message_id)
        data = await self.request(r, json={'token': self._ack_token})
        self._ack_token = data['token']

    def ack_guild(self, guild_id):
        return self.request(Route('POST', '/guilds/{guild_id}/ack', guild_id=guild_id))

    def delete_message(self, channel_id, message_id, *, reason=None):
        r = Route('DELETE', '/channels/{channel_id}/messages/{message_id}', channel_id=channel_id, message_id=message_id)
        return self.request(r, reason=reason)

    def delete_messages(self, channel_id, message_ids, *, reason=None):
        r = Route('POST', '/channels/{channel_id}/messages/bulk_delete', channel_id=channel_id)
        payload = {
            'messages': message_ids
        }

        return self.request(r, json=payload, reason=reason)

    def edit_message(self, channel_id, message_id, **fields):
        r = Route('PATCH', '/channels/{channel_id}/messages/{message_id}', channel_id=channel_id, message_id=message_id)
        return self.request(r, json=fields)

    def add_reaction(self, channel_id, message_id, emoji):
        r = Route('PUT', '/channels/{channel_id}/messages/{message_id}/reactions/{emoji}/@me',
                  channel_id=channel_id, message_id=message_id, emoji=emoji)
        return self.request(r)

    def remove_reaction(self, channel_id, message_id, emoji, member_id):
        r = Route('DELETE', '/channels/{channel_id}/messages/{message_id}/reactions/{emoji}/{member_id}',
                  channel_id=channel_id, message_id=message_id, member_id=member_id, emoji=emoji)
        return self.request(r)

    def remove_own_reaction(self, channel_id, message_id, emoji):
        r = Route('DELETE', '/channels/{channel_id}/messages/{message_id}/reactions/{emoji}/@me',
                  channel_id=channel_id, message_id=message_id, emoji=emoji)
        return self.request(r)

    def get_reaction_users(self, channel_id, message_id, emoji, limit, after=None):
        r = Route('GET', '/channels/{channel_id}/messages/{message_id}/reactions/{emoji}',
                  channel_id=channel_id, message_id=message_id, emoji=emoji)

        params = {'limit': limit}
        if after:
            params['after'] = after
        return self.request(r, params=params)

    def clear_reactions(self, channel_id, message_id):
        r = Route('DELETE', '/channels/{channel_id}/messages/{message_id}/reactions',
                  channel_id=channel_id, message_id=message_id)

        return self.request(r)

    def clear_single_reaction(self, channel_id, message_id, emoji):
        r = Route('DELETE', '/channels/{channel_id}/messages/{message_id}/reactions/{emoji}',
                   channel_id=channel_id, message_id=message_id, emoji=emoji)
        return self.request(r)

    def get_message(self, channel_id, message_id):
        r = Route('GET', '/channels/{channel_id}/messages/{message_id}', channel_id=channel_id, message_id=message_id)
        return self.request(r)

    def get_channel(self, channel_id):
        r = Route('GET', '/channels/{channel_id}', channel_id=channel_id)
        return self.request(r)

    def logs_from(self, channel_id, limit, before=None, after=None, around=None):
        params = {
            'limit': limit
        }

        if before is not None:
            params['before'] = before
        if after is not None:
            params['after'] = after
        if around is not None:
            params['around'] = around

        return self.request(Route('GET', '/channels/{channel_id}/messages', channel_id=channel_id), params=params)

    def publish_message(self, channel_id, message_id):
        return self.request(Route('POST', '/channels/{channel_id}/messages/{message_id}/crosspost',
                                  channel_id=channel_id, message_id=message_id))

    def pin_message(self, channel_id, message_id, reason=None):
        return self.request(Route('PUT', '/channels/{channel_id}/pins/{message_id}',
                                  channel_id=channel_id, message_id=message_id), reason=reason)

    def unpin_message(self, channel_id, message_id, reason=None):
        return self.request(Route('DELETE', '/channels/{channel_id}/pins/{message_id}',
                                  channel_id=channel_id, message_id=message_id), reason=reason)

    def pins_from(self, channel_id):
        return self.request(Route('GET', '/channels/{channel_id}/pins', channel_id=channel_id))

    # Member management

    def kick(self, user_id, guild_id, reason=None):
        r = Route('DELETE', '/guilds/{guild_id}/members/{user_id}', guild_id=guild_id, user_id=user_id)
        if reason:
            # thanks aiohttp
            r.url = '{0.url}?reason={1}'.format(r, _uriquote(reason))

        return self.request(r)

    def ban(self, user_id, guild_id, delete_message_days: int = 1, reason: Optional[str] = None) -> Response[None]:
        r = Route('PUT', '/guilds/{guild_id}/bans/{user_id}', guild_id=guild_id, user_id=user_id)
        params = {
            'delete_message_days': delete_message_days,
        }

        if reason:
            # thanks aiohttp
            r.url = '{0.url}?reason={1}'.format(r, _uriquote(reason))

        return self.request(r, params=params)

    def unban(self, user_id, guild_id, *, reason: Optional[str] = None):
        r = Route('DELETE', '/guilds/{guild_id}/bans/{user_id}', guild_id=guild_id, user_id=user_id)
        return self.request(r, reason=reason)

    def guild_voice_state(self, user_id, guild_id, *, mute=None, deafen=None, reason=None):
        r = Route('PATCH', '/guilds/{guild_id}/members/{user_id}', guild_id=guild_id, user_id=user_id)
        payload = {}
        if mute is not None:
            payload['mute'] = mute

        if deafen is not None:
            payload['deaf'] = deafen

        return self.request(r, json=payload, reason=reason)

    def edit_profile(self, password, username, avatar, **fields):
        payload = {
            'password': password,
            'username': username,
            'avatar': avatar
        }

        if 'email' in fields:
            payload['email'] = fields['email']

        if 'new_password' in fields:
            payload['new_password'] = fields['new_password']

        return self.request(Route('PATCH', '/users/@me'), json=payload)

    def change_my_nickname(self, guild_id, nickname, *, reason=None):
        r = Route('PATCH', '/guilds/{guild_id}/members/@me/nick', guild_id=guild_id)
        payload = {
            'nick': nickname
        }
        return self.request(r, json=payload, reason=reason)

    def change_nickname(self, guild_id, user_id, nickname, *, reason=None):
        r = Route('PATCH', '/guilds/{guild_id}/members/{user_id}', guild_id=guild_id, user_id=user_id)
        payload = {
            'nick': nickname
        }
        return self.request(r, json=payload, reason=reason)

    def edit_my_voice_state(self, guild_id, payload):
        r = Route('PATCH', '/guilds/{guild_id}/voice-states/@me', guild_id=guild_id)
        return self.request(r, json=payload)

    def edit_voice_state(self, guild_id, user_id, payload):
        r = Route('PATCH', '/guilds/{guild_id}/voice-states/{user_id}', guild_id=guild_id, user_id=user_id)
        return self.request(r, json=payload)

    def edit_member(self, guild_id, user_id, *, reason=None, **fields):
        r = Route('PATCH', '/guilds/{guild_id}/members/{user_id}', guild_id=guild_id, user_id=user_id)
        return self.request(r, json=fields, reason=reason)

    # Channel management

    def edit_channel(self, channel_id, *, reason=None, **options):
        r = Route('PATCH', '/channels/{channel_id}', channel_id=channel_id)
        valid_keys = ('name', 'parent_id', 'topic', 'bitrate', 'nsfw',
                      'user_limit', 'position', 'permission_overwrites', 'rate_limit_per_user',
                      'type', 'rtc_region')
        payload = {
            k: v for k, v in options.items() if k in valid_keys
        }
        return self.request(r, reason=reason, json=payload)

    def bulk_channel_update(self, guild_id, data, *, reason=None):
        r = Route('PATCH', '/guilds/{guild_id}/channels', guild_id=guild_id)
        return self.request(r, json=data, reason=reason)

    def create_channel(self, guild_id, channel_type, *, reason=None, **options):
        payload = {
            'type': channel_type
        }

        valid_keys = ('name', 'parent_id', 'topic', 'bitrate', 'nsfw',
                      'user_limit', 'position', 'permission_overwrites', 'rate_limit_per_user',
                      'rtc_region')
        payload.update({
            k: v for k, v in options.items() if k in valid_keys and v is not None
        })

        return self.request(Route('POST', '/guilds/{guild_id}/channels', guild_id=guild_id), json=payload, reason=reason)

    def delete_channel(self, channel_id, *, reason=None):
        return self.request(Route('DELETE', '/channels/{channel_id}', channel_id=channel_id), reason=reason)

    # Webhook management

    def create_webhook(self, channel_id, *, name, avatar=None, reason=None):
        payload = {
            'name': name
        }
        if avatar is not None:
            payload['avatar'] = avatar

        r = Route('POST', '/channels/{channel_id}/webhooks', channel_id=channel_id)
        return self.request(r, json=payload, reason=reason)

    def channel_webhooks(self, channel_id):
        return self.request(Route('GET', '/channels/{channel_id}/webhooks', channel_id=channel_id))

    def guild_webhooks(self, guild_id):
        return self.request(Route('GET', '/guilds/{guild_id}/webhooks', guild_id=guild_id))

    def get_webhook(self, webhook_id):
        return self.request(Route('GET', '/webhooks/{webhook_id}', webhook_id=webhook_id))

    def follow_webhook(self, channel_id, webhook_channel_id, reason=None):
        payload = {
            'webhook_channel_id': str(webhook_channel_id)
        }
        return self.request(Route('POST', '/channels/{channel_id}/followers', channel_id=channel_id), json=payload, reason=reason)

    # Guild management

    def get_guilds(self, limit, before=None, after=None):
        params = {
            'limit': limit
        }

        if before:
            params['before'] = before
        if after:
            params['after'] = after

        return self.request(Route('GET', '/users/@me/guilds'), params=params)

    def leave_guild(self, guild_id):
        return self.request(Route('DELETE', '/users/@me/guilds/{guild_id}', guild_id=guild_id))

    def get_guild(self, guild_id):
        return self.request(Route('GET', '/guilds/{guild_id}', guild_id=guild_id))

    def delete_guild(self, guild_id):
        return self.request(Route('DELETE', '/guilds/{guild_id}', guild_id=guild_id))

    def create_guild(self, name, region, icon):
        payload = {
            'name': name,
            'icon': icon,
            'region': region
        }

        return self.request(Route('POST', '/guilds'), json=payload)

    def edit_guild(self, guild_id, *, reason=None, **fields):
        valid_keys = (
            'name', 'region', 'icon', 'afk_timeout', 'owner_id',
            'afk_channel_id', 'splash', 'verification_level',
            'system_channel_id', 'default_message_notifications',
            'description', 'explicit_content_filter', 'banner',
            'system_channel_flags', 'rules_channel_id',
            'public_updates_channel_id', 'preferred_locale',
        )

        payload = {
            k: v for k, v in fields.items() if k in valid_keys
        }

        return self.request(Route('PATCH', '/guilds/{guild_id}', guild_id=guild_id), json=payload, reason=reason)

    def get_template(self, code):
        return self.request(Route('GET', '/guilds/templates/{code}', code=code))

    def guild_templates(self, guild_id):
        return self.request(Route('GET', '/guilds/{guild_id}/templates', guild_id=guild_id))

    def create_template(self, guild_id, payload):
        return self.request(Route('POST', '/guilds/{guild_id}/templates', guild_id=guild_id), json=payload)

    def sync_template(self, guild_id, code):
        return self.request(Route('PUT', '/guilds/{guild_id}/templates/{code}', guild_id=guild_id, code=code))

    def edit_template(self, guild_id, code, payload):
        valid_keys = (
            'name',
            'description',
        )
        payload = {
            k: v for k, v in payload.items() if k in valid_keys
        }
        return self.request(Route('PATCH', '/guilds/{guild_id}/templates/{code}', guild_id=guild_id, code=code), json=payload)

    def delete_template(self, guild_id, code):
        return self.request(Route('DELETE', '/guilds/{guild_id}/templates/{code}', guild_id=guild_id, code=code))

    def create_from_template(self, code, name, region, icon):
        payload = {
            'name': name,
            'icon': icon,
            'region': region
        }
        return self.request(Route('POST', '/guilds/templates/{code}', code=code), json=payload)

    def get_bans(self, guild_id):
        return self.request(Route('GET', '/guilds/{guild_id}/bans', guild_id=guild_id))

    def get_ban(self, user_id, guild_id):
        return self.request(Route('GET', '/guilds/{guild_id}/bans/{user_id}', guild_id=guild_id, user_id=user_id))

    def get_vanity_code(self, guild_id):
        return self.request(Route('GET', '/guilds/{guild_id}/vanity-url', guild_id=guild_id))

    def change_vanity_code(self, guild_id, code, *, reason=None):
        payload = {'code': code}
        return self.request(Route('PATCH', '/guilds/{guild_id}/vanity-url', guild_id=guild_id), json=payload, reason=reason)

    def get_all_guild_channels(self, guild_id):
        return self.request(Route('GET', '/guilds/{guild_id}/channels', guild_id=guild_id))

    def get_members(self, guild_id, limit, after):
        params = {
            'limit': limit,
        }
        if after:
            params['after'] = after

        r = Route('GET', '/guilds/{guild_id}/members', guild_id=guild_id)
        return self.request(r, params=params)

    def get_member(self, guild_id, member_id):
        return self.request(Route('GET', '/guilds/{guild_id}/members/{member_id}', guild_id=guild_id, member_id=member_id))

    def prune_members(self, guild_id, days, compute_prune_count, roles, *, reason=None):
        payload = {
            'days': days,
            'compute_prune_count': 'true' if compute_prune_count else 'false'
        }
        if roles:
            payload['include_roles'] = ', '.join(roles)

        return self.request(Route('POST', '/guilds/{guild_id}/prune', guild_id=guild_id), json=payload, reason=reason)

    def estimate_pruned_members(self, guild_id, days, roles):
        params = {
            'days': days
        }
        if roles:
            params['include_roles'] = ', '.join(roles)

        return self.request(Route('GET', '/guilds/{guild_id}/prune', guild_id=guild_id), params=params)

    def get_all_custom_emojis(self, guild_id):
        return self.request(Route('GET', '/guilds/{guild_id}/emojis', guild_id=guild_id))

    def get_custom_emoji(self, guild_id, emoji_id):
        return self.request(Route('GET', '/guilds/{guild_id}/emojis/{emoji_id}', guild_id=guild_id, emoji_id=emoji_id))

    def create_custom_emoji(self, guild_id, name, image, *, roles=None, reason=None):
        payload = {
            'name': name,
            'image': image,
            'roles': roles or []
        }

        r = Route('POST', '/guilds/{guild_id}/emojis', guild_id=guild_id)
        return self.request(r, json=payload, reason=reason)

    def delete_custom_emoji(self, guild_id, emoji_id, *, reason=None):
        r = Route('DELETE', '/guilds/{guild_id}/emojis/{emoji_id}', guild_id=guild_id, emoji_id=emoji_id)
        return self.request(r, reason=reason)

    def edit_custom_emoji(self, guild_id, emoji_id, *, name, roles=None, reason=None):
        payload = {
            'name': name,
            'roles': roles or []
        }
        r = Route('PATCH', '/guilds/{guild_id}/emojis/{emoji_id}', guild_id=guild_id, emoji_id=emoji_id)
        return self.request(r, json=payload, reason=reason)

    def get_all_integrations(self, guild_id):
        r = Route('GET', '/guilds/{guild_id}/integrations', guild_id=guild_id)

        return self.request(r)

    def create_integration(self, guild_id, type, id):
        payload = {
            'type': type,
            'id': id
        }

        r = Route('POST', '/guilds/{guild_id}/integrations', guild_id=guild_id)
        return self.request(r, json=payload)

    def edit_integration(self, guild_id, integration_id, **payload):
        r = Route('PATCH', '/guilds/{guild_id}/integrations/{integration_id}', guild_id=guild_id,
                  integration_id=integration_id)

        return self.request(r, json=payload)

    def sync_integration(self, guild_id, integration_id):
        r = Route('POST', '/guilds/{guild_id}/integrations/{integration_id}/sync', guild_id=guild_id,
                  integration_id=integration_id)

        return self.request(r)

    def delete_integration(self, guild_id, integration_id):
        r = Route('DELETE', '/guilds/{guild_id}/integrations/{integration_id}', guild_id=guild_id,
                  integration_id=integration_id)

        return self.request(r)

    def get_audit_logs(self, guild_id, limit=100, before=None, after=None, user_id=None, action_type=None):
        params = {'limit': limit}
        if before:
            params['before'] = before
        if after:
            params['after'] = after
        if user_id:
            params['user_id'] = user_id
        if action_type:
            params['action_type'] = action_type

        r = Route('GET', '/guilds/{guild_id}/audit-logs', guild_id=guild_id)
        return self.request(r, params=params)

    def get_widget(self, guild_id):
        return self.request(Route('GET', '/guilds/{guild_id}/widget.json', guild_id=guild_id))

    # Invite management

    def create_invite(self, channel_id, *, reason=None, **options):
        r = Route('POST', '/channels/{channel_id}/invites', channel_id=channel_id)
        payload = {
            'max_age': options.get('max_age', 0),
            'max_uses': options.get('max_uses', 0),
            'temporary': options.get('temporary', False),
            'unique': options.get('unique', True)
        }

        return self.request(r, reason=reason, json=payload)

    def get_invite(self, invite_id, *, with_counts=True):
        params = {
            'with_counts': int(with_counts)
        }
        return self.request(Route('GET', '/invites/{invite_id}', invite_id=invite_id), params=params)

    def invites_from(self, guild_id):
        return self.request(Route('GET', '/guilds/{guild_id}/invites', guild_id=guild_id))

    def invites_from_channel(self, channel_id):
        return self.request(Route('GET', '/channels/{channel_id}/invites', channel_id=channel_id))

    def delete_invite(self, invite_id, *, reason=None):
        return self.request(Route('DELETE', '/invites/{invite_id}', invite_id=invite_id), reason=reason)

    # Role management

    def get_roles(self, guild_id):
        return self.request(Route('GET', '/guilds/{guild_id}/roles', guild_id=guild_id))

    def edit_role(self, guild_id, role_id, *, reason=None, **fields):
        r = Route('PATCH', '/guilds/{guild_id}/roles/{role_id}', guild_id=guild_id, role_id=role_id)
        valid_keys = ('name', 'permissions', 'color', 'hoist', 'mentionable')
        payload = {
            k: v for k, v in fields.items() if k in valid_keys
        }
        return self.request(r, json=payload, reason=reason)

    def delete_role(self, guild_id, role_id, *, reason=None):
        r = Route('DELETE', '/guilds/{guild_id}/roles/{role_id}', guild_id=guild_id, role_id=role_id)
        return self.request(r, reason=reason)

    def replace_roles(self, user_id, guild_id, role_ids, *, reason=None):
        return self.edit_member(guild_id=guild_id, user_id=user_id, roles=role_ids, reason=reason)

    def create_role(self, guild_id, *, reason=None, **fields):
        r = Route('POST', '/guilds/{guild_id}/roles', guild_id=guild_id)
        return self.request(r, json=fields, reason=reason)

    def move_role_position(self, guild_id, positions, *, reason=None):
        r = Route('PATCH', '/guilds/{guild_id}/roles', guild_id=guild_id)
        return self.request(r, json=positions, reason=reason)

    def add_role(self, guild_id, user_id, role_id, *, reason=None):
        r = Route('PUT', '/guilds/{guild_id}/members/{user_id}/roles/{role_id}',
                  guild_id=guild_id, user_id=user_id, role_id=role_id)
        return self.request(r, reason=reason)

    def remove_role(self, guild_id, user_id, role_id, *, reason=None):
        r = Route('DELETE', '/guilds/{guild_id}/members/{user_id}/roles/{role_id}',
                  guild_id=guild_id, user_id=user_id, role_id=role_id)
        return self.request(r, reason=reason)

    def edit_channel_permissions(self, channel_id, target, allow, deny, type, *, reason=None):
        payload = {
            'id': target,
            'allow': allow,
            'deny': deny,
            'type': type
        }
        r = Route('PUT', '/channels/{channel_id}/permissions/{target}', channel_id=channel_id, target=target)
        return self.request(r, json=payload, reason=reason)

    def delete_channel_permissions(self, channel_id, target, *, reason=None):
        r = Route('DELETE', '/channels/{channel_id}/permissions/{target}', channel_id=channel_id, target=target)
        return self.request(r, reason=reason)

    # Voice management

    def move_member(self, user_id, guild_id, channel_id, *, reason=None):
        return self.edit_member(guild_id=guild_id, user_id=user_id, channel_id=channel_id, reason=reason)

    # Relationship related

    def remove_relationship(self, user_id):
        r = Route('DELETE', '/users/@me/relationships/{user_id}', user_id=user_id)
        return self.request(r)

    def add_relationship(self, user_id, type=None):
        r = Route('PUT', '/users/@me/relationships/{user_id}', user_id=user_id)
        payload = {}
        if type is not None:
            payload['type'] = type

        return self.request(r, json=payload)

    def send_friend_request(self, username, discriminator):
        r = Route('POST', '/users/@me/relationships')
        payload = {
            'username': username,
            'discriminator': int(discriminator)
        }
        return self.request(r, json=payload)

    # Misc

    def application_info(self):
        return self.request(Route('GET', '/oauth2/applications/@me'))

    async def get_gateway(self, *, encoding='json', v=6, zlib=True):
        try:
            data = await self.request(Route('GET', '/gateway'))
        except HTTPException as exc:
            raise GatewayNotFound() from exc
        if zlib:
            value = '{0}?encoding={1}&v={2}&compress=zlib-stream'
        else:
            value = '{0}?encoding={1}&v={2}&compress='
        return value.format(data['url'], encoding, v)

    async def get_bot_gateway(self, *, encoding='json', v=6, zlib=True):
        try:
            data = await self.request(Route('GET', '/gateway/bot'))
        except HTTPException as exc:
            raise GatewayNotFound() from exc

        if zlib:
            value = '{0}?encoding={1}&v={2}&compress=zlib-stream'
        else:
            value = '{0}?encoding={1}&v={2}&compress='
        return data['shards'], value.format(data['url'], encoding, v)

    def get_user(self, user_id):
        return self.request(Route('GET', '/users/{user_id}', user_id=user_id))

    def get_user_profile(self, user_id):
        return self.request(Route('GET', '/users/{user_id}/profile', user_id=user_id))

    def get_mutual_friends(self, user_id):
        return self.request(Route('GET', '/users/{user_id}/relationships', user_id=user_id))

    def change_hypesquad_house(self, house_id):
        payload = {'house_id': house_id}
        return self.request(Route('POST', '/hypesquad/online'), json=payload)

    def leave_hypesquad_house(self):
        return self.request(Route('DELETE', '/hypesquad/online'))

    def edit_settings(self, **payload):
        return self.request(Route('PATCH', '/users/@me/settings'), json=payload)
