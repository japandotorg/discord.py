URL = r"\w+:\/\/((?:[\w_.-]+@)?[\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"

HTTP_URL = r"https?:\/\/((?:[\w_.-]+@)?[\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"

def url_with_protocols(*protocols: str) -> str:
    """"
    Return a regex URL that will match one of the specified protocols.
    
    Returns:
        str: The resulting regex pattern string.
    """
    if not protocols:
        raise TypeError("A protocol string must be provided.")
    return rf"({'|'.join(protocols)}){URL[3]}"

CODE_BLOCK = r"```([^`\s]*)\n(((?!```).|\s|(?<=\\)```)+)```"

INLINE_CODE_BLOCK = r"`((?:(?<=\\)`|[^`\n])+)`"

USER_ROLE_MENTION = r"<@[&!]?(\d+)>"

USER_ROLE_CHANNEL_MENTION = r"<(?:@[&!]?|\#)(\d+)>"

USER_MENTION = r"<@!?(\d+)>"

ROLE_MENTION = r"<@&(\d+)>"

CHANNEL_MENTION = r"<#(\d+)>"

CUSTOM_EMOJI = r"<(a?):(\S+):(\d+)>"

EMOJI_SHORTCODE = r"\s*:(\S+):\s*"

UNIX_TIMESTAMP = r"<t:(-?\d+)(?::([tTdDfFR]))?>"
